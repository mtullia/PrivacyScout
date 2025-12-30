// popup.js — Privacy Risk Scout
// Coordinates:
//   popup  -> content-script (PRS_EXTRACT_TEXT) for text + policy candidates
//   popup  -> background      (PRS_ANALYZE_TEXT / PRS_ANALYZE_URL) for model inference

document.addEventListener("DOMContentLoaded", () => {
    const analyzeBtn = document.getElementById("prs-analyze");
    const statusDiv = document.getElementById("prs-status");
    const labelSpan = document.getElementById("prs-label");
    const scoreSpan = document.getElementById("prs-score");
    const probaList = document.getElementById("prs-proba-list");
    const keywordsList = document.getElementById("prs-keywords");
    const urlSpan = document.getElementById("prs-url");
    const policyPickerDiv = document.getElementById("prs-policy-picker");

    // State for two-step flow:
    //  - First click: scan the page and (a) analyze the current policy page, or
    //                 (b) show top-3 privacy links as radio buttons.
    //  - Second click (when in radio mode): analyze the selected policy URL.
    let awaitingPolicySelection = false;
    let lastPolicyCandidates = [];
    let lastSourceUrl = "";

    // Reset only the results area (do not clear status or radio picker)
    function resetResults() {
        labelSpan.textContent = "–";
        scoreSpan.textContent = "–";
        probaList.innerHTML = "";
        keywordsList.innerHTML = "";
        labelSpan.className = "prs-label-value";
    }

    // Helper: send message to active tab's content script
    function sendToContentScript(tabId, message) {
        return new Promise((resolve, reject) => {
            chrome.tabs.sendMessage(tabId, message, (response) => {
                const err = chrome.runtime.lastError;
                if (err) {
                    reject(new Error(err.message));
                } else {
                    resolve(response);
                }
            });
        });
    }

    // Background bridge: analyze plain text via backend
    function analyzeTextViaBackground(text) {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage(
                { type: "PRS_ANALYZE_TEXT", text },
                (response) => {
                    resolve(response);
                }
            );
        });
    }

    // Background bridge: fetch a policy URL and analyze its contents via backend
    function analyzeUrlViaBackground(url) {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage(
                { type: "PRS_ANALYZE_URL", url },
                (response) => {
                    resolve(response);
                }
            );
        });
    }

    // Render the radio button picker for top-3 privacy policy URLs
    function renderPolicyPicker(candidates) {
        if (!policyPickerDiv) return;

        if (!Array.isArray(candidates) || candidates.length === 0) {
            policyPickerDiv.innerHTML = "";
            return;
        }

        const top = candidates.slice(0, 3);
        let html = '<div class="prs-policy-picker-title">Select a privacy policy to analyze:</div>';

        top.forEach((c, index) => {
            const id = `prs-policy-radio-${index}`;
            const labelText = c.label || `Option ${index + 1}`;
            const url = c.url || "";
            const checkedAttr = index === 0 ? "checked" : "";

            html += `
                <label class="prs-policy-picker-item" for="${id}">
                    <input
                        type="radio"
                        name="prs-policy-radio"
                        id="${id}"
                        value="${url}"
                        ${checkedAttr}
                    />
                    ${labelText}
                    <div class="prs-policy-picker-url">${url}</div>
                </label>
            `;
        });

        policyPickerDiv.innerHTML = html;
    }

    // Render the analysis results from FastAPI
    function renderResult(data, source) {
        const label = data.label || "Unknown";
        const score =
            data.score !== undefined
                ? `${Number(data.score).toFixed(1)} / 100`
                : "N/A";
        const proba = data.proba || {};
        const keywords = data.keywords || [];

        // Update label & score
        labelSpan.textContent = label;
        scoreSpan.textContent = score;
        labelSpan.className = `prs-label-value pill-${label}`;

        // Probabilities
        probaList.innerHTML = "";
        ["High", "Medium", "Low"].forEach((cls) => {
            if (proba[cls] !== undefined) {
                const li = document.createElement("li");
                li.textContent = `${cls}: ${(proba[cls] * 100).toFixed(2)}%`;
                probaList.appendChild(li);
            }
        });

        // Keywords
        keywordsList.innerHTML = "";
        if (Array.isArray(keywords) && keywords.length > 0) {
            keywords.forEach((k) => {
                const li = document.createElement("li");
                li.textContent = typeof k === "string" ? k : (k.text || "");
                keywordsList.appendChild(li);
            });
        } else {
            const li = document.createElement("li");
            li.textContent = "(No keywords extracted)";
            keywordsList.appendChild(li);
        }

        statusDiv.innerHTML = `<p class="prs-source"><strong>Source:</strong> ${source || "Analyzed text"}</p>`;
    }

    // Main click handler
    analyzeBtn.addEventListener("click", async () => {
        resetResults();

        // If we already scanned and are in "radio mode", just analyze the selected URL.
        if (awaitingPolicySelection && lastPolicyCandidates.length > 0) {
            const selectedInput = document.querySelector(
                "input[name='prs-policy-radio']:checked"
            );
            if (!selectedInput) {
                statusDiv.innerHTML =
                    "<span style='color:red;'>Please select a privacy policy link first.</span>";
                return;
            }

            const selectedUrl = selectedInput.value;
            if (!selectedUrl) {
                statusDiv.innerHTML =
                    "<span style='color:red;'>Selected policy URL is invalid.</span>";
                return;
            }

            urlSpan.textContent = selectedUrl;
            statusDiv.textContent = "Fetching and analyzing selected privacy policy...";

            try {
                const backendResp = await analyzeUrlViaBackground(selectedUrl);
                if (!backendResp || !backendResp.ok) {
                    const msg =
                        (backendResp && backendResp.error) ||
                        "Unknown backend error while analyzing selected policy URL.";
                    statusDiv.innerHTML = `<span style='color:red;'>Backend error: ${msg}</span>`;
                    return;
                }

                const data = backendResp.data || backendResp;
                renderResult(
                    data,
                    "Fetched and analyzed selected privacy policy URL."
                );
            } catch (err) {
                console.error("Analyze selected policy failed:", err);
                statusDiv.innerHTML =
                    "<span style='color:red;'>Error while analyzing selected policy.</span>";
            }

            // Stay in radio mode so the user can change selection and re-run if they want.
            return;
        }

        // Otherwise: first step — scan the current tab for privacy text / links.
        statusDiv.textContent = "Scanning page for privacy information...";
        awaitingPolicySelection = false;
        lastPolicyCandidates = [];
        renderPolicyPicker([]);

        try {
            // 1) Get active tab
            const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
            const tab = tabs && tabs[0];
            if (!tab || !tab.id) {
                statusDiv.innerHTML =
                    "<span style='color:red;'>No active tab found.</span>";
                return;
            }

            const currentUrl = tab.url || "";
            urlSpan.textContent = currentUrl;

            // 2) Ask content script on that tab to extract privacy text + candidates
            let extractResp;
            try {
                extractResp = await sendToContentScript(tab.id, { type: "PRS_EXTRACT_TEXT" });
            } catch (err) {
                console.error("Content script error:", err);
                statusDiv.innerHTML =
                    "<span style='color:red;'>Could not communicate with content script. Make sure the extension is allowed on this site.</span>";
                return;
            }

            if (!extractResp) {
                statusDiv.innerHTML =
                    "<span style='color:red;'>No response from content script.</span>";
                return;
            }

            const {
                ok,
                text,
                sourceUrl,
                fromPolicy,
                policyCandidates,
                error
            } = extractResp;

            lastSourceUrl = sourceUrl || currentUrl;
            if (lastSourceUrl) {
                urlSpan.textContent = lastSourceUrl;
            }

            const trimmedText = (text || "").trim();
            const hasUsableText = trimmedText.length >= 20;

            // 3A) If this looks like a dedicated privacy policy page and we got text, analyze it directly
            if (fromPolicy && ok && hasUsableText) {
                statusDiv.textContent =
                    "Detected a privacy policy page. Analyzing this page...";
                analyzeBtn.textContent = "Analyze This Page";

                const backendResp = await analyzeTextViaBackground(trimmedText);

                if (!backendResp || !backendResp.ok) {
                    const msg =
                        (backendResp && backendResp.error) ||
                        "Unknown backend error. Is the FastAPI server running?";
                    statusDiv.innerHTML = `<span style='color:red;'>Backend error: ${msg}</span>`;
                    return;
                }

                const data = backendResp.data || backendResp;
                renderResult(
                    data,
                    "Detected privacy policy page (strict-filtered on-page text)."
                );
                return;
            }

            // 3B) If we *aren't* on a privacy policy page (or have no usable text),
            // but we found privacy-related links, show top-3 as radio buttons.
            if (Array.isArray(policyCandidates) && policyCandidates.length > 0) {
                lastPolicyCandidates = policyCandidates.slice(0, 3);
                awaitingPolicySelection = true;
                renderPolicyPicker(lastPolicyCandidates);
                analyzeBtn.textContent = "Analyze Selected Policy";

                statusDiv.innerHTML =
                    "<em>Select one of the privacy policy links below, then click “Analyze Selected Policy”.</em>";
                return;
            }

            // 3C) Fallback: we have some extracted text even if it's not a clear privacy page
            if (ok && hasUsableText) {
                statusDiv.textContent =
                    "Analyzing privacy-related text found on this page...";
                analyzeBtn.textContent = "Analyze This Page";

                const backendResp = await analyzeTextViaBackground(trimmedText);

                if (!backendResp || !backendResp.ok) {
                    const msg =
                        (backendResp && backendResp.error) ||
                        "Unknown backend error. Is the FastAPI server running?";
                    statusDiv.innerHTML = `<span style='color:red;'>Backend error: ${msg}</span>`;
                    return;
                }

                const data = backendResp.data || backendResp;
                const source = fromPolicy
                    ? "Detected privacy policy page (limited text)."
                    : "Current page (strict privacy text filter).";
                renderResult(data, source);
                return;
            }

            // 3D) No text and no links: show error
            const errMsg =
                error ||
                "Could not extract privacy-related text or locate a privacy policy link on this page.";
            statusDiv.innerHTML = `<span style='color:red;'>${errMsg}</span>`;
            analyzeBtn.textContent = "Analyze This Page";
        } catch (err) {
            console.error("Analyze failed:", err);
            statusDiv.innerHTML =
                "<span style='color:red;'>Error: " + err.message + "</span>";
        }
    });
});
