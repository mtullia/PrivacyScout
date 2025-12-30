// scripts/background.js

// Utility: get config from chrome.storage with defaults
function getConfig() {
    return new Promise((resolve) => {
        chrome.storage.sync.get(
            ["backendUrl", "mockMode"],
            (data) => {
                resolve({
                    backendUrl: data.backendUrl || "http://127.0.0.1:8000",
                    mockMode: !!data.mockMode
                });
            }
        );
    });
}

// Simple mock response for when "Use Mock Data" is enabled
function getMockResponse(text) {
    const base = text || "";
    const lengthFactor = Math.min(base.length / 2000, 1);
    const high = 0.2 + 0.6 * lengthFactor; // silly heuristic for demo
    const low = 0.5 - 0.3 * lengthFactor;
    const medium = 1 - high - low;

    return {
        ok: true,
        endpoint: "mock",
        data: {
            label: high >= 0.6 ? "High" : low >= 0.6 ? "Low" : "Medium",
            score: Math.max(high, medium, low),
            proba: {
                High: high,
                Medium: medium,
                Low: low
            },
            keywords: [
                { text: "mock keyword: data collection" },
                { text: "mock keyword: third-party sharing" },
                { text: "mock keyword: retention policy" }
            ]
        }
    };
}

async function handleAnalyzeText(message, sendResponse) {
    const { text } = message;
    if (!text || typeof text !== "string") {
        sendResponse({
            ok: false,
            error: "No text provided to analyze."
        });
        return;
    }

    const { backendUrl, mockMode } = await getConfig();

    if (mockMode) {
        // Return mocked data without hitting the backend
        sendResponse(getMockResponse(text));
        return;
    }

    const baseUrl = backendUrl.replace(/\/+$/, ""); // trim trailing slash
    const url = `${baseUrl}/analyze`;

    try {
        const resp = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        if (!resp.ok) {
            const body = await resp.text().catch(() => "");
            sendResponse({
                ok: false,
                error: `Backend HTTP ${resp.status}`,
                body
            });
            return;
        }

        const data = await resp.json();
        // FastAPI returns: { ok, label, score, proba, keywords, ... }
        sendResponse({
            ok: true,
            endpoint: "analyze",
            data
        });
    } catch (err) {
        console.error("Error calling backend:", err);
        sendResponse({
            ok: false,
            error: String(err)
        });
    }
}

// Very simple HTML -> text stripper for fetched policy pages
function stripHtmlToText(html) {
    if (!html || typeof html !== "string") return "";

    // Remove scripts and styles
    let cleaned = html
        .replace(/<script[\s\S]*?<\/script>/gi, " ")
        .replace(/<style[\s\S]*?<\/style>/gi, " ");

    // Remove all tags
    cleaned = cleaned.replace(/<[^>]+>/g, " ");

    // Decode some common HTML entities manually
    cleaned = cleaned
        .replace(/&nbsp;/gi, " ")
        .replace(/&amp;/gi, "&")
        .replace(/&lt;/gi, "<")
        .replace(/&gt;/gi, ">")
        .replace(/&quot;/gi, '"')
        .replace(/&#39;/gi, "'");

    // Collapse whitespace
    cleaned = cleaned.replace(/\s+/g, " ").trim();

    return cleaned;
}

async function handleAnalyzeUrl(message, sendResponse) {
    const targetUrl = message.url;
    if (!targetUrl || typeof targetUrl !== "string") {
        sendResponse({
            ok: false,
            error: "No URL provided to analyze."
        });
        return;
    }

    const { backendUrl, mockMode } = await getConfig();

    if (mockMode) {
        // Just mock on the URL string
        sendResponse(getMockResponse(targetUrl));
        return;
    }

    try {
        // 1) Fetch the remote policy page HTML
        const pageResp = await fetch(targetUrl, {
            method: "GET",
            credentials: "omit"
        });

        if (!pageResp.ok) {
            const body = await pageResp.text().catch(() => "");
            sendResponse({
                ok: false,
                error: `Policy fetch HTTP ${pageResp.status}`,
                body
            });
            return;
        }

        const html = await pageResp.text();
        const text = stripHtmlToText(html);

        if (!text || text.length < 50) {
            sendResponse({
                ok: false,
                error: "Fetched page text is too short or empty."
            });
            return;
        }

        // 2) Send stripped text to backend /analyze
        const baseUrl = backendUrl.replace(/\/+$/, "");
        const apiUrl = `${baseUrl}/analyze`;

        const resp = await fetch(apiUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        if (!resp.ok) {
            const body = await resp.text().catch(() => "");
            sendResponse({
                ok: false,
                error: `Backend HTTP ${resp.status}`,
                body
            });
            return;
        }

        const data = await resp.json();

        sendResponse({
            ok: true,
            endpoint: "analyze-url",
            url: targetUrl,
            data
        });
    } catch (err) {
        console.error("Error fetching policy URL or calling backend:", err);
        sendResponse({
            ok: false,
            error: String(err)
        });
    }
}

// Message router
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (!message || !message.type) return;

    if (message.type === "PRS_ANALYZE_TEXT") {
        handleAnalyzeText(message, sendResponse);
        return true; // async
    }

    if (message.type === "PRS_ANALYZE_URL") {
        handleAnalyzeUrl(message, sendResponse);
        return true; // async
    }

    // For other message types, let them fall through
});
