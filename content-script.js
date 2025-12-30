// scripts/content-script.js

// Heuristically decide if this is a dedicated privacy policy page
function isLikelyPrivacyPolicyPage() {
    const href = window.location.href.toLowerCase();
    const title = (document.title || "").toLowerCase();
    const path = window.location.pathname.toLowerCase();

    const privacyWords = [
        "privacy",
        "privacy-policy",
        "privacy_policy",
        "data-protection",
        "cookie-policy",
        "cookies",
        "gdpr",
        "ccpa",
        "cpra"
    ];

    return privacyWords.some((w) => {
        return href.includes(w) || title.includes(w) || path.includes(w);
    });
}

// Check if a text node is inside something we definitely don't want
function isInHiddenOrCodeLikeContainer(node) {
    const el = node.parentElement;
    if (!el) return true;

    // Skip script/style/etc directly
    const tag = el.tagName;
    if (["SCRIPT", "STYLE", "NOSCRIPT", "IFRAME", "CANVAS", "SVG"].includes(tag)) {
        return true;
    }

    // Skip hidden / decorative containers
    if (el.closest("[aria-hidden='true'],[hidden],[role='presentation']")) {
        return true;
    }

    // Skip obvious nav/button UI; we don't want menu labels
    if (el.closest("nav, header, footer, button")) {
        return true;
    }

    return false;
}

// Collect text nodes and keep only the ones that look privacy-related & readable
function extractStrictPrivacyText() {
    const keywords = [
        "privacy",
        "personal data",
        "personal information",
        "pii",
        "data",
        "information",
        "collect",
        "collection",
        "sharing",
        "share",
        "sell",
        "sale",
        "third party",
        "third-party",
        "cookies",
        "cookie",
        "tracking",
        "analytics",
        "opt-out",
        "opt out",
        "consent",
        "retention",
        "profiling",
        "processing",
        "data protection",
        "california consumer privacy act",
        "ccpa",
        "cpra"
    ];

    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        {
            acceptNode(node) {
                if (isInHiddenOrCodeLikeContainer(node)) {
                    return NodeFilter.FILTER_REJECT;
                }

                const text = node.textContent || "";
                const trimmed = text.trim();

                // Too short to be meaningful
                if (trimmed.length < 40) {
                    return NodeFilter.FILTER_REJECT;
                }

                const lower = trimmed.toLowerCase();

                // Only keep if it mentions at least one privacy-ish keyword
                const matchesKeyword = keywords.some((k) => lower.includes(k));
                if (!matchesKeyword) {
                    return NodeFilter.FILTER_REJECT;
                }

                // If it looks like pure code (tons of symbols), drop it
                const nonWordChars = (trimmed.match(/[^a-z0-9\s]/gi) || []).length;
                const letters = (trimmed.match(/[a-z]/gi) || []).length;
                if (letters > 0 && nonWordChars / (letters + 1) > 1.2) {
                    return NodeFilter.FILTER_REJECT;
                }

                return NodeFilter.FILTER_ACCEPT;
            }
        }
    );

    const chunks = [];
    let node;
    while ((node = walker.nextNode())) {
        const t = node.textContent.trim();
        if (t) chunks.push(t);
    }

    // De-duplicate-ish by using a Set
    const unique = Array.from(new Set(chunks));
    return unique.join("\n\n");
}

// Recursively collect <a href> elements from document + any open shadowRoots
function collectAnchorsDeep(root) {
    const anchors = [];

    function traverse(node) {
        if (!node) return;

        if (node.nodeType === Node.ELEMENT_NODE) {
            const el = /** @type {HTMLElement} */ (node);

            // If this element is an <a> with an href, record it
            if (el.tagName === "A" && el.hasAttribute("href")) {
                anchors.push(el);
            }

            // If this element has an open shadow root, traverse it too
            if (el.shadowRoot) {
                traverse(el.shadowRoot);
            }

            // Traverse children
            const children = el.children;
            for (let i = 0; i < children.length; i++) {
                traverse(children[i]);
            }
        } else if (
            node.nodeType === Node.DOCUMENT_FRAGMENT_NODE ||
            node.nodeType === Node.DOCUMENT_NODE
        ) {
            const children = node.children || [];
            for (let i = 0; i < children.length; i++) {
                traverse(children[i]);
            }
        }
    }

    traverse(root);
    return anchors;
}

// Find top privacy policy link candidates on the page
function findPrivacyPolicyCandidates() {
    // Use deep traversal so we see anchors inside open shadow DOM (TikTok, etc.)
    const anchors = collectAnchorsDeep(document);
    if (!anchors.length) return [];

    const strongPhrases = [
        "privacy policy",
        "privacy notice",
        "privacy statement",
        "privacy centre",
        "privacy center",
        "data policy",
        "data protection",
        "cookie policy",
        "cookies policy",
        "your privacy choices",
        "your privacy rights"
    ];

    const genericPrivacy = ["privacy", "your privacy", "data privacy", "data protection"];
    const seen = new Set();
    const candidates = [];

    for (const a of anchors) {
        const rawHref = a.getAttribute("href") || "";
        const hrefLower = rawHref.toLowerCase();

        // Ignore javascript: / mailto: / tel:
        if (
            hrefLower.startsWith("javascript:") ||
            hrefLower.startsWith("mailto:") ||
            hrefLower.startsWith("tel:")
        ) {
            continue;
        }

        let label =
            (a.innerText || a.textContent || "").trim() ||
            a.getAttribute("aria-label") ||
            a.getAttribute("title") ||
            "";
        label = label.trim();
        if (!label) continue;

        const lower = label.toLowerCase();
        let weight = 0;

        // Strong phrases get highest weight
        if (strongPhrases.some((p) => lower.includes(p))) {
            weight += 3;
        }

        // Generic privacy mentions (but avoid settings/preferences-only)
        if (genericPrivacy.some((p) => lower.includes("privacy"))) {
            if (!/\bsettings\b|\bpreferences\b/i.test(lower)) {
                weight += 2;
            }
        }

        // URL path hints
        if (/privacy|data[-_]?protection|privacy-policy|cookie[-_]?policy|cookies/i.test(hrefLower)) {
            weight += 2;
        }

        // Skip if no signal
        if (weight <= 0) continue;

        let absoluteUrl;
        try {
            absoluteUrl = new URL(rawHref, window.location.href).href;
        } catch {
            continue;
        }

        if (seen.has(absoluteUrl)) continue;
        seen.add(absoluteUrl);

        candidates.push({
            url: absoluteUrl,
            label,
            weight
        });
    }

    // Sort highest-weight first and keep top 3
    candidates.sort((a, b) => b.weight - a.weight);
    return candidates.slice(0, 3);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (!message || message.type !== "PRS_EXTRACT_TEXT") {
        return;
    }

    try {
        const text = extractStrictPrivacyText();
        const fromPolicy = isLikelyPrivacyPolicyPage();
        const sourceUrl = window.location.href;
        const policyCandidates = findPrivacyPolicyCandidates();
        const best = policyCandidates.length > 0 ? policyCandidates[0] : null;

        if (!text || text.trim().length === 0) {
            // No usable text, but we might still have policy link candidates
            sendResponse({
                ok: false,
                text: "",
                sourceUrl,
                fromPolicy,
                policyFound: !!best,
                policyUrl: best ? best.url : null,
                policyCandidates
            });
        } else {
            // We successfully extracted privacy-ish text
            sendResponse({
                ok: true,
                text,
                sourceUrl,
                fromPolicy,
                policyFound: !!best,
                policyUrl: best ? best.url : null,
                policyCandidates
            });
        }
    } catch (e) {
        console.error("Error extracting privacy text:", e);
        sendResponse({
            ok: false,
            error: String(e),
            text: "",
            sourceUrl: window.location.href,
            fromPolicy: isLikelyPrivacyPolicyPage(),
            policyFound: false,
            policyUrl: null,
            policyCandidates: []
        });
    }

    // Response is synchronous; no need to return true
});
