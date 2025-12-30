# -*- coding: utf-8 -*-
"""
Batch tester for Privacy Risk Scout backend.
- Tries /analyze_proba first (for per-class probabilities).
- Falls back to /analyze if needed.
- Prints a table and saves results to CSV.

Run:
  cd extension/backend
  .\venv\Scripts\activate  (Windows)
  python batch_test.py
"""

import csv
import json
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8000"
USE_PROBA_ENDPOINT = True  # will auto-fallback if /analyze_proba not available
TIMEOUT = 8

SAYINGS = [
    # High-risk
    ("High (likely)",  "We sell your personal information to third-party advertisers."),
    ("High (likely)",  "Your data may be shared with our partners for targeted marketing."),
    ("High (likely)",  "We disclose personal information to third parties without consent."),
    # Medium-risk
    ("Medium (likely)","We may share your information with service providers for analytics."),
    ("Medium (likely)","You can opt out of certain data sharing by changing your settings."),
    ("Medium (likely)","We may use cookies and tracking technologies to improve our services."),
    # Low / Safe
    ("Low (likely)",   "We never sell your personal information."),
    ("Low (likely)",   "Your data is only used to provide the service you requested."),
    ("Low (likely)",   "We collect minimal information necessary to operate our website."),
    # Edge / Ambiguous
    ("Edge",           "We may share anonymized data for research purposes."),
    ("Edge",           "We require your consent before collecting sensitive information."),
    ("Edge",           "Personal data may be transferred outside your country.")
]

def call(endpoint, payload):
    url = f"{BASE}{endpoint}"
    r = requests.post(url, json=payload, timeout=TIMEOUT)
    # Always try to decode JSON; if server returned HTML/text, raise a helpful error
    try:
        j = r.json()
    except Exception:
        raise RuntimeError(f"{endpoint} returned non-JSON ({r.status_code}): {r.text[:200]}")
    if r.status_code != 200:
        raise RuntimeError(f"{endpoint} error {r.status_code}: {json.dumps(j)[:200]}")
    return j

def analyze(text):
    payload = {"text": text}
    if USE_PROBA_ENDPOINT:
        try:
            data = call("/analyze_proba", payload)
            # normalize shape: ensure keywords present if you want them in CSV
            data.setdefault("keywords", None)  # /analyze_proba may not include keywords
            return "analyze_proba", data
        except Exception as e:
            print(f"[fallback] /analyze_proba failed: {e}")
            # fall back to plain analyze
    data = call("/analyze", payload)
    # normalize: plain /analyze doesn’t have per-class "proba"
    data.setdefault("proba", None)
    return "analyze", data

def fmt_prob(dist):
    if not dist:
        return ""
    # Example: "Low=0.12 | Medium=0.33 | High=0.55"
    parts = [f"{k}={v:.2f}" for k, v in dist.items()]
    return " | ".join(parts)

def maybe_keywords(kws, k=5):
    if not kws:
        return ""
    return "; ".join([kw.get("text", "") for kw in kws[:k]])

def main():
    # quick health check
    try:
        h = requests.get(f"{BASE}/health", timeout=TIMEOUT).json()
        print("Health:", h)
    except Exception as e:
        print("Health check failed:", e)

    rows = []
    for expected, text in SAYINGS:
        start = time.time()
        which, data = analyze(text)
        dt = (time.time() - start) * 1000.0

        label = data.get("label", "?")
        score = data.get("score", 0.0)
        proba = data.get("proba", None)
        kws   = data.get("keywords", None)

        rows.append({
            "expected": expected,
            "endpoint": which,
            "label": label,
            "score": f"{score:.2f}",
            "proba": fmt_prob(proba),
            "keywords": maybe_keywords(kws),
            "latency_ms": f"{dt:.0f}",
            "text": text
        })

    # pretty print table
    cols = ["expected", "endpoint", "label", "score", "latency_ms", "proba", "keywords", "text"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}

    def line(char="-"):
        return "+".join(char * (widths[c] + 2) for c in cols)

    # header
    print(line("="))
    print("| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |")
    print(line("="))

    # rows
    for r in rows:
        print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |")
    print(line("="))

    # save CSV
    out = Path("batch_results.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out.resolve()}")
    print("Tip: open in Excel to sort by score/label.")

if __name__ == "__main__":
    main()
