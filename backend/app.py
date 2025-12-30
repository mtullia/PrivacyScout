# -*- coding: utf-8 -*-
"""
app.py — FastAPI backend for Privacy Risk Scout.

- Loads a TF-IDF + LogisticRegression pipeline from artifacts/model_pipeline.joblib
- Uses thresholds + lexicon overrides to pick High / Medium / Low labels
- Uses RAKE (via rake_wrapper.RakeWrapper) to extract top keywords
- Exposes:
    GET  /health
    GET  /metrics
    GET  /viz          (HTML view: interactive analyzer + top features)
    POST /analyze      (main endpoint used by the browser extension)
"""

from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import json

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rake_wrapper import RakeWrapper


# ---------------- Paths & global objects ----------------

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

PIPELINE_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics_final.json"
TOP_FEATURES_PATH = ARTIFACTS_DIR / "top_features_final.json"

# Load pipeline
if not PIPELINE_PATH.exists():
    raise RuntimeError(f"Model pipeline not found at {PIPELINE_PATH}")

print(f"[INFO] Loading model pipeline from {PIPELINE_PATH}")
pipe = joblib.load(PIPELINE_PATH)

# Load metrics (optional)
metrics: Optional[Dict[str, Any]] = None
if METRICS_PATH.exists():
    try:
        with METRICS_PATH.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"[INFO] Loaded metrics from {METRICS_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load metrics from {METRICS_PATH}: {e}")

# Load top features (optional)
top_features: Optional[Dict[str, Any]] = None
if TOP_FEATURES_PATH.exists():
    try:
        with TOP_FEATURES_PATH.open("r", encoding="utf-8") as f:
            top_features = json.load(f)
        print(f"[INFO] Loaded top features from {TOP_FEATURES_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load top features from {TOP_FEATURES_PATH}: {e}")

# RAKE for keyword extraction
rake = RakeWrapper(language="english")


# ---------------- Pydantic models ----------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Page or privacy policy text to analyze")


# ---------------- Lexicon heuristics (lightweight) ----------------

HIGH_PHRASES = [
    "sell your data",
    "sell your personal information",
    "sell, rental, and profiling",
    "targeted advertising",
    "cross-device and cross-context behavioral ads",
    "behavioral, interest-based, or personalized ads",
    "profiling",
    "automated decision",
    "automated decision-making",
    "data broker",
    "data brokers",
    "biometric",
    "facial recognition",
    "location data",
    "sensitive personal information",
    "share your information with third parties for marketing",
    "indefinite retention",
    "data sets with third parties for their own purposes",
]

LOW_PHRASES = [
    "we do not sell your personal information",
    "we do not sell personal information",
    "do not sell my personal information",
    "we will not sell your data",
]

# Very generic stopwords we never want as standalone "keywords"
STOPWORD_TOKENS = {
    "and", "or", "but", "the", "a", "an",
    "may", "can", "could", "would", "should",
    "we", "you", "our", "us", "your", "they", "them", "their",
    "it", "its", "this", "that", "these", "those",
    "to", "from", "for", "of", "in", "on", "at", "by", "as",
    "is", "are", "was", "were", "be", "been", "being",
    "will", "shall", "do", "does", "did", "not",
    "out", "up", "down", "over", "under"
}

# Phrases we want to avoid surfacing as "top keywords" because
# they are boilerplate titles or legal references rather than
# meaningful risk signals.
KEYWORD_PHRASE_BLOCKLIST = [
    "privacy policy",
    "privacy notice",
    "privacy statement",
    "ccpa",
    "california consumer privacy act",
    "california probate code",
    "probate code",
    "sections ",
    "service privacy statement",
    "smart tv service privacy",
    "terms of use",
    "terms and conditions",
]


def text_has_low_risk_phrases(text: str) -> bool:
    """
    Check if text contains strong 'we do not sell' style language.
    """
    lowered = text.lower()
    return any(phrase in lowered for phrase in LOW_PHRASES)


def count_high_risk_hits(text: str) -> int:
    """
    Count how many high-risk phrases appear in the text.
    """
    lowered = text.lower()
    return sum(1 for phrase in HIGH_PHRASES if phrase in lowered)


def lexicon_boost(text: str, proba_map: Dict[str, float]) -> Dict[str, float]:
    """
    Simple heuristic tweaks to the model probabilities based on key phrases.

    - If we see strong high-risk phrases, nudge High upward.
    - If we see strong low-risk phrases, nudge Low upward and High downward.
    """
    if not proba_map:
        return proba_map

    p = proba_map.copy()

    keyH = next((k for k in p if k.lower() == "high"), None)
    keyM = next((k for k in p if k.lower() == "medium"), None)
    keyL = next((k for k in p if k.lower() == "low"), None)

    # Normalize first
    total = sum(p.values()) or 1.0
    for k in list(p.keys()):
        p[k] = float(p[k]) / total

    high_hits = count_high_risk_hits(text)
    boost_low = text_has_low_risk_phrases(text)

    # Boost High if high-risk phrases are present
    if high_hits >= 1 and keyH:
        p[keyH] += 0.05 * high_hits

    # Boost Low if strong "we do not sell" style phrases appear
    if boost_low and keyL:
        p[keyL] += 0.08
        if keyH:
            p[keyH] = max(0.0, p[keyH] - 0.05)

    # Re-normalize
    total = sum(p.values()) or 1.0
    for k in list(p.keys()):
        p[k] = float(p[k]) / total

    return p


# ---------------- Decision thresholds ----------------

THR_HIGH = 0.40   # if High is top and >= 0.40 -> High
THR_LOW = 0.55    # if Low  is top and >= 0.55 -> Low

OVERRIDE_HIGH_MIN = 0.15   # if we see risky phrases and High is at least this
OVERRIDE_HIT_COUNT = 3     # if we see this many risky phrases, force High


def pick_label_from_proba(proba_map: Dict[str, float], text: Optional[str] = None) -> Tuple[str, float]:
    """
    Decide the final label and confidence score from a probability map.

    - High if P(High) is top and >= THR_HIGH
    - Low  if P(Low)  is top and >= THR_LOW
    - Otherwise Medium

    Extra overrides using HIGH_PHRASES:
      - If high-risk phrases count >= OVERRIDE_HIT_COUNT, force High.
      - If high-risk phrases count >= 1 and P(High) >= OVERRIDE_HIGH_MIN, force High.
    """
    if not proba_map:
        return "Medium", 0.0

    # Normalize and canonicalize
    total = sum(proba_map.values()) or 1.0
    probs = {k: float(v) / total for k, v in proba_map.items()}

    keyH = next((k for k in probs if k.lower() == "high"), None)
    keyM = next((k for k in probs if k.lower() == "medium"), None)
    keyL = next((k for k in probs if k.lower() == "low"), None)

    vH = probs.get(keyH, 0.0) if keyH else 0.0
    vM = probs.get(keyM, 0.0) if keyM else 0.0
    vL = probs.get(keyL, 0.0) if keyL else 0.0

    # Identify top class (before override)
    top_label = None
    top_val = -1.0
    for lbl, val in [("High", vH), ("Medium", vM), ("Low", vL)]:
        if val > top_val:
            top_label = lbl
            top_val = val

    # --- Hard overrides for obviously high-risk language ---
    if text is not None:
        hits = count_high_risk_hits(text)

        # If the text is saturated with high-risk phrases, force High.
        if hits >= OVERRIDE_HIT_COUNT:
            return "High", max(vH, 0.75)

        # If we see at least one risky phrase AND the model isn't completely against High,
        # upgrade to High even if Medium is slightly higher.
        if hits >= 1 and vH >= OVERRIDE_HIGH_MIN:
            return "High", vH

    # --- Normal threshold-based logic ---
    if top_label == "High" and vH >= THR_HIGH:
        return "High", vH

    if top_label == "Low" and vL >= THR_LOW:
        return "Low", vL

    # Default to Medium; use its probability as the score if available
    return "Medium", vM


def compute_risk_score(label: str, proba: Dict[str, float]) -> float:
    """
    Compute a 0–100 privacy risk rating from probabilities + final label.

    Rough idea:
        risk_base = P(High) + 0.5 * P(Medium)

    Then clamp into reasonable bands based on the final label so that:
      - High   -> always looks clearly high (>= ~70)
      - Medium -> stays mid (40–69)
      - Low    -> stays low (<= 30)
    """
    pH = float(proba.get("High", 0.0))
    pM = float(proba.get("Medium", 0.0))

    base = pH + 0.5 * pM
    if base < 0.0:
        base = 0.0
    if base > 1.0:
        base = 1.0

    if label == "High":
        if base < 0.70:
            base = 0.70
    elif label == "Medium":
        if base < 0.40:
            base = 0.40
        if base > 0.69:
            base = 0.69
    elif label == "Low":
        if base > 0.30:
            base = 0.30

    return round(base * 100.0, 1)


# ---------------- Keyword helpers ----------------

def _get_vectorizer_and_classifier(pipeline):
    """
    Best-effort extraction of vectorizer and classifier from a sklearn Pipeline.
    Returns (vectorizer, classifier) or (None, None) if not found.
    """
    vec = None
    clf = None

    if hasattr(pipeline, "named_steps"):
        steps = pipeline.named_steps
        for step in steps.values():
            if hasattr(step, "get_feature_names_out"):
                vec = step
            if hasattr(step, "coef_") and hasattr(step, "predict_proba"):
                clf = step
    elif hasattr(pipeline, "steps"):
        for _, step in getattr(pipeline, "steps", []):
            if hasattr(step, "get_feature_names_out"):
                vec = step
            if hasattr(step, "coef_") and hasattr(step, "predict_proba"):
                clf = step

    if vec is None or clf is None:
        return None, None
    return vec, clf


def get_model_top_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Use TF–IDF + LogisticRegression coefficients to get the most
    influential features for the model's prediction on this text.

    Returns a list of keyword strings.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    vec, clf = _get_vectorizer_and_classifier(pipe)
    if vec is None or clf is None:
        return []

    try:
        X_vec = vec.transform([cleaned])
    except Exception:
        return []

    if not hasattr(vec, "get_feature_names_out"):
        return []

    feature_names = vec.get_feature_names_out()

    try:
        probs = clf.predict_proba(X_vec)[0]
    except Exception:
        return []

    if not hasattr(clf, "coef_"):
        return []

    pred_idx = int(np.argmax(probs))
    coefs = clf.coef_[pred_idx]

    x_dense = X_vec.toarray()[0]
    contrib = coefs * x_dense

    sorted_idx = np.argsort(contrib)[::-1]

    keywords: List[str] = []
    for idx in sorted_idx:
        if contrib[idx] <= 0:
            break

        token = feature_names[idx]
        if not token:
            continue
        lower = token.lower()

        if len(lower) < 3:
            continue
        if lower in STOPWORD_TOKENS:
            continue
        if lower.isdigit():
            continue

        keywords.append(token)

        if len(keywords) >= top_n:
            break

    return keywords


def clean_rake_keywords(raw_keywords: Any, top_n: int = 5) -> List[str]:
    """
    Take raw RAKE output and:
      - normalize to strings
      - drop very short / stopword-only entries
      - drop boilerplate titles like '... privacy policy' etc.
    """
    if not raw_keywords:
        return []

    # Normalize everything to plain strings
    normalized: List[str] = []
    for item in raw_keywords:
        text = None

        if isinstance(item, str):
            text = item
        elif isinstance(item, dict) and "text" in item:
            text = str(item.get("text"))
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            text = str(item[0])
        else:
            continue

        if not text:
            continue

        s = text.strip()
        if not s:
            continue

        normalized.append(s)

    # Filter
    cleaned: List[str] = []
    for s in normalized:
        lower = s.lower()

        # Skip single words in stopwords
        if " " not in lower and lower in STOPWORD_TOKENS:
            continue

        # Skip very short "keywords"
        if len(lower) < 4:
            continue

        # Skip boilerplate/legal titles
        if any(bad in lower for bad in KEYWORD_PHRASE_BLOCKLIST):
            continue

        cleaned.append(s)

    # If filtering killed everything, fall back to less strict filtering:
    if not cleaned:
        fallback: List[str] = []
        for s in normalized:
            lower = s.lower()
            if " " not in lower and lower in STOPWORD_TOKENS:
                continue
            if len(lower) < 4:
                continue
            fallback.append(s)
            if len(fallback) >= top_n:
                break
        return fallback[:top_n]

    return cleaned[:top_n]


# ---------------- FastAPI app ----------------

app = FastAPI(
    title="Privacy Risk Scout Backend",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Routes ----------------

@app.get("/health")
async def health():
    return {"ok": True, "status": "healthy"}


@app.get("/metrics")
async def get_metrics():
    """
    Returns training / evaluation metrics saved by training script.
    """
    if metrics is None:
        raise HTTPException(status_code=404, detail="No metrics file found.")
    return metrics


@app.get("/viz", response_class=HTMLResponse)
async def viz():
    """
    HTML dashboard:

    - Top section: interactive analyzer (textarea + Analyze button)
    - Below: Top features per class (if available)
    """
    html_parts: List[str] = [
        "<html><head><title>Privacy Risk Scout - Dashboard</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;max-width:1000px;}",
        "h1,h2,h3{font-family:Arial,sans-serif;}",
        "textarea{width:100%;height:180px;font-family:monospace;font-size:13px;padding:8px;}",
        "button{padding:8px 14px;margin-top:8px;cursor:pointer;}",
        "#result{margin-top:15px;padding:10px;border:1px solid #ccc;border-radius:4px;background:#fafafa;font-size:14px;}",
        "table{border-collapse:collapse;margin-top:10px;margin-bottom:30px;width:100%;}",
        "th,td{border:1px solid #ccc;padding:4px 8px;font-size:13px;text-align:left;}",
        "th{background:#f0f0f0;}",
        ".pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:4px;}",
        ".pill-High{background:#ffe5e5;color:#a00000;}",
        ".pill-Medium{background:#fff5cc;color:#a06800;}",
        ".pill-Low{background:#e5ffe5;color:#007000;}",
        "</style>",
        "</head><body>",
        "<h1>Privacy Risk Scout – Dashboard</h1>",
        "<h2>Interactive Text Analyzer</h2>",
        "<p>Paste any privacy policy or page text below and click <strong>Analyze Text</strong> to see how the model classifies it.</p>",
        "<textarea id=\"inputText\" placeholder=\"Paste privacy policy or page text here...\"></textarea>",
        "<br>",
        "<button id=\"analyzeBtn\">Analyze Text</button>",
        "<div id=\"result\">Result will appear here.</div>",
        "<hr>",
        "<h2>Top Features per Class</h2>",
    ]

    if not top_features:
        html_parts.append("<p><em>No top_features_final.json found.</em></p>")
    else:
        for cls, feats in top_features.items():
            html_parts.append(f"<h3>{cls}</h3>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Term</th><th>Weight</th></tr>")
            for item in feats:
                term = item.get("term", "")
                weight = item.get("weight", 0.0)
                html_parts.append(f"<tr><td>{term}</td><td>{weight:.4f}</td></tr>")
            html_parts.append("</table>")

    html_parts.append("""
<script>
document.getElementById('analyzeBtn').addEventListener('click', async function () {
  const text = document.getElementById('inputText').value.trim();
  const resultDiv = document.getElementById('result');

  if (!text || text.length < 20) {
    resultDiv.innerHTML = "<p><em>Please paste at least a few sentences of text.</em></p>";
    return;
  }

  resultDiv.innerHTML = "<p><em>Analyzing...</em></p>";

  try {
    const resp = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    });

    if (!resp.ok) {
      const txt = await resp.text();
      resultDiv.innerHTML = "<p><strong>Error:</strong> " + resp.status + " " + txt + "</p>";
      return;
    }

    const data = await resp.json();
    if (!data.ok) {
      resultDiv.innerHTML = "<p><strong>Backend:</strong> " + (data.error || "Not enough text to analyze.") + "</p>";
      return;
    }

    const label = data.label || "Unknown";
    const score = (data.score !== undefined) ? data.score.toFixed(1) : "N/A";
    const proba = data.proba || {};
    const keywords = data.keywords || [];

    let probaRows = "";
    const order = ["High","Medium","Low"];
    order.forEach(lbl => {
      if (proba.hasOwnProperty(lbl)) {
        probaRows += "<tr><td>" + lbl + "</td><td>" + proba[lbl].toFixed(3) + "</td></tr>";
      }
    });
    if (!probaRows) {
      for (const [k,v] of Object.entries(proba)) {
        probaRows += "<tr><td>" + k + "</td><td>" + v.toFixed(3) + "</td></tr>";
      }
    }

    let kwHtml = "";
    if (keywords.length > 0) {
      kwHtml = "<ul>";
      keywords.forEach(k => {
        if (typeof k === "string") {
          kwHtml += "<li>" + k + "</li>";
        } else if (k && k.text) {
          kwHtml += "<li>" + k.text + "</li>";
        }
      });
      kwHtml += "</ul>";
    } else {
      kwHtml = "<p><em>No keywords extracted.</em></p>";
    }

    resultDiv.innerHTML = `
      <p><strong>Label:</strong>
         <span class="pill pill-${label}">${label}</span>
         <strong>Risk rating:</strong> ${score}/100
      </p>
      <h4>Class probabilities</h4>
      <table>
        <tr><th>Class</th><th>Probability</th></tr>
        ${probaRows}
      </table>
      <h4>Top keywords</h4>
      ${kwHtml}
    `;
  } catch (e) {
    console.error(e);
    resultDiv.innerHTML = "<p><strong>Error:</strong> " + e + "</p>";
  }
});
</script>
""")

    html_parts.append("</body></html>")
    return HTMLResponse("".join(html_parts))


@app.post("/analyze")
async def analyze(req: AnalyzeRequest = Body(...)):
    """
    Main analysis endpoint.

    Request JSON:
        { "text": "...." }

    Response JSON:
        {
          "ok": true,
          "label": "High" | "Medium" | "Low",
          "score": 78.3,                  # risk rating 0–100
          "proba": { "High": 0.87,
                     "Medium": 0.10,
                     "Low": 0.03 },
          "top_spans": [],
          "keywords": [ "third-party advertising", "profiling", ... ]
        }
    """
    text = (req.text or "").strip()
    if not text or len(text) < 20:
        return {
            "ok": False,
            "error": "Not enough text to analyze.",
            "label": "Medium",
            "score": 0.0,
            "proba": {},
            "top_spans": [],
            "keywords": [],
        }

    # Model prediction (base probabilities)
    try:
        proba_arr = pipe.predict_proba([text])[0]
        classes = list(map(str, pipe.classes_))
        base = {c: float(p) for c, p in zip(classes, proba_arr)}
    except Exception as e:
        print(f"[ERROR] Model predict_proba failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")

    # Apply lexicon boost first
    boosted = lexicon_boost(text, base)

    # Pick final label + probability-based score
    label, _raw_score = pick_label_from_proba(boosted, text=text)

    # Compute 0–100 risk rating
    risk_score = compute_risk_score(label, boosted)

    # Model-driven keywords (best-effort; may be empty)
    try:
        model_keywords = get_model_top_keywords(text, top_n=5)
    except Exception as e:
        print(f"[WARN] get_model_top_keywords failed: {e}")
        model_keywords = []

    # RAKE keywords
    try:
        raw_keywords = rake.extract(text, k=20)
        rake_keywords = clean_rake_keywords(raw_keywords, top_n=5)
    except Exception as e:
        print(f"[WARN] RAKE failed: {e}")
        rake_keywords = []

    # Combine: model-driven first, then RAKE, deduplicated
    keywords: List[str] = []
    seen = set()
    for kw in model_keywords + rake_keywords:
        if not kw:
            continue
        s = str(kw).strip()
        lower = s.lower()
        if not s:
            continue
        if lower in seen:
            continue
        seen.add(lower)
        keywords.append(s)

    return {
        "ok": True,
        "label": label,
        "score": float(risk_score),
        "proba": boosted,
        "top_spans": [],
        "keywords": keywords,
    }
