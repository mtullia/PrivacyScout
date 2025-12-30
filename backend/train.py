# -*- coding: utf-8 -*-
# train.py — calibrated TF-IDF + LogisticRegression, metrics, and feature export (compatible with sklearn variants)
from pathlib import Path
import csv, json
import numpy as np
import joblib
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "train_final.csv"
ART  = HERE / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# -------- load data --------
texts, labels = [], []
with DATA.open("r", encoding="utf-8", newline="") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        t = (row.get("text") or "").strip()
        y = (row.get("label") or "").strip()
        if t and y:
            texts.append(t)
            labels.append(y)

print(f"Loaded {len(texts)} rows, label distribution: {Counter(labels)}")

# -------- split --------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# -------- pipeline --------
tfidf = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.9,
    sublinear_tf=True,
    lowercase=True,
)

base_lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=None,
    random_state=42,
)

# Calibrated probabilities for better thresholds
clf = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)

pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
pipe.fit(X_train, y_train)

# -------- evaluation --------
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)
classes = list(map(str, pipe.classes_))

print("\n=== Metrics (holdout) ===")
print(classification_report(y_test, y_pred, digits=3))
cm = confusion_matrix(y_test, y_pred, labels=classes)
acc = accuracy_score(y_test, y_pred)
print("Labels:", classes)
print("Confusion matrix:\n", cm)
print("Accuracy:", round(acc, 3))

# Save metrics for /metrics endpoint
metrics = {
    "labels": classes,
    "confusion_matrix": cm.tolist(),
    "accuracy": acc,
    "report": classification_report(y_test, y_pred, output_dict=True),
}
(ART / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# -------- top features per class (robust to sklearn versions) --------
def get_wrapped_lr(calibrated: CalibratedClassifierCV):
    """
    Return the underlying LogisticRegression no matter which attribute name is used.
    Some versions use .estimator, older doc examples mention .base_estimator.
    """
    if hasattr(calibrated, "estimator") and calibrated.estimator is not None:
        return calibrated.estimator
    if hasattr(calibrated, "base_estimator") and calibrated.base_estimator is not None:
        return calibrated.base_estimator
    # If neither is present, just return object (will fail gracefully below)
    return calibrated

def top_ngrams_per_class(pipe: Pipeline, k: int = 25):
    vec = pipe.named_steps["tfidf"]
    cal = pipe.named_steps["clf"]
    lr  = get_wrapped_lr(cal)  # underlying LogisticRegression

    # If we can't reach coef_, return empty dict
    if not hasattr(lr, "coef_"):
        return {}

    feats = np.array(vec.get_feature_names_out())
    tops = {}
    for idx, c in enumerate(lr.classes_):
        coefs = lr.coef_[idx]
        topk = np.argsort(coefs)[-k:][::-1]
        tops[str(c)] = [(feats[i], float(coefs[i])) for i in topk]
    return tops

top_feats = top_ngrams_per_class(pipe, k=25)
(ART / "top_features.json").write_text(json.dumps(top_feats, indent=2), encoding="utf-8")

# -------- save single pipeline --------
joblib.dump(pipe, ART / "model_pipeline.joblib")
print(f"\nSaved {ART / 'model_pipeline.joblib'}")
print(f"Metrics: {ART / 'metrics.json'}")
print(f"Top features: {ART / 'top_features.json'}")
