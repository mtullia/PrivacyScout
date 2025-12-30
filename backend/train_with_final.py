# -*- coding: utf-8 -*-
"""
train_with_final.py

Final training script for Privacy Risk Scout.

Trains a TF-IDF + LogisticRegression model for privacy-risk classification,
with extra penalty for misclassifying 'High' risk.

Uses:
    data/train_balanced.csv

Expected columns:
    text   - the raw text of the policy snippet
    label  - one of: High, Medium, Low  (case-sensitive)

Outputs:
    artifacts/model_pipeline.joblib
    artifacts/metrics_final.json
    artifacts/top_features_final.json
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

import joblib

# ---------------- Paths & config ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Training data: use the good balanced dataset
DATA_PATH = DATA_DIR / "train_balanced.csv"

# Column names in train_balanced.csv
TEXT_COL = "text"
LABEL_COL = "label"

# How many top features per class to store for interpretation
TOP_K_FEATURES = 50


# ---------------- Data loading ----------------

def load_data() -> pd.DataFrame:
    """
    Load the balanced training dataset and show label distribution.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {DATA_PATH}. "
            f"Update DATA_PATH in train_with_final.py if your file is elsewhere."
        )

    df = pd.read_csv(DATA_PATH)

    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{TEXT_COL}' and '{LABEL_COL}' in {DATA_PATH}, "
            f"but found: {list(df.columns)}"
        )

    # Keep only text + label, drop missing, coerce text to string
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str)

    print(f"[INFO] Loaded {len(df)} rows from {DATA_PATH}")
    print("[INFO] Label distribution:")
    print(df[LABEL_COL].value_counts())

    return df


# ---------------- Pipeline definition ----------------

def build_pipeline(labels) -> Pipeline:
    """
    Build a TF-IDF + LogisticRegression pipeline.

    We apply heavier class_weight to 'High' to make the model less forgiving
    and more willing to assign High risk where appropriate.
    """
    # Base class_weight: every label gets weight 1.0
    class_weight = {lab: 1.0 for lab in labels}

    # If 'High' is one of the labels, give it extra weight
    if "High" in class_weight:
        class_weight["High"] = 2.0  # bump to 3.0 later if you want more aggression

    print("[INFO] Using class_weight:", class_weight)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=200_000,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
    )

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight=class_weight,
        solver="liblinear",
    )

    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])

    return pipe


# ---------------- Training / evaluation ----------------

def evaluate_and_save(pipe: Pipeline,
                      X_train, y_train,
                      X_test, y_test):
    """
    Fit the pipeline, evaluate on a holdout set, then save model + metrics.
    """
    print("[INFO] Fitting model...")
    pipe.fit(X_train, y_train)

    # Save pipeline
    model_path = ARTIFACTS_DIR / "model_pipeline.joblib"
    joblib.dump(pipe, model_path)
    print(f"[INFO] Saved model pipeline to {model_path}")

    print("[INFO] Evaluating on holdout set...")
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    labels = list(pipe.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True)

    # Diagnostics for High risk behavior
    proba_df = pd.DataFrame(y_proba, columns=labels)
    mask_high_true = (y_test == "High")
    mask_med_true = (y_test == "Medium")
    mask_low_true = (y_test == "Low")

    avg_p_high_on_high = float(proba_df.loc[mask_high_true, "High"].mean()) if mask_high_true.any() and "High" in labels else None
    avg_p_high_on_med  = float(proba_df.loc[mask_med_true, "High"].mean()) if mask_med_true.any()  and "High" in labels else None
    avg_p_high_on_low  = float(proba_df.loc[mask_low_true, "High"].mean()) if mask_low_true.any()  and "High" in labels else None

    metrics = {
        "accuracy": float(acc),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "avg_p_high_on_high": avg_p_high_on_high,
        "avg_p_high_on_medium": avg_p_high_on_med,
        "avg_p_high_on_low": avg_p_high_on_low,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    metrics_path = ARTIFACTS_DIR / "metrics_final.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics to {metrics_path}")

    print("[INFO] Holdout metrics summary:")
    print(f"  Accuracy: {acc:.3f}")
    if avg_p_high_on_high is not None:
        print(f"  Avg P(High) | true High:   {avg_p_high_on_high:.3f}")
        print(f"  Avg P(High) | true Medium: {avg_p_high_on_med:.3f}")
        print(f"  Avg P(High) | true Low:    {avg_p_high_on_low:.3f}")


def compute_top_features(pipe: Pipeline):
    """
    Compute top positive features per class from the trained pipeline and save
    them into artifacts/top_features_final.json for interpretability / viz.
    """
    print("[INFO] Computing top features per class...")

    tfidf = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    feature_names = np.array(tfidf.get_feature_names_out())
    classes = clf.classes_
    coefs = clf.coef_  # shape (n_classes, n_features) in OVR mode

    top_features = {}

    for idx, cls in enumerate(classes):
        coef = coefs[idx]
        # Get indices of top K positive coefficients
        top_idx = np.argsort(coef)[-TOP_K_FEATURES:][::-1]
        feats = feature_names[top_idx]
        weights = coef[top_idx]

        top_features[str(cls)] = [
            {"term": f, "weight": float(w)}
            for f, w in zip(feats, weights)
        ]

    out_path = ARTIFACTS_DIR / "top_features_final.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(top_features, f, indent=2)
    print(f"[INFO] Saved top features to {out_path}")


def main():
    df = load_data()
    X = df[TEXT_COL].values
    y = df[LABEL_COL].values

    # Stratified split so each label is represented in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    unique_labels = sorted(df[LABEL_COL].unique().tolist())
    pipe = build_pipeline(unique_labels)

    evaluate_and_save(pipe, X_train, y_train, X_test, y_test)
    compute_top_features(pipe)


if __name__ == "__main__":
    main()
