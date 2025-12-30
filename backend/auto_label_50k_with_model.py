# -*- coding: utf-8 -*-
"""
auto_label_50k_with_model.py

Re-label the 50k dataset using the CURRENT model pipeline in:
    artifacts/model_pipeline.joblib

Assumes you have already trained a good model (e.g., using train_balanced.csv)
and saved it as artifacts/model_pipeline.joblib via train_with_final.py.

Input:
    data/train_autolabeled_50k.csv

Expected columns (at minimum):
    policy_text  - raw text to classify

Other columns (snapshot_id, domain, length, old label, old proba_*) are preserved
but ignored for the new labeling logic.

Output:
    data/train_autolabeled_50k_relabelled.csv

Columns in output:
    - all original columns
    - label_model          (new label from the current model: High/Medium/Low)
    - proba_High_model
    - proba_Medium_model
    - proba_Low_model
    - label                (overwritten to new label_model for convenience)
"""

from pathlib import Path
import pandas as pd
import joblib

# ---------------- Paths ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

IN_PATH = DATA_DIR / "train_autolabeled_50k.csv"
OUT_PATH = DATA_DIR / "train_autolabeled_50k_relabelled.csv"
MODEL_PATH = ARTIFACTS_DIR / "model_pipeline.joblib"

# Your big file uses this column name:
TEXT_COL = "policy_text"

# Thresholds similar to your backend logic:
THR_HIGH = 0.45   # if High is top and >= 0.45 -> High
THR_LOW  = 0.55   # if Low  is top and >= 0.55 -> Low
# otherwise -> Medium


def pick_label_from_proba(proba_map):
    """
    Decide label based on probabilities and thresholds.

    - High if P(High) >= THR_HIGH and it's the top class
    - Low  if P(Low)  >= THR_LOW  and it's the top class
    - Otherwise Medium

    Expects a dict like {"High": pH, "Medium": pM, "Low": pL} (any casing okay).
    """
    if not proba_map:
        return "Medium"

    # Normalize for safety
    total = sum(proba_map.values()) or 1.0
    probs = {k: v / total for k, v in proba_map.items()}

    keyH = next((k for k in probs if k.lower() == "high"), None)
    keyM = next((k for k in probs if k.lower() == "medium"), None)
    keyL = next((k for k in probs if k.lower() == "low"), None)

    vH = probs.get(keyH, 0.0) if keyH else 0.0
    vM = probs.get(keyM, 0.0) if keyM else 0.0
    vL = probs.get(keyL, 0.0) if keyL else 0.0

    # High if it's top and above threshold
    if keyH and vH >= THR_HIGH and vH >= vM and vH >= vL:
        return "High"

    # Low if it's top and above threshold
    if keyL and vL >= THR_LOW and vL >= vM and vL >= vH:
        return "Low"

    # Otherwise Medium
    return "Medium"


def main():
    # Load model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model pipeline not found at {MODEL_PATH}. "
            f"Train it first (e.g. with train_with_final.py)."
        )

    print(f"[INFO] Loading model from {MODEL_PATH}")
    pipe = joblib.load(MODEL_PATH)

    # Load data
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found at {IN_PATH}")

    print(f"[INFO] Loading data from {IN_PATH}")
    df = pd.read_csv(IN_PATH)

    if TEXT_COL not in df.columns:
        raise ValueError(
            f"Expected text column '{TEXT_COL}' in {IN_PATH}, "
            f"but found: {list(df.columns)}"
        )

    texts = df[TEXT_COL].astype(str).values
    n = len(texts)
    print(f"[INFO] Predicting probabilities for {n} rows...")

    # Predict probabilities for all texts
    proba = pipe.predict_proba(texts)  # shape: (n_samples, n_classes)
    classes = [str(c) for c in pipe.classes_]

    # Map class -> index
    idx_H = next((i for i, c in enumerate(classes) if c.lower() == "high"), None)
    idx_M = next((i for i, c in enumerate(classes) if c.lower() == "medium"), None)
    idx_L = next((i for i, c in enumerate(classes) if c.lower() == "low"), None)

    if idx_H is None or idx_M is None or idx_L is None:
        raise RuntimeError(
            f"Model classes {classes} do not include High/Medium/Low as expected."
        )

    new_labels = []
    pH_list = []
    pM_list = []
    pL_list = []

    for row_proba in proba:
        pH = float(row_proba[idx_H])
        pM = float(row_proba[idx_M])
        pL = float(row_proba[idx_L])

        p_map = {
            "High": pH,
            "Medium": pM,
            "Low": pL,
        }

        label = pick_label_from_proba(p_map)
        new_labels.append(label)
        pH_list.append(pH)
        pM_list.append(pM)
        pL_list.append(pL)

    # Build output DataFrame
    out_df = df.copy()
    out_df["label_model"] = new_labels
    out_df["proba_High_model"] = pH_list
    out_df["proba_Medium_model"] = pM_list
    out_df["proba_Low_model"] = pL_list

    # Overwrite the generic 'label' column with the model's label for convenience
    out_df["label"] = new_labels

    print("[INFO] New label distribution (label):")
    print(out_df["label"].value_counts())

    print(f"[INFO] Writing relabelled dataset to {OUT_PATH}")
    out_df.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
