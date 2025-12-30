# -*- coding: utf-8 -*-
"""
relabel_from_probs.py

Re-label the 50k dataset using the existing probability columns:
    proba_High, proba_Low, proba_Medium

Input:
    data/train_autolabeled_50k.csv

Expected columns:
    snapshot_id, domain, policy_text, ..., label, proba_High, proba_Low, proba_Medium

Output:
    data/train_autolabeled_50k_fixed.csv

Columns in output:
    (all original columns, but 'label' is replaced with a new value)
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

IN_PATH = DATA_DIR / "train_autolabeled_50k.csv"
OUT_PATH = DATA_DIR / "train_autolabeled_50k_fixed.csv"

# Thresholds for assigning labels
# More aggressive for High, stricter for Low
THR_HIGH = 0.40   # if P(High) >= 0.40 -> High (even if Medium is slightly larger)
THR_LOW  = 0.60   # if P(Low)  >= 0.60 and it's top -> Low
# otherwise -> Medium


def pick_label(pH: float, pL: float, pM: float) -> str:
    """
    Decide label based on probabilities and thresholds.

    - If P(High) >= THR_HIGH -> High  (we treat anything with solid High prob as High)
    - Else if Low has the highest prob and >= THR_LOW -> Low
    - Else -> Medium
    """
    # Safeguard NaNs
    pH = float(pH) if pH == pH else 0.0
    pL = float(pL) if pL == pL else 0.0
    pM = float(pM) if pM == pM else 0.0

    # 1) High wins if its probability is reasonably large
    if pH >= THR_HIGH:
        return "High"

    # 2) Otherwise, decide between Low and Medium
    probs = {"High": pH, "Low": pL, "Medium": pM}
    top_label = max(probs, key=probs.get)
    top_val = probs[top_label]

    if top_label == "Low" and top_val >= THR_LOW:
        return "Low"

    # 3) Anything else -> Medium
    return "Medium"


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found at {IN_PATH}")

    print(f"[INFO] Loading data from {IN_PATH}")
    df = pd.read_csv(IN_PATH)

    required_cols = ["proba_High", "proba_Low", "proba_Medium"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"Expected column '{c}' in {IN_PATH}, but found: {list(df.columns)}"
            )

    print(f"[INFO] Recomputing labels for {len(df)} rows...")
    new_labels = []

    for _, row in df.iterrows():
        pH = row["proba_High"]
        pL = row["proba_Low"]
        pM = row["proba_Medium"]
        new_labels.append(pick_label(pH, pL, pM))

    df["label"] = new_labels  # overwrite label

    print("[INFO] New label distribution:")
    print(df["label"].value_counts())

    print(f"[INFO] Writing fixed dataset to {OUT_PATH}")
    df.to_csv(OUT_PATH, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
