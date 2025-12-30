# auto_label_50k.py
"""
Auto-label ~50k policies using the existing model and save:
    data/train_autolabeled_50k.csv

Usage (from backend folder, venv active):
    python auto_label_50k.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
ART_DIR = HERE / "artifacts"

UNLABELED_PATH = DATA_DIR / "train_unlabeled_50k.csv"
OUT_PATH = DATA_DIR / "train_autolabeled_50k.csv"
MODEL_PATH = ART_DIR / "model_pipeline.joblib"

def main():
    print("Loading model...")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run train.py first.")

    pipe = joblib.load(MODEL_PATH)

    print("Loading 50k dataset...")
    if not UNLABELED_PATH.exists():
        raise SystemExit(f"Unlabeled CSV not found at {UNLABELED_PATH}")

    df = pd.read_csv(UNLABELED_PATH)
    if "policy_text" not in df.columns:
        raise SystemExit("Expected column 'policy_text' in train_unlabeled_50k.csv")

    texts = df["policy_text"].astype(str).fillna("").tolist()

    print("Running predictions...")
    proba = pipe.predict_proba(texts)
    classes = list(pipe.classes_)

    # Map class to column index
    idx_map = {cls: i for i, cls in enumerate(classes)}

    # Compute argmax labels
    best_idx = np.argmax(proba, axis=1)
    labels = [classes[i] for i in best_idx]

    df["label"] = labels

    # Add probability columns for convenience
    def col_for(cls_name):
        return proba[:, idx_map[cls_name]] if cls_name in idx_map else None

    for cls in ["High", "Medium", "Low"]:
        col = col_for(cls)
        if col is not None:
            df[f"proba_{cls}"] = col
        else:
            # If class name not present, just fill zeros
            df[f"proba_{cls}"] = 0.0

    print("Saving autolabeled file...")
    df.to_csv(OUT_PATH, index=False)
    print("Done!")
    print(f"Saved autolabeled dataset to: {OUT_PATH}")

if __name__ == "__main__":
    main()
