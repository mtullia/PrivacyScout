# combine_datasets.py
"""
Combine:
    - data/train_labeled.csv       (manual labels, columns: text,label)
    - data/train_autolabeled_50k.csv (auto labels, has policy_text, label, etc.)

into:
    - data/train_final.csv

Usage:
    python combine_datasets.py
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

LABELED_PATH = DATA_DIR / "train_labeled.csv"
AUTO_PATH = DATA_DIR / "train_autolabeled_50k.csv"
OUT_PATH = DATA_DIR / "train_final.csv"

def main():
    print("[*] Loading datasets...")

    if not LABELED_PATH.exists():
        raise SystemExit(f"Missing manual labeled file: {LABELED_PATH}")
    if not AUTO_PATH.exists():
        raise SystemExit(f"Missing auto-labeled file: {AUTO_PATH}")

    labeled = pd.read_csv(LABELED_PATH)
    auto = pd.read_csv(AUTO_PATH)

    print(f"[*] Labeled rows: {len(labeled)}")
    print(f"    Labeled columns: {list(labeled.columns)}")
    print(f"[*] Autolabeled rows: {len(auto)}")
    print(f"    Autolabeled columns: {list(auto.columns)}")

    # Basic sanity checks
    for col in ["text", "label"]:
        if col not in labeled.columns:
            raise ValueError("Column missing from labeled dataset: " + col)

    if "policy_text" not in auto.columns:
        raise ValueError("Column missing from autolabeled dataset: policy_text")
    if "label" not in auto.columns:
        raise ValueError("Column missing from autolabeled dataset: label")

    # Normalize column names
    auto = auto.rename(columns={"policy_text": "text"})

    # Add source columns so we can track provenance
    labeled["source"] = "manual_486"
    auto["source"] = "auto_50k"

    # Keep text and label always present; keep other columns from auto as extras
    # This will give at least: text,label,source,...
    cols_union = list(dict.fromkeys(["text", "label", "source"] + list(auto.columns)))
    labeled = labeled.reindex(columns=cols_union, fill_value=None)
    auto = auto.reindex(columns=cols_union, fill_value=None)

    df = pd.concat([labeled, auto], ignore_index=True)
    print(f"[*] Combined rows: {len(df)}")

    df.to_csv(OUT_PATH, index=False)
    print(f"[+] Saved combined dataset to: {OUT_PATH}")

if __name__ == "__main__":
    main()
