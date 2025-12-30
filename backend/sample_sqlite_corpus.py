# sample_sqlite_corpus.py
"""
Sample 50,000 privacy policies from the big SQLite corpus and save them
as data/train_unlabeled_50k.csv.

Usage (from backend folder):
    python sample_sqlite_corpus.py

Assumes:
    - SQLite database file is next to this script or you update DB_PATH.
"""

from pathlib import Path
import sqlite3
import csv

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Update this if your DB lives somewhere else
DB_PATH = HERE / "release_db.sqlite"

OUT_PATH = DATA_DIR / "train_unlabeled_50k.csv"
LIMIT = 50000

QUERY = """
SELECT
    ps.id AS snapshot_id,
    s.domain,
    t.policy_text,
    t.length,
    t.flesch_kincaid,
    t.smog,
    t.flesch_ease
FROM policy_snapshots ps
JOIN sites s ON s.id = ps.site_id
JOIN policy_texts t ON t.id = ps.policy_text_id
LIMIT ?;
"""

def main():
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found at: {DB_PATH}")

    print(f"[+] Connecting to {DB_PATH} ...")
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    print(f"[+] Running query (LIMIT {LIMIT}) ...")
    cur.execute(QUERY, (LIMIT,))

    cols = [d[0] for d in cur.description]
    print(f"[+] Columns: {cols}")

    print(f"[+] Writing CSV to {OUT_PATH} ...")
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        count = 0
        for row in cur:
            w.writerow(row)
            count += 1
            if count % 5000 == 0:
                print(f"    wrote {count} rows...")

    conn.close()
    print(f"[+] Done. Wrote {count} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
