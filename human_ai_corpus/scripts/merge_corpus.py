"""
Phase 1 - Merge human + AI CSVs into one corpus

Usage:
  python scripts/merge_corpus.py

Looks for:
  data/raw/human_responses.csv
  data/raw/ai_responses.csv

Writes:
  data/processed/corpus.csv
  Columns: response_id, prompt_id, prompt, response_text, source_type
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW = BASE_DIR / "data" / "raw"
OUT = BASE_DIR / "data" / "processed" / "corpus.csv"

REQUIRED = ["response_id", "prompt_id", "prompt", "response_text", "source_type"]


def load_one(path: Path, expected_source: str) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing: {path}")
        return pd.DataFrame(columns=REQUIRED)

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = df[REQUIRED].copy()
    df["response_text"] = df["response_text"].astype(str).str.strip()
    df = df[df["response_text"].str.len() > 20]
    df["source_type"] = df["source_type"].astype(str).str.strip()

    # soft check
    bad = ~df["source_type"].isin(["human", "AI", "Human", "ai"])
    if bad.any():
        print(f"Warning: {bad.sum()} rows in {path.name} have odd source_type")

    df["source_type"] = df["source_type"].str.replace("Human", "human", regex=False)
    df["source_type"] = df["source_type"].str.replace("ai", "AI", regex=False)
    return df


def main():
    human = load_one(RAW / "human_responses.csv", "human")
    ai = load_one(RAW / "ai_responses.csv", "AI")

    corpus = pd.concat([human, ai], ignore_index=True)
    if corpus.empty:
        print("No data to merge. Add human_responses.csv and/or ai_responses.csv")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(OUT, index=False)

    print(f"Saved corpus -> {OUT}")
    print(corpus["source_type"].value_counts().to_string())
    print(f"Total rows: {len(corpus)}")


if __name__ == "__main__":
    main()
