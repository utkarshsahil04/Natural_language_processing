"""
Prepare the ticket dataset for priority classification.

- Filters to English-language rows only
- Builds a single input text field: "[TYPE] subject | body"
- Encodes priority labels (low=0, medium=1, high=2)
- Splits into train/val/test (80/10/10, stratified)
- Saves as CSVs the training script can load directly
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

RAW_PATH = "data/raw_tickets.csv"
OUT_DIR = "data"
LABEL2ID = {"low": 0, "medium": 1, "high": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def build_text(row):
    ticket_type = row["type"] if pd.notna(row["type"]) else ""
    subject = row["subject"] if pd.notna(row["subject"]) else ""
    body = row["body"] if pd.notna(row["body"]) else ""
    return f"[{ticket_type}] {subject} | {body}".strip()

def main():
    df = pd.read_csv(RAW_PATH)

    # Keep English only
    df = df[df["language"] == "en"].copy()

    # Build combined text input
    df["text"] = df.apply(build_text, axis=1)

    # Encode labels
    df["label"] = df["priority"].map(LABEL2ID)

    # Drop anything that failed to map (shouldn't happen, but be safe)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)

    df = df[["text", "label", "priority"]]

    # Stratified split: 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print(f"Total English rows: {len(df)}")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print("\nLabel distribution (train):")
    print(train_df["priority"].value_counts())

if __name__ == "__main__":
    main()
