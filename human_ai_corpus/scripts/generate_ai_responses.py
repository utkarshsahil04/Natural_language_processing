"""
Phase 1 - Generate AI responses for every prompt in prompts.csv

Usage:
  1. Copy .env.example to .env and put GEMINI_API_KEY there
  2. python scripts/generate_ai_responses.py
  3. Optional: python scripts/generate_ai_responses.py --per-prompt 5

Output: data/raw/ai_responses.csv
Columns: response_id, prompt_id, prompt, response_text, source_type
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from google import genai
except ImportError:
    print("Install google-genai: pip install google-genai")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPTS_PATH = BASE_DIR / "data" / "raw" / "prompts.csv"
OUT_PATH = BASE_DIR / "data" / "raw" / "ai_responses.csv"
ENV_PATH = BASE_DIR / ".env"
MODEL_NAME = "gemini-2.0-flash"

SYSTEM = (
    "You are a helpful assistant answering a user question. "
    "Reply in natural English. Keep answers between 60 and 180 words. "
    "Do not mention that you are an AI unless asked."
)


def get_client():
    load_dotenv(ENV_PATH, override=True)
    key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not key or key.startswith("your_"):
        print("Add your Gemini API key to .env as GEMINI_API_KEY=...")
        print("Get one at: https://aistudio.google.com/apikey")
        sys.exit(1)
    os.environ["GEMINI_API_KEY"] = key
    return genai.Client()


def generate_one(client, prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"{SYSTEM}\n\nUser question:\n{prompt}",
    )
    return (response.text or "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-prompt", type=int, default=5,
                        help="How many AI replies per prompt (default 5)")
    parser.add_argument("--sleep", type=float, default=1.2,
                        help="Seconds to wait between API calls")
    args = parser.parse_args()

    if not PROMPTS_PATH.exists():
        print(f"Missing prompts file: {PROMPTS_PATH}")
        sys.exit(1)

    prompts = pd.read_csv(PROMPTS_PATH)
    client = get_client()

    rows = []
    counter = 1
    total = len(prompts) * args.per_prompt
    print(f"Generating {total} AI responses ({args.per_prompt} per prompt)...")

    with tqdm(total=total) as bar:
        for _, row in prompts.iterrows():
            for i in range(args.per_prompt):
                try:
                    text = generate_one(client, row["prompt"])
                except Exception as e:
                    print(f"\nError on {row['prompt_id']}#{i+1}: {e}")
                    text = ""

                rows.append({
                    "response_id": f"AI_{counter:04d}",
                    "prompt_id": row["prompt_id"],
                    "prompt": row["prompt"],
                    "response_text": text,
                    "source_type": "AI",
                })
                counter += 1
                bar.update(1)
                time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    df = df[df["response_text"].astype(str).str.len() > 20]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} AI responses -> {OUT_PATH}")


if __name__ == "__main__":
    main()
