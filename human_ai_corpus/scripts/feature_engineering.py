"""
Phase 2 - Linguistic Feature Engineering

Extract classic NLP features from each response using spaCy + NLTK-style counts.
Writes:
  data/processed/features.csv
  data/processed/feature_matrix.csv   (numeric only + response_id/source_type)
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import spacy

BASE = Path(__file__).resolve().parents[1]
CORPUS = BASE / "data" / "processed" / "corpus.csv"
OUT_FEATURES = BASE / "data" / "processed" / "features.csv"
OUT_MATRIX = BASE / "data" / "processed" / "feature_matrix.csv"

HEDGES = {
    "might", "maybe", "perhaps", "possibly", "probably", "seems", "seem",
    "i think", "i feel", "kind of", "sort of", "not sure", "arguably",
    "likely", "apparently", "presumably", "i guess", "somewhat",
}

DISCOURSE = {
    "however", "additionally", "moreover", "furthermore", "therefore",
    "thus", "meanwhile", "nevertheless", "also", "instead", "finally",
    "first", "second", "overall", "in addition", "on the other hand",
    "for example", "in conclusion", "as a result",
}

PASSIVE_AUX = {"is", "are", "was", "were", "be", "been", "being"}


def tree_depth(token) -> int:
    if not list(token.children):
        return 1
    return 1 + max(tree_depth(c) for c in token.children)


def parse_depth(doc) -> float:
    depths = []
    for sent in doc.sents:
        roots = [t for t in sent if t.head == t]
        if not roots:
            continue
        depths.append(tree_depth(roots[0]))
    return float(np.mean(depths)) if depths else 0.0


def passive_active_ratio(doc) -> float:
    passive = 0
    active = 0
    for sent in doc.sents:
        has_passive = any(t.dep_ == "nsubjpass" or t.tag_ == "VBN" for t in sent)
        # count clauses roughly by root verbs
        verbs = [t for t in sent if t.pos_ == "VERB"]
        if not verbs:
            continue
        if has_passive:
            passive += 1
        else:
            active += 1
    total = passive + active
    return passive / total if total else 0.0


def count_phrases(text: str, phrases: set[str]) -> int:
    t = " " + text.lower() + " "
    return sum(t.count(" " + p + " ") for p in phrases)


def extract_one(nlp, text: str) -> dict:
    text = str(text).strip()
    doc = nlp(text)

    tokens = [t for t in doc if not t.is_space]
    words = [t for t in tokens if not t.is_punct]
    sents = list(doc.sents)

    n_words = len(words)
    n_sents = max(len(sents), 1)
    avg_sent_len = n_words / n_sents

    lemmas = [t.lemma_.lower() for t in words if t.is_alpha]
    ttr = (len(set(lemmas)) / len(lemmas)) if lemmas else 0.0

    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "PRON": 0}
    for t in words:
        if t.pos_ in pos_counts:
            pos_counts[t.pos_] += 1
    pos_pct = {f"pct_{k.lower()}": (v / n_words if n_words else 0.0) for k, v in pos_counts.items()}

    lower = text.lower()
    hedges = count_phrases(lower, HEDGES)
    discourse = count_phrases(lower, DISCOURSE)

    feats = {
        "avg_sentence_length": avg_sent_len,
        "num_sentences": n_sents,
        "num_words": n_words,
        "parse_tree_depth": parse_depth(doc),
        "type_token_ratio": ttr,
        "passive_ratio": passive_active_ratio(doc),
        "hedge_count": hedges,
        "hedge_rate": hedges / n_words if n_words else 0.0,
        "discourse_count": discourse,
        "discourse_rate": discourse / n_words if n_words else 0.0,
        "avg_word_length": float(np.mean([len(t.text) for t in words])) if words else 0.0,
        "punct_ratio": (sum(1 for t in tokens if t.is_punct) / len(tokens)) if tokens else 0.0,
    }
    feats.update(pos_pct)
    return feats


def main():
    if not CORPUS.exists():
        raise SystemExit(f"Missing corpus: {CORPUS}. Run create_sample_corpus.py first.")

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    df = pd.read_csv(CORPUS)
    rows = []
    for i, row in df.iterrows():
        feats = extract_one(nlp, row["response_text"])
        feats.update({
            "response_id": row["response_id"],
            "prompt_id": row["prompt_id"],
            "source_type": row["source_type"],
            "response_text": row["response_text"],
        })
        rows.append(feats)
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(df)}")

    feat_df = pd.DataFrame(rows)
    feature_cols = [
        "avg_sentence_length", "num_sentences", "num_words", "parse_tree_depth",
        "type_token_ratio", "passive_ratio", "hedge_count", "hedge_rate",
        "discourse_count", "discourse_rate", "avg_word_length", "punct_ratio",
        "pct_noun", "pct_verb", "pct_adj", "pct_adv", "pct_pron",
    ]

    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(OUT_FEATURES, index=False)

    matrix = feat_df[["response_id", "source_type"] + feature_cols].copy()
    matrix.to_csv(OUT_MATRIX, index=False)

    print(f"Saved features -> {OUT_FEATURES}")
    print(f"Saved matrix   -> {OUT_MATRIX}")
    print("\nMean by source_type:")
    print(matrix.groupby("source_type")[feature_cols].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
