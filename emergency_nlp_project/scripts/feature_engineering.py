"""
Phase 3 - Feature Engineering
Extracts 13 numerical features per token from annotated_bio.csv.

Features:
  1.  pos_enc          POS tag (label-encoded)
  2.  dep_enc          Dependency relation (label-encoded)
  3.  prev_pos_enc     Previous token POS
  4.  next_pos_enc     Next token POS
  5.  prev_is_prep     Previous token is a location preposition (0/1)
  6.  next_is_noun     Next token is noun/proper noun (0/1)
  7.  is_act_verb      Token lemma is an action verb (0/1)
  8.  is_loc_word      Token lemma is a location indicator word (0/1)
  9.  is_capitalized   Token starts with uppercase (0/1)
  10. is_ent           Token is a named entity (0/1)
  11. token_pos_norm   Position in sentence normalized 0.0-1.0
  12. sentence_len     Sentence length / 50 (normalized)
  13. is_number        Token is numeric (0/1)

Output:
  data/processed/features.csv
  data/processed/labels.csv
  models/encoders.pkl
"""

import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LOC_PREPS = {"in","at","near","by","from","around","along","across","beside",
             "outside","inside","behind","beneath","between","within","onto","into","towards"}
ACT_VERBS = {"need","send","require","help","rescue","evacuate","dispatch","deploy",
             "bring","call","request","provide","save","assist","supply","get",
             "airlift","mobilize","respond","relief","recover"}
LOC_WORDS = {"area","district","region","zone","sector","block","ward","village","town",
             "city","road","bridge","river","colony","nagar","marg","lane","chowk",
             "ghat","station","coast","highway","street","avenue","park","island",
             "valley","mountain","hill","beach","shore","dam","camp","shelter",
             "junction","crossing","flyover","border","port"}

FEATURE_NAMES = [
    "pos_enc","dep_enc","prev_pos_enc","next_pos_enc",
    "prev_is_prep","next_is_noun","is_act_verb","is_loc_word",
    "is_capitalized","is_ent","token_pos_norm","sentence_len","is_number"
]

def compute_features(df):
    pos_le = LabelEncoder()
    dep_le = LabelEncoder()
    bio_le = LabelEncoder()

    all_pos = pd.concat([df["pos"], pd.Series(["X","NOUN","PROPN","VERB","ADJ","ADV","NUM","PUNCT","ADP","DET"])]).fillna("X")
    pos_le.fit(all_pos)
    dep_le.fit(df["dep"].fillna("dep"))
    bio_le.fit(df["bio_tag"])

    df["pos_enc"] = pos_le.transform(df["pos"].fillna("X"))
    df["dep_enc"] = dep_le.transform(df["dep"].fillna("dep"))
    df["bio_enc"] = bio_le.transform(df["bio_tag"])

    features, targets = [], []

    for msg_id, grp in df.groupby("message_id", sort=False):
        grp = grp.reset_index(drop=True)
        n = len(grp)

        for i in range(n):
            row      = grp.iloc[i]
            prev_row = grp.iloc[i-1] if i > 0   else None
            next_row = grp.iloc[i+1] if i < n-1 else None

            prev_pos  = prev_row["pos"]   if prev_row is not None else "X"
            prev_lem  = prev_row["lemma"] if prev_row is not None else ""
            next_pos  = next_row["pos"]   if next_row is not None else "X"

            def enc_pos(p):
                return int(pos_le.transform([p])[0]) if p in pos_le.classes_ else 0

            feat = {
                "pos_enc"       : int(row["pos_enc"]),
                "dep_enc"       : int(row["dep_enc"]),
                "prev_pos_enc"  : enc_pos(prev_pos),
                "next_pos_enc"  : enc_pos(next_pos),
                "prev_is_prep"  : int(prev_lem in LOC_PREPS),
                "next_is_noun"  : int(next_pos in ("NOUN","PROPN")),
                "is_act_verb"   : int(row["lemma"] in ACT_VERBS),
                "is_loc_word"   : int(row["lemma"] in LOC_WORDS),
                "is_capitalized": int(len(row["token"]) > 0 and row["token"][0].isupper()),
                "is_ent"        : int(row["is_ent"]),
                "token_pos_norm": round(i / max(n-1, 1), 4),
                "sentence_len"  : round(n / 50.0, 4),
                "is_number"     : int(str(row["token"]).replace(".","").replace(",","").isdigit()),
            }
            features.append(feat)
            targets.append(int(row["bio_enc"]))

    X = pd.DataFrame(features, columns=FEATURE_NAMES)
    y = np.array(targets)
    return X, y, pos_le, dep_le, bio_le

def main():
    in_path  = os.path.join(PROC_DIR, "annotated_bio.csv")
    out_feat = os.path.join(PROC_DIR, "features.csv")
    out_y    = os.path.join(PROC_DIR, "labels.csv")
    enc_path = os.path.join(MODEL_DIR, "encoders.pkl")

    if not os.path.exists(in_path):
        print("annotated_bio.csv not found. Run annotate_bio.py first.")
        sys.exit(1)

    print("Loading annotated tokens ...")
    df = pd.read_csv(in_path)
    print(f"  Tokens loaded: {len(df)}")

    print("Extracting features ...")
    X, y, pos_le, dep_le, bio_le = compute_features(df)

    X.to_csv(out_feat, index=False)
    pd.Series(y, name="bio_enc").to_csv(out_y, index=False)

    with open(enc_path, "wb") as f:
        pickle.dump({"pos_le": pos_le, "dep_le": dep_le, "bio_le": bio_le,
                     "feature_names": FEATURE_NAMES}, f)

    print(f"\nFeatures : {out_feat}  shape={X.shape}")
    print(f"Labels   : {out_y}")
    print(f"Encoders : {enc_path}")
    print("\nLabel distribution:")
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"  {bio_le.inverse_transform([c])[0]:8s}: {n:5d} ({100*n/len(y):.1f}%)")

if __name__ == "__main__":
    main()
