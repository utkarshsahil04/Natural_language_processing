"""
Phase 2 - Automatic BIO Annotation
Uses spaCy rules + NER to tag each token as:
  B-LOC / I-LOC  -- Location span
  B-ACT / I-ACT  -- Action-Needed span
  O              -- Neither

Input:  data/processed/corpus.csv  (or sample_corpus.csv)
Output: data/processed/annotated_bio.csv
"""

import os, sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── spaCy ─────────────────────────────────────────────────────────────────────
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded: en_core_web_sm")
    except OSError:
        print("Downloading spaCy English model ...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    print("ERROR: pip install spacy"); sys.exit(1)

# ── Annotation vocabulary ─────────────────────────────────────────────────────
LOC_PREPS = {"in","at","near","by","from","around","along","across",
             "beside","outside","inside","behind","beneath","between",
             "within","onto","into","towards","toward","off"}

ACT_VERBS = {"need","needs","needed","send","sent","require","requires",
             "required","help","rescue","evacuate","evacuated","dispatch",
             "deploy","bring","call","request","requests","requested",
             "provide","provided","save","saves","saved","assist",
             "assists","assisted","supply","supplies","supplied",
             "get","gets","urgently","emergency","trapped","stranded",
             "missing","injured","critical","sos","mayday","airlift",
             "evacuate","relief","respond","react","mobilize"}

LOC_WORDS = {"area","district","region","zone","sector","block","ward",
             "village","town","city","road","bridge","river","colony",
             "nagar","marg","lane","chowk","ghat","station","coast",
             "highway","street","avenue","park","island","valley",
             "mountain","hill","beach","shore","dam","camp","shelter",
             "junction","crossing","flyover","underpass","border","port"}

def annotate_message(msg_id, text):
    doc = nlp(str(text)[:500])

    # NER entity map: token_idx -> position_in_span (0=start)
    ent_map = {}
    for ent in doc.ents:
        if ent.label_ in ("GPE","LOC","FAC","ORG"):
            for i, tok in enumerate(ent):
                ent_map[tok.i] = i

    rows = []
    prev_tag = "O"

    for i, tok in enumerate(doc):
        tag = "O"
        prev_tok = doc[i-1] if i > 0 else None
        next_tok = doc[i+1] if i < len(doc)-1 else None
        lem      = tok.lemma_.lower()
        prev_lem = prev_tok.lemma_.lower() if prev_tok else ""

        # Rule 1: Named entity (GPE/LOC/FAC) → LOCATION
        if tok.i in ent_map:
            tag = "B-LOC" if ent_map[tok.i] == 0 else "I-LOC"

        # Rule 2: Proper noun / noun after location preposition → B-LOC
        elif prev_lem in LOC_PREPS and prev_tag == "O":
            if tok.pos_ in ("PROPN","NOUN","NUM","ADJ"):
                tag = "B-LOC"

        # Rule 3: Continue LOC span (compound / flat nouns)
        elif prev_tag in ("B-LOC","I-LOC"):
            if tok.dep_ in ("compound","flat","nmod","appos","conj","pobj") \
               or tok.pos_ in ("NUM",) \
               or lem in LOC_WORDS \
               or tok.text in ("-","/","&"):
                tag = "I-LOC"

        # Rule 4: Standalone location word → B-LOC
        elif lem in LOC_WORDS and tok.pos_ in ("NOUN","PROPN") \
             and tok.dep_ in ("pobj","nsubj","attr"):
            tag = "B-LOC"

        # Rule 5: Action / imperative verb → B-ACT
        elif lem in ACT_VERBS \
             or (tok.pos_ == "VERB" and tok.dep_ == "ROOT" and tok.tag_ in ("VB","VBP","VBZ")):
            tag = "B-ACT"

        # Rule 6: Direct object / modifier of action verb → I-ACT
        elif prev_tag in ("B-ACT","I-ACT"):
            if tok.dep_ in ("dobj","nsubj","attr","advmod","amod","compound") \
               and tok.pos_ in ("NOUN","PROPN","ADJ","ADV"):
                tag = "I-ACT"

        rows.append({
            "message_id" : msg_id,
            "token_idx"  : i,
            "token"      : tok.text,
            "pos"        : tok.pos_,
            "tag_"       : tok.tag_,
            "dep"        : tok.dep_,
            "lemma"      : lem,
            "is_ent"     : int(tok.i in ent_map),
            "bio_tag"    : tag,
        })
        prev_tag = tag

    return rows

def main():
    corpus_path = os.path.join(PROC_DIR, "corpus.csv")
    if not os.path.exists(corpus_path):
        print("corpus.csv not found. Creating sample corpus ...")
        import subprocess
        subprocess.run([sys.executable,
                        os.path.join(BASE_DIR,"scripts","create_sample_corpus.py")])

    df = pd.read_csv(corpus_path)
    print(f"Loaded {len(df)} messages")

    all_rows = []
    for i, row in df.iterrows():
        if i % 200 == 0:
            print(f"  Annotating {i}/{len(df)} ...")
        rows = annotate_message(str(row["message_id"]), str(row["raw_text"]))
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)
    out_path = os.path.join(PROC_DIR, "annotated_bio.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nSaved: {out_path}")
    print(f"Total tokens : {len(out_df)}")
    print("\nBIO Tag distribution:")
    print(out_df["bio_tag"].value_counts().to_string())

if __name__ == "__main__":
    main()
