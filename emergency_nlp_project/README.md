# Emergency Message Corpus & Syntactic Pattern Detection

> A classic ML + NLP project that constructs a corpus of disaster/emergency messages
> and identifies syntactic patterns indicating **locations** and **required actions**.
> Built strictly with classical ML — no deep learning, no transformers.

---

## Problem Statement

*"Construct a corpus of disaster/emergency messages and identify syntactic patterns
that indicate locations and required actions."*

Each message is tokenised and every token is classified as one of:

| Label | Meaning | Example |
|-------|---------|---------|
| `B-LOC` / `I-LOC` | Beginning / inside of a **Location** span | *"near **Kalindi Kunj bridge**"* |
| `B-ACT` / `I-ACT` | Beginning / inside of an **Action-Needed** span | *"**send rescue teams**"* |
| `O` | Neither | *"the"*, *"is"*, *"a"* |

---

## Project Structure

```
emergency_nlp_project/
├── data/
│   ├── raw/                          <- Kaggle CSVs land here after download
│   └── processed/
│       ├── corpus.csv                <- Phase 1  | message_id, raw_text, source
│       ├── annotated_bio.csv         <- Phase 2  | BIO-tagged tokens (1,376 tokens)
│       ├── features.csv              <- Phase 3  | 1,376 x 13 feature matrix
│       └── labels.csv                <- Phase 3  | encoded BIO labels
├── models/
│   ├── encoders.pkl                  <- LabelEncoders for POS / DEP / BIO
│   ├── decision_tree.pkl             <- Phase 4  | Baseline DT
│   ├── random_forest.pkl             <- Phase 4  | Best model (F1=0.934)
│   └── adaboost.pkl                  <- Phase 4  | AdaBoost
├── scripts/
│   ├── create_sample_corpus.py       <- Phase 1b | 120 messages, no Kaggle needed
│   ├── download_data.py              <- Phase 1  | Downloads from Kaggle (~3,200 msgs)
│   ├── annotate_bio.py               <- Phase 2  | spaCy rule-based BIO tagger
│   ├── feature_engineering.py        <- Phase 3  | Extracts 13 features per token
│   ├── train_models.py               <- Phase 4  | Trains DT / RF / AdaBoost
│   └── analyze_results.py            <- Phase 5  | Patterns, importances, report
├── notebooks/
│   └── phase1_corpus_collection.ipynb
├── app/
│   ├── app.py                        <- Flask API backend
│   └── templates/
│       └── index.html                <- Dark-themed testing UI
├── reports/
│   ├── project_report.md             <- Submission-ready report
│   ├── model_comparison.png
│   ├── cm_decision_tree.png
│   ├── cm_random_forest.png
│   ├── cm_adaboost.png
│   ├── fi_decision_tree.png
│   ├── fi_random_forest.png
│   ├── fi_adaboost.png
│   ├── pattern_pos_dist.png
│   └── feature_importance_table.csv
├── run_pipeline.py                   <- One-command runner for all phases
└── requirements.txt
```

---

## Quick Start

### Option A — No Kaggle (recommended for first run)
```bash
cd emergency_nlp_project
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Runs all 5 phases using the built-in 120-message sample corpus
python run_pipeline.py --skip-download

# Launch the testing UI
python app/app.py
# Open: http://localhost:5000
```

### Option B — With Kaggle (~3,200 real disaster tweets)
```bash
# 1. Get your API token: https://www.kaggle.com/settings -> API -> Create New Token
# 2. Place kaggle.json at: C:/Users/<You>/.kaggle/kaggle.json
python run_pipeline.py
python app/app.py
```

---

## Dataset

| Source | Messages | Description |
|--------|----------|-------------|
| [Kaggle — NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) | ~3,200 | Real disaster tweets (`target=1`) |
| Manual Hinglish / Indian context | 35 | Floods, earthquakes, fires in India |
| Built-in sample corpus | 120 | Ready to use without Kaggle account |

---

## The 5 Phases

### Phase 1 — Corpus Collection
- Downloads `train.csv` from Kaggle and filters `target == 1`
- Appends 35 Hinglish/Indian emergency messages
- Output: `data/processed/corpus.csv`  (`message_id`, `raw_text`, `source`)

### Phase 2 — BIO Annotation
Auto-annotates each token using spaCy rules:

| Rule | Tag applied |
|------|------------|
| Named entity of type `GPE` / `LOC` / `FAC` | `B-LOC` / `I-LOC` |
| Token immediately after a location preposition (*in, at, near, by, from…*) | `B-LOC` |
| Compound/flat noun continuing a LOC span | `I-LOC` |
| Imperative verb lemma (*need, send, rescue, evacuate, dispatch…*) | `B-ACT` |
| Direct object of an action verb | `I-ACT` |

**Results on 120 messages (1,376 tokens):**

| Tag | Count | % |
|-----|-------|---|
| O | 756 | 54.9 % |
| I-LOC | 205 | 14.9 % |
| B-LOC | 185 | 13.4 % |
| B-ACT | 134 | 9.7 % |
| I-ACT | 96 | 7.0 % |

### Phase 3 — Feature Engineering
13 numerical features per token (extracted with **spaCy `en_core_web_sm`**):

| # | Feature | Description |
|---|---------|-------------|
| 1 | `pos_enc` | POS tag (label-encoded) |
| 2 | `dep_enc` | Dependency relation (label-encoded) |
| 3 | `prev_pos_enc` | POS of the previous token |
| 4 | `next_pos_enc` | POS of the next token |
| 5 | `prev_is_prep` | Previous token is a location preposition (0/1) |
| 6 | `next_is_noun` | Next token is `NOUN` or `PROPN` (0/1) |
| 7 | `is_act_verb` | Token lemma is an action verb (0/1) |
| 8 | `is_loc_word` | Token lemma is a location indicator word (0/1) |
| 9 | `is_capitalized` | Token starts with uppercase (0/1) |
| 10 | `is_ent` | Token is a named entity (0/1) |
| 11 | `token_pos_norm` | Position in sentence, normalised 0.0–1.0 |
| 12 | `sentence_len` | Sentence length / 50 |
| 13 | `is_number` | Token is numeric (0/1) |

### Phase 4 — Modeling

Three classical ML classifiers, all handling class imbalance via `class_weight='balanced'`:

| Model | Params | Role |
|-------|--------|------|
| **Decision Tree** | `max_depth=12`, `min_samples_leaf=3` | Interpretable baseline |
| **Random Forest** | `n_estimators=300`, `max_depth=15` | Bagging ensemble |
| **AdaBoost** | `n_estimators=200`, `lr=0.5`, base `DT(depth=4)` | Boosting ensemble |

**Results (25 % held-out test set):**

| Model | Macro F1 | Accuracy | 5-Fold CV F1 |
|-------|:--------:|:--------:|:------------:|
| Decision Tree | 0.864 | 91 % | 0.836 ± 0.026 |
| **Random Forest** ⭐ | **0.934** | **96 %** | **0.924 ± 0.019** |
| AdaBoost | 0.890 | 93 % | 0.881 ± 0.024 |

### Phase 5 — Analysis & Report

**Top features (mean importance across all 3 models):**

| Rank | Feature | Importance |
|------|---------|:----------:|
| 1 | `is_act_verb` | 0.186 |
| 2 | `dep_enc` | 0.160 |
| 3 | `token_pos_norm` | 0.140 |
| 4 | `prev_pos_enc` | 0.127 |
| 5 | `prev_is_prep` | 0.117 |
| 6 | `is_ent` | 0.080 |

**Key syntactic patterns found:**

- **LOCATION** — 77 % of B-LOC tokens are `PROPN`; dependency roles: `compound`, `pobj`
- **ACTION-NEEDED** — 57 % of B-ACT tokens are `VERB`; dependency role: `ROOT`
- Top action lemmas across corpus: *need* (47), *send* (20), *rescue* (20), *urgently* (13)

See `reports/project_report.md` for the full submission-ready writeup.

---

## Testing UI

```bash
python app/app.py
```

Open **http://localhost:5000** to:
- Paste any disaster message and see tokens highlighted
  - 🔵 **Blue underline** = LOCATION span
  - 🟠 **Orange underline** = ACTION-NEEDED span
- Switch between Decision Tree / Random Forest / AdaBoost
- View per-token POS, DEP and BIO label in a detail table
- Click example chips to test instantly (English + Hinglish)
- Re-train models with one button click (no terminal needed)

---

## ML Stack (Classic Only — No Deep Learning)

| Library | Version | Use |
|---------|---------|-----|
| `spaCy` | ≥ 3.7 | POS tagging, dependency parsing, NER |
| `scikit-learn` | ≥ 1.3 | Decision Tree, Random Forest, AdaBoost |
| `pandas` / `numpy` | — | Data wrangling |
| `matplotlib` / `seaborn` | — | Plots and confusion matrices |
| `Flask` | — | Testing UI backend |
| `kaggle` | — | Dataset download |

---

## References

1. Imran et al. (2016). *CrisisNLP: A Multilingual Dataset*. AAAI.
2. Kaggle Competition: [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
3. Honnibal & Montani (2020). *spaCy: Industrial-strength NLP*. Explosion.
4. Breiman (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
5. Freund & Schapire (1997). *A Decision-Theoretic Generalization of AdaBoost*. JCSS.
