# Human-AI Conversation Corpus (Problem Statement 1)

Create a corpus of human–AI interactions and analyze sentence-structure differences with **classic ML only** (no transformers).

> `emergency_nlp_project` = PS 3 (done)  
> `human_ai_corpus` = **PS 1** (this project)

---

## What this project does

1. **Corpus** — paired prompts with human-style and AI-style responses  
2. **Features** — spaCy linguistic features (syntax, lexicon, discourse, POS)  
3. **Clustering** — K-Means + Hierarchical (labels hidden)  
4. **Validation** — Decision Tree + AdaBoost + Gradient Boosting  
5. **Report** — which features separate human vs AI

---

## Quick start

```powershell
cd "d:\COllege\3rd Year\NLP\human_ai_corpus"
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_pipeline.py
```

Outputs:

| Path | Content |
|------|---------|
| `data/processed/corpus.csv` | Full corpus |
| `data/processed/feature_matrix.csv` | Numeric features |
| `data/processed/clusters.csv` | Cluster assignments |
| `reports/*.png` | Elbow, PCA, t-SNE, importance plots |
| `reports/project_report.md` | Final write-up |

---

## Folder structure

```
human_ai_corpus/
├── data/
│   ├── raw/prompts.csv
│   ├── raw/human_responses.csv
│   ├── raw/ai_responses.csv
│   └── processed/corpus.csv, features.csv, feature_matrix.csv, clusters.csv
├── scripts/
│   ├── create_sample_corpus.py
│   ├── generate_ai_responses.py
│   ├── merge_corpus.py
│   ├── feature_engineering.py
│   ├── cluster_analysis.py
│   ├── classify_validate.py
│   └── analyze_results.py
├── notebooks/phase1_corpus.ipynb
├── reports/
├── run_pipeline.py
└── README.md
```

---

## Current sample results (360 responses)

- Human: 180 | AI: 180
- **K-Means ARI vs true human/AI labels: 1.000** (clusters recovered the split)
- Hierarchical ARI: ~0.88
- Decision Tree / AdaBoost / Gradient Boosting: strong separation on these features

Important features usually include sentence length, word length, pronoun %, noun %, hedge rate, parse depth.

> Note: the bundled corpus is a **synthetic sample** for pipeline/demo.  
> For submission-quality data, collect real Reddit/Quora answers and generate AI replies with Gemini.

### Build a real corpus

```powershell
# 1) Fill data/raw/human_responses.csv (see template)
# 2) Generate AI replies
copy .env.example .env
# add GEMINI_API_KEY
python scripts/generate_ai_responses.py --per-prompt 5
python scripts/merge_corpus.py
python run_pipeline.py --skip-sample
```

---

## Phases

| Phase | Script | Status |
|------|--------|--------|
| 1 Corpus | `create_sample_corpus.py` / `generate_ai_responses.py` | Done |
| 2 Features | `feature_engineering.py` | Done |
| 3 Clustering | `cluster_analysis.py` | Done |
| 4 Classification | `classify_validate.py` | Done |
| 5 Report | `analyze_results.py` | Done |

Read the full write-up: [`reports/project_report.md`](reports/project_report.md)
