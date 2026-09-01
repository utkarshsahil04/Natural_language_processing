# Emergency Message Corpus & Syntactic Pattern Detection

NLP + Classic ML project to detect **locations** and **actions needed** in disaster messages.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
# Run full pipeline (uses built-in sample corpus)
python run_pipeline.py --skip-download

# OR with Kaggle data (add kaggle.json first, then accept rules at kaggle.com/competitions/nlp-getting-started/rules)
python run_pipeline.py

# Start the UI
python app/app.py
# Open: http://localhost:5000
```

## What it does

1. **Corpus** — 2,911 real disaster tweets (Kaggle) + 35 Hinglish messages
2. **Annotation** — auto-tags each token as `B-LOC`, `I-LOC`, `B-ACT`, `I-ACT`, or `O`
3. **Features** — 13 spaCy-based features per token (POS, dependency, NER, etc.)
4. **Models** — Decision Tree, Random Forest, AdaBoost (best: RF, F1 = 0.677)
5. **UI** — paste any emergency message, see highlighted location/action spans

## Results

| Model | Macro F1 |
|-------|----------|
| Decision Tree | 0.470 |
| Random Forest | 0.677 |
| AdaBoost | 0.621 |

## Tech Stack

- `spaCy` — POS tagging, dependency parsing, NER
- `scikit-learn` — classical ML models
- `Flask` — testing UI
- `Kaggle API` — dataset download
