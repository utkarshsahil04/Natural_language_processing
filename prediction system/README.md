# ?? PriorityPulse: Ticket Priority Classifier

An NLP system that predicts IT support ticket priority (**Low**, **Medium**, **High**) using a fine-tuned **DistilBERT** model. Includes an interactive dark-mode Web UI and FastAPI backend.

---

## ? Quick Start (Run with Docker)

```bash
docker compose up --build -d
```
Open **[http://localhost:8000](http://localhost:8000)** to use the Web UI.

---

## ?? Local Setup (Without Docker)

```bash
# 1. Create and activate virtual environment
python -m venv .env
.\.env\Scripts\activate      # On Linux/Mac: source .env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Web UI & API Server
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

---

## ??? Training & CLI Testing

```bash
# Prepare dataset
python src/prepare_data.py

# Train model on GPU (saves to model/priority-distilbert/)
python src/train.py --epochs 4

# Test predictions via CLI
python src/predict.py --sample
python src/predict.py --type Incident --subject "DB Down" --body "Server unresponsive"
```

---

## ?? Results on Test Set

- **Model:** DistilBERT (`distilbert-base-uncased`)
- **Dataset:** 16,338 English IT Support Tickets
- **Accuracy:** ~58% | **Macro F1:** 0.59
- **Latency:** ~11 ms on RTX 4060 GPU

---

## ?? Project Structure

```text
+-- api/                  # FastAPI backend + Web UI (static HTML/CSS/JS)
+-- data/                 # Processed train/val/test CSVs
+-- model/                # Fine-tuned DistilBERT weights
+-- src/
¦   +-- prepare_data.py   # Dataset preprocessing & splitting
¦   +-- train.py          # PyTorch training script
¦   +-- predict.py        # CLI prediction script
+-- Dockerfile            # Container configuration
+-- docker-compose.yml    # One-click Docker deployment
+-- requirements.txt      # Python dependencies
```
