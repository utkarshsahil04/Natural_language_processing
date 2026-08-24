# Natural Language Processing (NLP) Coursework & Projects

This repository contains coursework, laboratory experiments, and production-ready NLP applications developed as part of the Natural Language Processing curriculum.

---

## 📁 Repository Structure

```
NLP/
├── LAB_01/
│   └── lab_01.ipynb                 # NLTK tokenization, stemming, lemmatization & stopword removal
│
├── chatbot/
│   ├── iilm_chatbot.py              # Context-aware RAG-style terminal chatbot using Google Gemini API
│   ├── iilm_knowledge.txt           # Structured knowledge base for IILM University
│   ├── nltk_first.ipynb             # Introductory NLTK exploration & experiments
│   ├── requirements.txt             # Dependencies for chatbot
│   └── README.md                    # Chatbot setup & usage guide
│
├── prediction system/
│   ├── api/
│   │   ├── main.py                  # FastAPI REST API backend with rate-limiting & health endpoints
│   │   └── static/                  # Modern Glassmorphic Web UI dashboard (HTML5/CSS3/JS)
│   ├── data/                        # Processed stratified train/val/test splits & raw dataset
│   ├── model/                       # Fine-tuned DistilBERT weights, tokenizer & configuration
│   ├── src/
│   │   ├── prepare_data.py          # Data cleaning, multilingual handling & stratified splitting
│   │   ├── train.py                 # DistilBERT fine-tuning pipeline with early stopping & metrics
│   │   └── predict.py               # Batch & single ticket inference engine
│   ├── Dockerfile                   # Production-ready multi-stage Docker configuration
│   ├── docker-compose.yml           # One-command container orchestration
│   ├── requirements.txt             # Production API dependencies
│   ├── requirements-train.txt       # Training dependencies (PyTorch, Transformers, Datasets)
│   └── README.md                    # Prediction System architecture & deployment guide
│
└── README.md                        # Master repository documentation
```

---

## 🚀 Projects Overview

### 1. 🎫 [Ticket Priority Prediction System](prediction%20system/)
An end-to-end MLOps NLP application that classifies customer support tickets into priority levels (**Urgent**, **High**, **Medium**, **Low**):
- **Model**: Fine-tuned `distilbert-base-uncased` transformer on multilingual IT support tickets.
- **Backend**: High-performance FastAPI with JSON request/response schema validation and batch inference.
- **Frontend**: Responsive, modern glassmorphic dashboard featuring live confidence bar distributions, batch file uploads, and sample loaders.
- **Deployment**: Dockerized with multi-stage build and health check support.

### 2. 🤖 [IILM University Chatbot](chatbot/)
A multi-turn conversational AI powered by **Google Gemini 2.0 Flash** and grounded on a specialized university knowledge base:
- Custom knowledge grounding (RAG-style retrieval from domain documents).
- Multi-turn conversation history management.
- Secure environment configuration via `.env`.

### 3. 🧪 [Lab Experiments & Notebooks](LAB_01/)
- **Lab 01**: Text preprocessing pipeline including sentence/word tokenization, stemming (Porter & Snowball), lemmatization (WordNet), and stopword filtering.

---

## 🛠️ Tech Stack

- **NLP / Deep Learning**: PyTorch, Hugging Face `transformers`, `datasets`, `evaluate`, NLTK, Scikit-learn
- **LLMs & GenAI**: Google Gemini API (`google-genai`), Python-dotenv
- **API & Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: HTML5, Vanilla Modern CSS (CSS Grid, Glassmorphism, Micro-animations), JavaScript ES6+
- **DevOps & Containers**: Docker, Docker Compose, Git

---

## 📜 License
Educational and academic coursework repository.
