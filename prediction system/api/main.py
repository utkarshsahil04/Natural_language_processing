import os
import time
from typing import List, Optional
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model/priority-distilbert"
ID2LABEL = {0: "low", 1: "medium", 2: "high"}
LABEL2ID = {"low": 0, "medium": 1, "high": 2}

app = FastAPI(
    title="Ticket Priority Prediction API",
    description="Fine-tuned DistilBERT inference service for organizational ticket priority classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
state = {
    "model": None,
    "tokenizer": None,
    "device": "cpu",
    "device_name": "CPU",
    "model_name": "DistilBERT (Fine-Tuned)",
    "loaded": False,
}


def load_model_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model path '{MODEL_PATH}' not found. Please train model first.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print(f"Loading model on {device} ({device_name})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    state["model"] = model
    state["tokenizer"] = tokenizer
    state["device"] = device
    state["device_name"] = device_name
    state["loaded"] = True
    print("Model loaded successfully!")


@app.on_event("startup")
def startup_event():
    load_model_pipeline()


class TicketInput(BaseModel):
    type: Optional[str] = Field(default="", description="Ticket type (Incident, Request, Problem, Change)")
    subject: Optional[str] = Field(default="", description="Ticket subject/title")
    body: Optional[str] = Field(default="", description="Detailed message body")
    text: Optional[str] = Field(default="", description="Pre-formatted input text override")


class BatchTicketInput(BaseModel):
    tickets: List[TicketInput]


def format_text(ticket: TicketInput) -> str:
    if ticket.text and ticket.text.strip():
        return ticket.text.strip()
    ticket_type = (ticket.type or "").strip()
    subject = (ticket.subject or "").strip()
    body = (ticket.body or "").strip()
    return f"[{ticket_type}] {subject} | {body}".strip()


def run_inference(text: str):
    start_time = time.perf_counter()
    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()
        pred_id = torch.argmax(logits, dim=1).item()

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    if isinstance(probs, float):
        probs = [probs]

    return {
        "text": text,
        "predicted_priority": ID2LABEL[pred_id],
        "confidence": round(probs[pred_id], 4),
        "probabilities": {
            "low": round(probs[0], 4),
            "medium": round(probs[1], 4),
            "high": round(probs[2], 4),
        },
        "latency_ms": elapsed_ms,
        "device": state["device_name"],
    }


@app.get("/api/health")
@app.get("/api/info")
def get_info():
    return {
        "status": "healthy" if state["loaded"] else "initializing",
        "device": state["device"],
        "device_name": state["device_name"],
        "cuda_available": torch.cuda.is_available(),
        "model_name": state["model_name"],
        "classes": ["low", "medium", "high"],
    }


@app.post("/api/predict")
def predict_single(ticket: TicketInput):
    if not state["loaded"]:
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    text = format_text(ticket)
    if not text or text == "[] |":
        raise HTTPException(status_code=400, detail="Ticket content cannot be empty")
    
    return run_inference(text)


@app.post("/api/predict/batch")
def predict_batch(batch: BatchTicketInput):
    if not state["loaded"]:
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    results = []
    for item in batch.tickets:
        text = format_text(item)
        if text and text != "[] |":
            results.append(run_inference(text))
    return {"results": results, "count": len(results)}


@app.get("/api/samples")
def get_samples():
    return [
        {
            "id": "s1",
            "type": "Incident",
            "subject": "Production Database Cluster Outage",
            "body": "Primary PostgreSQL cluster in US-East is unresponsive. All dependent microservices throwing 500 errors. Immediate DBA escalation requested.",
            "expected": "high",
            "category": "Critical Infrastructure",
        },
        {
            "id": "s2",
            "type": "Incident",
            "subject": "Payment Processing Gateway 504 Timeout",
            "body": "Stripe & PayPal webhook endpoints are failing to acknowledge transactions. Customers unable to complete checkout.",
            "expected": "high",
            "category": "Revenue Impact",
        },
        {
            "id": "s3",
            "type": "Problem",
            "subject": "Memory leak causing periodic container restart",
            "body": "Auth microservice pod restarts every 6 hours due to OOM kill. Investigating heap dump.",
            "expected": "medium",
            "category": "Backend Stability",
        },
        {
            "id": "s4",
            "type": "Request",
            "subject": "VPN Access & Active Directory Password Reset",
            "body": "User locked out after returning from leave. Needs MFA re-enrollment and domain password reset.",
            "expected": "medium",
            "category": "IT Support",
        },
        {
            "id": "s5",
            "type": "Request",
            "subject": "Request for dual 27-inch monitor arm",
            "body": "Ergonomic equipment request for workstation desk 4B on 2nd floor.",
            "expected": "low",
            "category": "Workplace Facilities",
        },
        {
            "id": "s6",
            "type": "Incident",
            "subject": "Office Pantry Printer Paper Jam",
            "body": "HP LaserJet in Breakroom 3 is flashing tray 2 paper jam error code 13.00.",
            "expected": "low",
            "category": "Office Equipment",
        },
    ]


# Mount static frontend
os.makedirs("api/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="api/static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("api/static/index.html")
