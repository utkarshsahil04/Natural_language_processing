"""
Message Priority Prediction - Inference & Testing Script

Usage:
    # Interactive mode:
    python src/predict.py

    # Single line test:
    python src/predict.py --text "[Incident] Payment gateway timeout | Customers unable to checkout"

    # Field-based test:
    python src/predict.py --type Incident --subject "VPN Connection Failed" --body "Cannot connect to internal network since morning"

    # Batch test samples:
    python src/predict.py --sample
"""

import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model/priority-distilbert"
ID2LABEL = {0: "low", 1: "medium", 2: "high"}
LABEL2ID = {"low": 0, "medium": 1, "high": 2}


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory '{model_path}' not found. Run 'python src/train.py' first.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict(text: str, model, tokenizer, device):
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

    if isinstance(probs, float):
        probs = [probs]

    return {
        "text": text,
        "predicted_priority": ID2LABEL[pred_id],
        "confidence": probs[pred_id],
        "probabilities": {
            "low": round(probs[0], 4),
            "medium": round(probs[1], 4),
            "high": round(probs[2], 4),
        },
    }


def format_input(ticket_type: str = "", subject: str = "", body: str = "") -> str:
    ticket_type = ticket_type.strip()
    subject = subject.strip()
    body = body.strip()
    return f"[{ticket_type}] {subject} | {body}".strip()


def run_sample_tests(model, tokenizer, device):
    samples = [
        ("[Incident] Database cluster offline | Primary and replica DBs unresponsive. Critical outage across all services.", "high"),
        ("[Problem] Core API experiencing 500 error spikes | Checkout service failing for 30% of requests.", "high"),
        ("[Request] Password reset assistance | User locked out after 3 incorrect attempts.", "medium"),
        ("[Incident] Printer on 3rd floor paper jam | Office printer is showing jam error.", "low"),
        ("[Request] Request for second monitor | New employee onboarding setup.", "low"),
    ]

    print("\n" + "=" * 70)
    print("RUNNING SAMPLE PREDICTIONS")
    print("=" * 70)
    
    for text, expected in samples:
        res = predict(text, model, tokenizer, device)
        status = "[CORRECT]" if res["predicted_priority"] == expected else "[MISMATCH]"
        print(f"\nInput: {text}")
        print(f"Predicted: {res['predicted_priority'].upper():<6} (Confidence: {res['confidence']*100:.1f}%) | Expected: {expected.upper()} {status}")
        print(f"Probabilities -> Low: {res['probabilities']['low']*100:.1f}%, Medium: {res['probabilities']['medium']*100:.1f}%, High: {res['probabilities']['high']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test Priority Prediction Model")
    parser.add_argument("--text", type=str, help="Full input text (e.g. '[Incident] subject | body')")
    parser.add_argument("--type", type=str, default="", help="Ticket type (Incident, Request, Problem, Change)")
    parser.add_argument("--subject", type=str, default="", help="Ticket subject")
    parser.add_argument("--body", type=str, default="", help="Ticket body description")
    parser.add_argument("--sample", action="store_true", help="Run a batch of sample test tickets")
    args = parser.parse_args()

    model, tokenizer, device = load_model(MODEL_PATH)
    print(f"Loaded model from '{MODEL_PATH}' on device: {device.upper()}")

    if args.sample:
        run_sample_tests(model, tokenizer, device)
        return

    if args.text:
        text = args.text
    elif args.subject or args.body or args.type:
        text = format_input(args.type, args.subject, args.body)
    else:
        # Interactive mode
        print("\n" + "=" * 50)
        print("INTERACTIVE TICKET PRIORITY PREDICTOR")
        print("Type 'exit' or 'quit' at any prompt to stop.")
        print("=" * 50 + "\n")
        
        while True:
            try:
                ticket_type = input("\nEnter Ticket Type [Incident/Request/Problem/Change] (optional): ").strip()
                if ticket_type.lower() in ["exit", "quit"]:
                    break
                
                subject = input("Enter Subject: ").strip()
                if subject.lower() in ["exit", "quit"]:
                    break
                
                body = input("Enter Body / Description: ").strip()
                if body.lower() in ["exit", "quit"]:
                    break
                
                if not subject and not body:
                    print("Please enter at least a subject or body.")
                    continue
                
                text = format_input(ticket_type, subject, body)
                res = predict(text, model, tokenizer, device)
                
                print("\n" + "-" * 40)
                print(f"Formatted Input: {text}")
                print(f"PREDICTED PRIORITY: {res['predicted_priority'].upper()} ({res['confidence']*100:.1f}% confidence)")
                print(f"Probabilities: Low={res['probabilities']['low']*100:.1f}%, Medium={res['probabilities']['medium']*100:.1f}%, High={res['probabilities']['high']*100:.1f}%")
                print("-" * 40)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
        return

    res = predict(text, model, tokenizer, device)
    print("\nResult:")
    print(f"Input: {res['text']}")
    print(f"Priority: {res['predicted_priority'].upper()} ({res['confidence']*100:.1f}% confidence)")
    print(f"Probabilities: {res['probabilities']}")


if __name__ == "__main__":
    main()
