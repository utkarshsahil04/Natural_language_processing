"""
Fine-tune DistilBERT for 3-class message priority prediction (low/medium/high).

Optimized for a single consumer GPU (tested against RTX 4060, 8GB VRAM):
- fp16 mixed precision
- Batch size tuned to fit 8GB VRAM comfortably (raise/lower via --batch_size if needed)
- Class-weighted loss to handle the 'low' class being underrepresented
- Early stopping on validation F1

Run:
    python src/train.py
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

LABEL2ID = {"low": 0, "medium": 1, "high": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--output_dir", type=str, default="model/priority-distilbert")
    return p.parse_args()


def load_datasets(tokenizer):
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN
        )

    ds_train = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(
        tokenize, batched=True
    )
    ds_val = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False).map(
        tokenize, batched=True
    )
    ds_test = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False).map(
        tokenize, batched=True
    )

    cols = ["input_ids", "attention_mask", "label"]
    ds_train.set_format(type="torch", columns=cols)
    ds_val.set_format(type="torch", columns=cols)
    ds_test.set_format(type="torch", columns=cols)

    return ds_train, ds_val, ds_test, train_df


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy to counter class imbalance."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels") if "labels" in inputs else inputs.pop("label")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds_train, ds_val, ds_test, train_df = load_datasets(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Class weights to counter 'low' class underrepresentation
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["label"].values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Class weights (low, medium, high): {class_weights.tolist()}")

    training_args = TrainingArguments(
        output_dir="model/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=100,
        fp16=(device == "cuda"),           # mixed precision on GPU
        dataloader_num_workers=0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    print("\n=== Validation results ===")
    print(trainer.evaluate())

    print("\n=== Test set results ===")
    test_results = trainer.predict(ds_test)
    preds = np.argmax(test_results.predictions, axis=1)
    labels = test_results.label_ids

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    print(f"Test accuracy: {acc:.4f}")
    for i, name in ID2LABEL.items():
        print(f"  {name:8s}  precision={precision[i]:.3f}  recall={recall[i]:.3f}  f1={f1[i]:.3f}")

    print("\nConfusion matrix (rows=true, cols=pred), order [low, medium, high]:")
    print(confusion_matrix(labels, preds))

    # Save final model + tokenizer for deployment
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


if __name__ == "__main__":
    main()
