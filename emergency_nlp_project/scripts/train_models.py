"""
Phase 4 - Model Training
Trains 3 classical ML classifiers:
  1. Decision Tree    (baseline, max_depth=12, interpretable)
  2. Random Forest    (bagging, 200 trees)
  3. AdaBoost         (boosting, 150 estimators)

All use class_weight='balanced' or sample_weight to handle O >> LOC/ACT imbalance.
Saves confusion matrices and feature importance plots to reports/
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

def load_data():
    X = pd.read_csv(os.path.join(PROC_DIR, "features.csv"))
    y = pd.read_csv(os.path.join(PROC_DIR, "labels.csv"))["bio_enc"].values
    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
        enc = pickle.load(f)
    return X.values, y, list(X.columns), enc

def plot_confusion_matrix(y_te, y_pred, labels, name):
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
                cmap="YlOrRd", linewidths=0.5)
    plt.title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    fname = os.path.join(REPORT_DIR, f"cm_{name.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=150); plt.close()
    return fname

def plot_feature_importance(clf, feat_names, name):
    if not hasattr(clf, "feature_importances_"):
        return None
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1]
    plt.figure(figsize=(11,5))
    bars = plt.bar(range(len(idx)), imp[idx], color="steelblue", edgecolor="white")
    plt.xticks(range(len(idx)), [feat_names[i] for i in idx], rotation=45, ha="right", fontsize=9)
    plt.title(f"Feature Importances — {name}", fontsize=13, fontweight="bold")
    plt.ylabel("Importance")
    plt.tight_layout()
    fname = os.path.join(REPORT_DIR, f"fi_{name.lower().replace(' ','_')}.png")
    plt.savefig(fname, dpi=150); plt.close()
    return fname

def train_model(name, clf, X_tr, X_te, y_tr, y_te, bio_le, feat_names, sw=None):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    fit_kwargs = {}
    if sw is not None:
        fit_kwargs["sample_weight"] = sw

    clf.fit(X_tr, y_tr, **fit_kwargs)
    y_pred = clf.predict(X_te)

    target_names = bio_le.classes_
    report_str = classification_report(y_te, y_pred, target_names=target_names, zero_division=0)
    print(report_str)

    macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"  Macro F1: {macro_f1:.4f}")

    cm_file = plot_confusion_matrix(y_te, y_pred, target_names, name)
    fi_file = plot_feature_importance(clf, feat_names, name)
    print(f"  Confusion matrix: {cm_file}")
    if fi_file:
        print(f"  Feature importance: {fi_file}")

    model_file = os.path.join(MODEL_DIR, f"{name.lower().replace(' ','_')}.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(clf, f)
    print(f"  Model saved: {model_file}")

    return macro_f1

def main():
    print("Loading features ...")
    X, y, feat_names, enc = load_data()
    bio_le = enc["bio_le"]
    print(f"  Shape: {X.shape}, Classes: {bio_le.classes_}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    sw = compute_sample_weight("balanced", y_tr)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, class_weight="balanced"),
            n_estimators=200, learning_rate=0.5, random_state=42
        ),
    }

    results = {}
    for name, clf in models.items():
        use_sw = sw if name != "AdaBoost" else None
        f1 = train_model(name, clf, X_tr, X_te, y_tr, y_te, bio_le, feat_names, sw=use_sw)
        results[name] = f1

    # ── Model comparison bar chart ─────────────────────────────────────────
    print("\n=== Cross-Validation Comparison (5-fold Macro F1) ===")
    cv_means, cv_stds = [], []
    for name, clf in models.items():
        scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                 scoring="f1_macro", n_jobs=-1)
        cv_means.append(scores.mean())
        cv_stds.append(scores.std())
        print(f"  {name:18s}: {scores.mean():.3f} ± {scores.std():.3f}")

    fig, ax = plt.subplots(figsize=(8,5))
    colors  = ["#2196F3","#4CAF50","#FF9800"]
    names   = list(models.keys())
    bars    = ax.bar(names, cv_means, yerr=[s*2 for s in cv_stds],
                     color=colors, capsize=8, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, cv_means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Model Comparison — 5-Fold Cross-Validation", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    print(f"\nAll models and plots saved.")
    print(f"Best model: {max(results, key=results.get)} (F1={max(results.values()):.4f})")

if __name__ == "__main__":
    main()
