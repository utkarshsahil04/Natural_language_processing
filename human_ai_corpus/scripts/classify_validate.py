"""
Phase 4 - Validation via Classification

Train Decision Tree and Boosting models to predict human vs AI
from the same linguistic features used in clustering.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parents[1]
MATRIX = BASE / "data" / "processed" / "feature_matrix.csv"
CLUSTERS = BASE / "data" / "processed" / "clusters.csv"
REPORTS = BASE / "reports"


def metrics_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1)),
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MATRIX)
    feature_cols = [c for c in df.columns if c not in {"response_id", "source_type"}]

    X = df[feature_cols].values
    y = (df["source_type"] == "AI").astype(int).values  # 1=AI, 0=human

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=4, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=100,
            learning_rate=0.6,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []
    importances = {}

    for name, model in models.items():
        # trees/boosting don't need scaling, but harmless
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        m = metrics_dict(y_test, pred)
        cv = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring="f1")
        m.update({"model": name, "cv_f1_mean": float(cv.mean()), "cv_f1_std": float(cv.std())})
        results.append(m)

        print(f"\n=== {name} ===")
        print(classification_report(y_test, pred, target_names=["human", "AI"]))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred))

        if hasattr(model, "feature_importances_"):
            importances[name] = dict(zip(feature_cols, model.feature_importances_.tolist()))

    res_df = pd.DataFrame(results)
    res_df.to_csv(REPORTS / "classification_metrics.csv", index=False)

    # Feature importance from Gradient Boosting (main analysis model)
    gb_imp = pd.Series(importances["GradientBoosting"]).sort_values(ascending=False)
    gb_imp.to_csv(REPORTS / "feature_importance.csv", header=["importance"])

    plt.figure(figsize=(8, 5))
    gb_imp.head(12).iloc[::-1].plot(kind="barh", color="steelblue")
    plt.title("Feature Importance (Gradient Boosting)")
    plt.tight_layout()
    plt.savefig(REPORTS / "feature_importance.png", dpi=150)
    plt.close()

    # Compare clusters vs true labels
    cluster_cmp = {"note": "no clusters file"}
    if CLUSTERS.exists():
        c = pd.read_csv(CLUSTERS)
        # majority mapping for kmeans
        tab = pd.crosstab(c["kmeans_cluster"], c["source_type"])
        tab.to_csv(REPORTS / "cluster_vs_true_crosstab.csv")
        # purity-like agreement after best matching
        from sklearn.metrics import adjusted_rand_score
        true = (c["source_type"] == "AI").astype(int)
        cluster_cmp = {
            "kmeans_ARI": float(adjusted_rand_score(true, c["kmeans_cluster"])),
            "hierarchical_ARI": float(adjusted_rand_score(true, c["hierarchical_cluster"])),
            "crosstab": tab.to_dict(),
        }

    summary = {
        "classification": results,
        "top_features_gradient_boosting": gb_imp.head(8).to_dict(),
        "cluster_alignment": cluster_cmp,
    }
    with open(REPORTS / "phase4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved metrics ->", REPORTS / "classification_metrics.csv")
    print("Saved importance ->", REPORTS / "feature_importance.csv")
    print(res_df[["model", "accuracy", "precision", "recall", "f1", "cv_f1_mean"]].round(3))


if __name__ == "__main__":
    main()
