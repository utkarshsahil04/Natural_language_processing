"""
Phase 5 - Build analysis report markdown from pipeline outputs.
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
FEATURES = BASE / "data" / "processed" / "feature_matrix.csv"
OUT = REPORTS / "project_report.md"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(FEATURES)
    means = feat.groupby("source_type").mean(numeric_only=True).round(3)

    metrics = pd.read_csv(REPORTS / "classification_metrics.csv") if (REPORTS / "classification_metrics.csv").exists() else None
    imp = pd.read_csv(REPORTS / "feature_importance.csv") if (REPORTS / "feature_importance.csv").exists() else None
    clus = pd.read_csv(REPORTS / "clustering_summary.csv") if (REPORTS / "clustering_summary.csv").exists() else None

    summary = {}
    if (REPORTS / "phase4_summary.json").exists():
        summary = json.loads((REPORTS / "phase4_summary.json").read_text(encoding="utf-8"))

    lines = []
    lines.append("# Corpus of Human-AI Conversation with Synthetic Annotation")
    lines.append("")
    lines.append("## 1. Problem Statement")
    lines.append("")
    lines.append(
        "Create a corpus of human-AI interactions and analyze differences in sentence "
        "structure between human-generated responses and AI-generated responses."
    )
    lines.append("")
    lines.append("This project uses **classic ML only** (clustering, decision trees, boosting).")
    lines.append("")
    lines.append("## 2. Corpus")
    lines.append("")
    lines.append(f"- Total responses: **{len(feat)}**")
    lines.append(f"- Human: **{(feat['source_type']=='human').sum()}**")
    lines.append(f"- AI: **{(feat['source_type']=='AI').sum()}**")
    lines.append("- Prompt categories: advice, factual, creative, opinion")
    lines.append("- Format: `response_id, prompt_id, prompt, response_text, source_type`")
    lines.append("")
    lines.append("## 3. Method")
    lines.append("")
    lines.append("### Phase 1 - Collection")
    lines.append("Paired prompts answered by human-style and AI-style responses.")
    lines.append("")
    lines.append("### Phase 2 - Linguistic features")
    lines.append("Extracted with spaCy:")
    lines.append("")
    lines.append("- avg sentence length, parse tree depth, type-token ratio")
    lines.append("- passive ratio, hedging rate, discourse marker rate")
    lines.append("- POS percentages (noun/verb/adj/adv/pron)")
    lines.append("- word length and punctuation ratio")
    lines.append("")
    lines.append("### Phase 3 - Clustering")
    lines.append("K-Means and Hierarchical clustering on standardized features (labels hidden).")
    lines.append("Optimal k checked via elbow + silhouette. PCA/t-SNE used for visualization.")
    lines.append("")
    lines.append("### Phase 4 - Classification validation")
    lines.append("Decision Tree, AdaBoost, and Gradient Boosting predict human vs AI.")
    lines.append("")
    lines.append("## 4. Results")
    lines.append("")
    lines.append("### Feature means by source")
    lines.append("")
    lines.append("```")
    lines.append(means.T.to_string())
    lines.append("```")
    lines.append("")

    if clus is not None:
        lines.append("### Clustering summary")
        lines.append("")
        lines.append("```")
        lines.append(clus.to_string(index=False))
        lines.append("```")
        lines.append("")
        lines.append("Figures: `elbow_silhouette.png`, `kmeans_pca.png`, `kmeans_tsne.png`, `true_labels_pca.png`")
        lines.append("")

    if metrics is not None:
        lines.append("### Classification metrics")
        lines.append("")
        lines.append("```")
        lines.append(metrics[["model", "accuracy", "precision", "recall", "f1", "cv_f1_mean"]].round(3).to_string(index=False))
        lines.append("```")
        lines.append("")

    if imp is not None:
        # file may be series-like with first col as index name
        cols = list(imp.columns)
        if len(cols) >= 2:
            top = imp.sort_values(cols[1], ascending=False).head(8)
            lines.append("### Top distinguishing features (Gradient Boosting)")
            lines.append("")
            lines.append("```")
            lines.append(top.to_string(index=False))
            lines.append("```")
            lines.append("")

    if summary.get("cluster_alignment"):
        lines.append("### Did clustering align with human/AI?")
        lines.append("")
        ca = summary["cluster_alignment"]
        if isinstance(ca, dict) and "kmeans_ARI" in ca:
            lines.append(f"- K-Means ARI vs true labels: **{ca['kmeans_ARI']:.3f}**")
            lines.append(f"- Hierarchical ARI vs true labels: **{ca['hierarchical_ARI']:.3f}**")
            lines.append("")
            lines.append(
                "ARI close to 1 means clusters recovered the human/AI split well. "
                "Lower ARI means linguistic styles overlap more than expected."
            )
            lines.append("")

    lines.append("## 5. Linguistic interpretation")
    lines.append("")
    lines.append("- **Syntactic layer:** parse-tree depth and sentence length often differ; AI replies tend to be more uniformly structured.")
    lines.append("- **Lexical/morphological layer:** type-token ratio and average word length capture vocabulary style differences.")
    lines.append("- **Discourse/pragmatic layer:** discourse markers and hedging reflect politeness/certainty patterns.")
    lines.append("- **POS distribution:** noun/verb/adj balance indicates informational vs conversational style.")
    lines.append("")
    lines.append("## 6. Conclusion")
    lines.append("")
    lines.append(
        "Human and AI responses can be compared using transparent linguistic features and classic ML. "
        "Clustering checks whether structure alone separates sources; supervised models quantify separability "
        "and highlight which features matter most."
    )
    lines.append("")
    lines.append("## 7. How to reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python run_pipeline.py")
    lines.append("```")
    lines.append("")
    lines.append("## Suggested submission structure")
    lines.append("")
    lines.append("1. Introduction & problem statement")
    lines.append("2. Related linguistic background")
    lines.append("3. Corpus design and collection")
    lines.append("4. Feature engineering")
    lines.append("5. Clustering experiments")
    lines.append("6. Classification validation")
    lines.append("7. Discussion of human vs AI differences")
    lines.append("8. Limitations & future work")
    lines.append("9. Conclusion")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written -> {OUT}")


if __name__ == "__main__":
    main()
