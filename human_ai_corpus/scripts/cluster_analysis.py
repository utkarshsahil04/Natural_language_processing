"""
Phase 3 - Unsupervised Clustering

K-Means + Hierarchical clustering on linguistic features (labels unused).
Also elbow + silhouette analysis and PCA/t-SNE plots.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[1]
MATRIX = BASE / "data" / "processed" / "feature_matrix.csv"
OUT_CLUSTERS = BASE / "data" / "processed" / "clusters.csv"
REPORTS = BASE / "reports"


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"response_id", "source_type"}
    return [c for c in df.columns if c not in skip]


def elbow_silhouette(X: np.ndarray, k_range=range(2, 8)):
    inertias, sils = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, labels))
    return list(k_range), inertias, sils


def plot_elbow_sil(ks, inertias, sils):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ks, inertias, marker="o")
    ax[0].set_title("Elbow Method (K-Means)")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Inertia")

    ax[1].plot(ks, sils, marker="o", color="green")
    ax[1].set_title("Silhouette Score")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(REPORTS / "elbow_silhouette.png", dpi=150)
    plt.close(fig)


def plot_2d(X2, labels, title, path, hue_name="cluster"):
    plot_df = pd.DataFrame({"x": X2[:, 0], "y": X2[:, 1], hue_name: labels.astype(str)})
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df, x="x", y="y", hue=hue_name, palette="tab10", s=40)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MATRIX)
    cols = feature_columns(df)
    X = StandardScaler().fit_transform(df[cols].values)
    true = (df["source_type"] == "AI").astype(int).values

    ks, inertias, sils = elbow_silhouette(X)
    plot_elbow_sil(ks, inertias, sils)
    best_k = ks[int(np.argmax(sils))]
    print(f"Best k by silhouette: {best_k} (score={max(sils):.3f})")

    # For this project we also force k=2 to compare with human/AI split
    kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels = kmeans2.fit_predict(X)

    hier = AgglomerativeClustering(n_clusters=2, linkage="ward")
    hier_labels = hier.fit_predict(X)

    km_sil = silhouette_score(X, km_labels)
    hier_sil = silhouette_score(X, hier_labels)
    km_ari = adjusted_rand_score(true, km_labels)
    hier_ari = adjusted_rand_score(true, hier_labels)

    print(f"KMeans k=2     silhouette={km_sil:.3f}  ARI vs human/AI={km_ari:.3f}")
    print(f"Hierarchical   silhouette={hier_sil:.3f}  ARI vs human/AI={hier_ari:.3f}")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    plot_2d(X_pca, km_labels, "K-Means clusters (PCA)", REPORTS / "kmeans_pca.png")
    plot_2d(X_pca, hier_labels, "Hierarchical clusters (PCA)", REPORTS / "hierarchical_pca.png")
    plot_2d(X_pca, df["source_type"].values, "True labels human vs AI (PCA)",
            REPORTS / "true_labels_pca.png", hue_name="source")

    # t-SNE (can be slow on large data; fine for ~360)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X)
    plot_2d(X_tsne, km_labels, "K-Means clusters (t-SNE)", REPORTS / "kmeans_tsne.png")
    plot_2d(X_tsne, df["source_type"].values, "True labels human vs AI (t-SNE)",
            REPORTS / "true_labels_tsne.png", hue_name="source")

    out = df[["response_id", "source_type"]].copy()
    out["kmeans_cluster"] = km_labels
    out["hierarchical_cluster"] = hier_labels
    out.to_csv(OUT_CLUSTERS, index=False)

    summary = pd.DataFrame([
        {"method": "KMeans_k2", "silhouette": km_sil, "ARI_vs_true": km_ari},
        {"method": "Hierarchical_k2", "silhouette": hier_sil, "ARI_vs_true": hier_ari},
        {"method": "best_k_by_silhouette", "silhouette": max(sils), "ARI_vs_true": best_k},
    ])
    summary.to_csv(REPORTS / "clustering_summary.csv", index=False)
    print(f"Saved clusters -> {OUT_CLUSTERS}")
    print(f"Plots saved in {REPORTS}")


if __name__ == "__main__":
    main()
