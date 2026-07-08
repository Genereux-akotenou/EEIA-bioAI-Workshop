"""
viz.py — the visual payoffs: embedding-space scatter (PCA/UMAP), the final
accuracy-vs-size-vs-speed comparison chart, and SAE feature/periodicity plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_embedding_space(embeddings, labels, method="pca", title="Embedding space",
                          ax=None):
    """2D scatter of embeddings colored by coding(1)/non-coding(0) label."""
    if method == "pca":
        coords = PCA(n_components=2).fit_transform(embeddings)
    elif method == "umap":
        import umap
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        raise ValueError("method must be 'pca' or 'umap'")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 5))

    for label, name, color in [(1, "coding", "#2196F3"), (0, "non-coding", "#E91E63")]:
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.5, label=name, color=color)
    ax.set_title(title)
    ax.legend()
    if own_fig:
        plt.tight_layout()
        plt.show()
    return coords


def plot_efficiency_tradeoff(results: dict):
    """results: {model_name: {"accuracy": float, "params": int, "latency_ms": float}}

    Bubble chart: x=latency, y=accuracy, bubble size=params. This is the
    Day-4 wrap-up payoff — every model built during the week on one chart.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    max_params = max(r["params"] for r in results.values())

    for name, r in results.items():
        size = 200 + 1800 * (r["params"] / max_params)
        ax.scatter(r["latency_ms"], r["accuracy"], s=size, alpha=0.6, label=name)
        ax.annotate(name, (r["latency_ms"], r["accuracy"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Inference latency (ms/sample)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Latency (bubble size = parameter count)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sae_feature_activity(feature_matrix: np.ndarray, feature_indices,
                               title="SAE feature activity along the sequence"):
    """feature_matrix: (seq_len, n_selected_features) — one line per feature,
    x-axis = nucleotide position. Used to spot periodic (e.g. period-3,
    codon-frame) activation patterns."""
    fig, ax = plt.subplots(figsize=(12, 3 + 0.5 * len(feature_indices)))
    for i, feat_idx in enumerate(feature_indices):
        ax.plot(feature_matrix[:, i] + i * 1.2, label=f"feature {feat_idx}")
    ax.set_xlabel("Position in sequence (nt)")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
