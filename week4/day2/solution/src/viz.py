"""
viz.py — visualisation de l'espace des embeddings (PCA/UMAP).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_embedding_space(embeddings, labels, method="pca", title="Embedding space",
                          ax=None):
    """Nuage de points 2D des embeddings, colorés par étiquette codant(1)/non-codant(0)."""
    if method == "pca":
        coords = PCA(n_components=2).fit_transform(embeddings)
    elif method == "umap":
        import umap
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        raise ValueError("method doit être 'pca' ou 'umap'")

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
