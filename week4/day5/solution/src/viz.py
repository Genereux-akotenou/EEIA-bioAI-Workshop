"""
viz.py — visualisation de l'activité des caractéristiques du SAE le long de la séquence.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_sae_feature_activity(feature_matrix: np.ndarray, feature_indices,
                               title="SAE feature activity along the sequence"):
    """feature_matrix: (seq_len, n_selected_features) — une ligne par caractéristique,
    axe x = position du nucléotide. Sert à repérer des motifs d'activation périodiques
    (par ex. période 3, cadre de lecture des codons)."""
    fig, ax = plt.subplots(figsize=(12, 3 + 0.5 * len(feature_indices)))
    for i, feat_idx in enumerate(feature_indices):
        ax.plot(feature_matrix[:, i] + i * 1.2, label=f"feature {feat_idx}")
    ax.set_xlabel("Position in sequence (nt)")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
