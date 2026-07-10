"""
embeddings.py — chargeur pour les embeddings Evo2 supervisés (extraits une fois par
l'organisateur avec extract_evo2_embeddings.py).

Ce fichier est complet (code d'infrastructure/E-S, pas l'objectif pédagogique du jour) —
utilisez-le tel quel.
"""

from pathlib import Path

import numpy as np


def load_supervised_embeddings(embeddings_dir, split: str):
    """Retourne (embeddings: (N, d), labels: (N,), ids: (N,))."""
    path = Path(embeddings_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable — lancez extract_evo2_embeddings.py d'abord."
        )
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["labels"], data["ids"]
