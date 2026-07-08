"""
embeddings.py — loaders for the two separate Evo2 artifacts used in this project.

    load_supervised_embeddings(...)  -> labeled, per-window (extract_evo2_embeddings.py
                                         output) — used by the teacher classifier /
                                         distillation modules.
    load_autoencoder_activations(...) -> unlabeled, per-token (offline extraction,
                                          acts.dat + meta.json) — SAE bonus track only.
                                          NOT usable for supervised classification: these
                                          activations were never traced back to labeled
                                          coding/non-coding windows.
"""

from pathlib import Path

import numpy as np


def load_supervised_embeddings(embeddings_dir, split: str):
    """Returns (embeddings: (N, d), labels: (N,), ids: (N,))."""
    path = Path(embeddings_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run extract_evo2_embeddings.py first."
        )
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["labels"], data["ids"]


def load_autoencoder_activations(autoencoder_dir, split: str, dtype=np.float32):
    """Read-only memmap over the offline unlabeled Evo2 activation dump.

    split: "train" or "eval" (this dataset uses "eval", not "val").
    Returns a (total_tokens, d_model) memmap — slice it, don't load it whole.
    """
    split_dir = Path(autoencoder_dir) / split
    meta_path, data_path = split_dir / "meta.json", split_dir / "acts.dat"
    if not meta_path.exists() or not data_path.exists():
        raise FileNotFoundError(f"missing acts.dat/meta.json in {split_dir}")

    import json
    meta = json.loads(meta_path.read_text())
    return np.memmap(
        data_path, mode="r", dtype=dtype,
        shape=(meta["total_tokens"], meta["d_model"]),
    )
