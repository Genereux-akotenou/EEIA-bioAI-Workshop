"""
featurize.py — turn raw nucleotide strings into model-ready features.

Two encodings, matching the two Day-1 baselines:
    kmer_frequencies(seq, k)   -> classical ML input (k-mer count vector)
    one_hot(seq)               -> deep-learning input (CNN over A/C/G/T channels)
"""

import itertools
from functools import lru_cache

import numpy as np

BASES = "ACGT"


@lru_cache(maxsize=None)
def _kmer_vocab(k: int):
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]


def kmer_frequencies(seq: str, k: int = 4) -> np.ndarray:
    """Normalized k-mer frequency vector (fixed vocabulary of 4**k k-mers)."""
    vocab = _kmer_vocab(k)
    index = {kmer: i for i, kmer in enumerate(vocab)}
    counts = np.zeros(len(vocab), dtype=np.float32)
    n = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        idx = index.get(kmer)
        if idx is not None:  # skip k-mers containing N or other ambiguous bases
            counts[idx] += 1
            n += 1
    if n > 0:
        counts /= n
    return counts


def kmer_matrix(seqs, k: int = 4) -> np.ndarray:
    return np.stack([kmer_frequencies(s, k) for s in seqs])


def one_hot(seq: str, length: int = None) -> np.ndarray:
    """One-hot encode a sequence -> shape (4, length), channels = A,C,G,T.

    Ambiguous bases (N, etc.) map to an all-zero column. Sequences are
    truncated/right-padded (with zero columns) to `length` if given.
    """
    length = length or len(seq)
    arr = np.zeros((4, length), dtype=np.float32)
    base_to_idx = {b: i for i, b in enumerate(BASES)}
    for i, base in enumerate(seq[:length]):
        idx = base_to_idx.get(base)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


def one_hot_batch(seqs, length: int = None) -> np.ndarray:
    length = length or max(len(s) for s in seqs)
    return np.stack([one_hot(s, length) for s in seqs])
