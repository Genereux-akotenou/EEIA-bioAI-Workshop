"""
baselines.py — Day-1 baselines: k-mer + classical ML, and one-hot + CNN.
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def make_kmer_classifier(kind: str = "logreg"):
    """kind: "logreg" or "rf" — trained directly with sklearn's .fit(X, y)."""
    if kind == "logreg":
        return LogisticRegression(max_iter=1000)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    raise ValueError(f"unknown kind: {kind}")


class OneHotCNN(nn.Module):
    """Small 1D CNN over one-hot (4, L) nucleotide windows -> binary logit."""

    def __init__(self, seq_len: int, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, 1)

    def forward(self, x):  # x: (B, 4, L)
        feats = self.net(x).squeeze(-1)  # (B, channels)
        return self.head(feats).squeeze(-1)  # (B,) logits
