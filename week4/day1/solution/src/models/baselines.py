"""
baselines.py — modèles de référence du Jour 1 : k-mer + ML classique, et one-hot + CNN.
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def make_kmer_classifier(kind: str = "logreg"):
    """kind: "logreg" ou "rf" — entraîné directement avec .fit(X, y) de sklearn."""
    if kind == "logreg":
        return LogisticRegression(max_iter=1000)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    raise ValueError(f"kind inconnu : {kind}")


class OneHotCNN(nn.Module):
    """Petit CNN 1D sur des fenêtres one-hot (4, L) de nucléotides -> logit binaire."""

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
