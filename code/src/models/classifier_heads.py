"""
classifier_heads.py — lightweight heads trained on top of frozen embeddings
(Evo2 embeddings for the teacher, or SAE latents for the bonus track probe).
"""

import torch.nn as nn


class MLPHead(nn.Module):
    """Single hidden-layer MLP -> binary logit. Kept small on purpose: the
    point of this module is that the embeddings already did the hard work."""

    def __init__(self, d_in: int, d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):  # x: (B, d_in)
        return self.net(x).squeeze(-1)  # (B,) logits
