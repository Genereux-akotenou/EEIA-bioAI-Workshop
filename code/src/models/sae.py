"""
sae.py — minimal Top-K sparse autoencoder for the bonus interpretability
track. Trained with zero labels on raw Evo2 activations (data/autoencoder/).

Deliberately tiny: this is a teaching artifact, not the research-scale
version in temp_chunk_of_code_will_be_deleted/. If you outgrow this, that's
where the DDP/checkpointed version lives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSparseAutoencoder(nn.Module):
    """Overcomplete dictionary (d_hidden > d_in) with a hard top-k sparsity
    constraint: each activation vector is explained by exactly k "features"."""

    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        self.k = k
        W = F.normalize(0.1 * torch.randn(d_in, d_hidden), dim=0)
        self.W = nn.Parameter(W)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        f = F.relu(x @ self.W + self.b_enc)
        values, indices = torch.topk(f, self.k, dim=-1)
        sparse_f = torch.zeros_like(f)
        return sparse_f.scatter_(-1, indices, values)

    def decode(self, f):
        return f @ self.W.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


def train_sae(sae, activations: torch.Tensor, epochs=10, lr=2e-4,
              batch_size=512, device="cpu"):
    """activations: (N, d_in) tensor, already loaded/subsampled into memory —
    this is meant for a few thousand to tens of thousands of vectors, not
    the full multi-GB dump."""
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    n = activations.shape[0]
    history = {"recon_loss": []}

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            x = activations[perm[start:start + batch_size]].to(device)
            optimizer.zero_grad()
            x_hat, _ = sae(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.shape[0]
        history["recon_loss"].append(epoch_loss / n)
        print(f"epoch {epoch + 1}/{epochs}  recon_loss={history['recon_loss'][-1]:.4f}")

    return sae, history
