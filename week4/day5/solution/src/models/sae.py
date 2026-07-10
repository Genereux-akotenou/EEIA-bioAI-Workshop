"""
sae.py — autoencodeur parcimonieux Top-K minimal pour la piste bonus d'interprétabilité.
Entraîné sans aucune étiquette sur des activations Evo2 brutes (data/autoencoder/).

Volontairement minuscule : c'est un artefact pédagogique, pas la version à l'échelle
recherche.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSparseAutoencoder(nn.Module):
    """Dictionnaire surcomplet (d_hidden > d_in) avec une contrainte stricte de parcimonie
    top-k : chaque vecteur d'activation est expliqué par exactement k « caractéristiques »."""

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
    """activations: tenseur (N, d_in), déjà chargé/sous-échantillonné en mémoire — prévu
    pour quelques milliers à dizaines de milliers de vecteurs, pas le dump complet
    multi-Go."""
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
