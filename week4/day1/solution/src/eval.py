"""
eval.py — les métriques payoff du Jour 4 : accuracy/F1, nombre de paramètres, latence
d'inférence. Fonctionne pour les modules torch et les estimateurs sklearn.
"""

import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_logits(logits, labels, threshold=0.0):
    """logits: scores bruts (seuil sigmoïde 0.5 == seuil de logit 0.0)."""
    preds = (logits > threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def evaluate_sklearn(model, X, y):
    preds = model.predict(X)
    return {"accuracy": accuracy_score(y, preds), "f1": f1_score(y, preds)}


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_latency_torch(model, example_input, n_runs=100, device="cpu"):
    """Latence moyenne d'inférence par échantillon, en millisecondes."""
    model = model.to(device).eval()
    example_input = example_input.to(device)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(example_input)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(example_input)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0


def measure_latency_sklearn(model, X, n_runs=100):
    for _ in range(5):
        model.predict(X)
    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(X)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0
