"""
eval.py — the Day-4 payoff metrics: accuracy/F1, parameter counts, inference
latency. Works for both torch modules and sklearn estimators.
"""

import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def evaluate_logits(logits, labels, threshold=0.0):
    """logits: raw scores (sigmoid threshold 0.5 == logit threshold 0.0)."""
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
    """Mean per-sample inference latency in milliseconds."""
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
