"""
viz.py — le graphique payoff du Jour 4 : précision vs. latence vs. taille.
"""

import matplotlib.pyplot as plt


def plot_efficiency_tradeoff(results: dict):
    """results: {nom_modèle: {"accuracy": float, "params": int, "latency_ms": float}}

    Graphique à bulles : x=latence, y=précision, taille de bulle=nombre de paramètres.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    max_params = max(r["params"] for r in results.values())

    for name, r in results.items():
        size = 200 + 1800 * (r["params"] / max_params)
        ax.scatter(r["latency_ms"], r["accuracy"], s=size, alpha=0.6, label=name)
        ax.annotate(name, (r["latency_ms"], r["accuracy"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Inference latency (ms/sample)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Latency (bubble size = parameter count)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
