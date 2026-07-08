# Grading / "done" checkpoints

Use these as pass/fail checkpoints per day, not a fine-grained rubric — the goal is making
sure no group is silently stuck, not scoring precision.

## Day 1 — Baselines

- [ ] Can explain, in their own words, what a CDS is and what the label means
- [ ] `train.csv`/`val.csv`/`test.csv` loaded, class balance inspected
- [ ] k-mer + logistic regression trained, accuracy/F1 computed on val
- [ ] one-hot + CNN trained, accuracy/F1 computed on val
- **Minimum bar**: at least the k-mer baseline works end-to-end with a real (not
  placeholder) number.

## Day 2 — Evo2 embeddings

- [ ] Precomputed embeddings loaded, shapes understood (N windows, d_model dims)
- [ ] MLP head trained on frozen embeddings
- [ ] Result compared numerically against Day 1 baselines (embeddings should win)
- [ ] PCA (or UMAP) plot produced, can describe what it shows
- **Minimum bar**: a trained classifier with a real accuracy number beating Day 1.

## Day 3 — Distillation

- [ ] Can explain what "soft targets" and "temperature" mean, in their own words
- [ ] Student trained with the hybrid loss (not just plain supervised — alpha < 1 tried
      at least once)
- [ ] Student vs. teacher accuracy, params, and latency all recorded
- **Minimum bar**: a distilled student that's meaningfully smaller/faster than the
  teacher, even if it loses several points of accuracy.

## Day 4 — Compression analysis

- [ ] `results` dict filled with real numbers from every model built so far
- [ ] Efficiency chart produced (`plot_efficiency_tradeoff`)
- [ ] Can articulate the tradeoff: which model would they actually ship, and why
- **Minimum bar**: the chart exists with at least baselines + teacher + student plotted.

## Day 5 — Bonus track (optional, ungraded pass/fail is fine)

- [ ] Vanilla autoencoder trained, bottleneck concept explained
- [ ] Sparse autoencoder trained on cached Evo2 activations
- [ ] At least attempted the periodicity/interpretability analysis, regardless of outcome
- **Note**: this track has no guaranteed "correct" result. Credit exploration and correct
  methodology, not a specific finding.

## Final presentation

Every group should be able to show, in under 5 minutes:
1. The task and why it matters (1 sentence)
2. The efficiency chart from Day 4
3. One sentence on what they'd actually deploy and why
4. (Bonus groups) what they found in the SAE features, if anything
