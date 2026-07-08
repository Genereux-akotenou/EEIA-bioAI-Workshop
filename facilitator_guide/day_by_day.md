# Facilitator guide — day by day

Project: coding vs. non-coding DNA classification in bacterial genomes. Pipeline:
k-mer/CNN baselines → Evo2 embeddings + classifier (teacher) → knowledge distillation →
efficiency analysis, with an optional sparse-autoencoder interpretability track.

Audience: mostly first-year CS. Assume basic Python, no assumed ML background beyond
"a model takes numbers in and predicts a number out."

## Before Day 0 (organizer only)

- [ ] Confirm `data/supervised/raw/{train,val,test}` genomes are final.
- [ ] Run `src/build_dataset.py` to produce `data/supervised/processed/{train,val,test}.csv`.
- [ ] Set `NVIDIA_API_KEY` and run `src/extract_evo2_embeddings.py` with
      `--max_per_split` around 2000–4000 per split (balance API cost/time against having
      enough data for a good classifier — a few thousand windows is plenty here).
      **Do this well before Day 2**, rate limits mean this can take a while.
  - Watch for 429/503 retries in the output — the script backs off automatically, but budget
    real wall-clock time for it, not just "however long the model itself would take."
- [ ] Confirm `data/autoencoder/{train,eval}/acts.dat` finished downloading and
      `embeddings.load_autoencoder_activations(...)` returns the expected shape.
- [ ] Rotate/verify no API keys are hardcoded anywhere in what gets handed to students —
      `temp_chunk_of_code_will_be_deleted/` should not ship with the workshop repo.
- [ ] Dry-run notebooks 00, 01, 03, 04 top-to-bottom in a fresh environment for timing.
      Notebook 02 depends on the precomputed embeddings existing — verify separately.

## Day 1 — Biology + baselines

**Morning**: central dogma recap, what a CDS is, coding vs. non-coding, why this is a
real bioinformatics task (gene finding). Walk through raw FASTA/GFF files together
(`00_biology_intro_and_data_setup.ipynb`) before opening anything processed — the point is
that students see where labels come from, not that they parse GFF themselves.

**Afternoon**: `01_kmer_and_cnn_baselines.ipynb` — k-mer + logistic regression, then
one-hot + small CNN.

**Checkpoint**: every group has an accuracy/F1/params/latency row for both baselines.

**If a group is behind**: skip the CNN training loop entirely (paste in the pre-written
cell, don't debug from scratch) — the k-mer baseline alone is enough to keep pace with Day 2.

## Day 2 — Evo2 embeddings

**Content**: what a genomic foundation model is (conceptually — no local Evo2 inference),
why frozen embeddings + a small head is the standard efficient-transfer-learning move.

**Notebook**: `02_evo2_embeddings_and_classifier.ipynb` — load precomputed embeddings,
train the MLP head, PCA/UMAP visualization.

**Checkpoint**: Evo2-embedding classifier beats both Day-1 baselines, and the PCA plot shows
visible (if imperfect) separation between coding/non-coding clusters.

**If a group is behind**: the visualization cell is the one to cut, not the classifier —
keep the number, drop the plot if time is short.

## Day 3 — Knowledge distillation

**Content**: soft targets, temperature, why distillation transfers more than hard labels
("dark knowledge"). This is a good day to broaden the LLM connection — same idea used to
compress large language models.

**Notebook**: `03_knowledge_distillation.ipynb`.

**Checkpoint**: the distilled student (k-mer input, tiny MLP) recovers most of the
teacher's accuracy at a small fraction of the parameter count and latency.

**If a group is behind**: fix `alpha=1.0` (pure supervised, no distillation) to get a
baseline result quickly, then come back to soft targets if time allows — better to have
a number than a broken loss function at wrap-up time.

## Day 4 — Compression analysis + early bonus start

**Notebook**: `04_compression_analysis_and_wrapup.ipynb` — one chart with every model
built this week (accuracy vs. latency, bubble size = params).

Groups that reach this checkpoint with time to spare start the bonus track
(`bonus_sparse_autoencoder_interpretability.ipynb`) — this is explicitly optional, don't
let it become a blocker for groups still finishing Day 3.

## Day 5 — Bonus track + presentations

**Bonus/advanced groups**: vanilla autoencoder recap → sparse autoencoder → train on
cached Evo2 activations (`data/autoencoder/`, no labels) → look for periodic (period-3,
codon-frame) or otherwise interpretable features. Frame this explicitly as exploratory —
there's no guaranteed "correct" result, the payoff is seeing unsupervised structure emerge
at all.

**Everyone else**: polish the core pipeline, prepare the efficiency chart + narrative for
presentation.

**Afternoon**: group presentations — every group presents the core pipeline result
(baselines → teacher → student → efficiency chart); bonus-track groups additionally show
whatever they found in the SAE features.
