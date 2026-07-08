# Coding vs Non-Coding DNA — Workshop Code

Bacterial DNA coding/non-coding classification: k-mer & CNN baselines → Evo2 embeddings →
knowledge distillation → efficiency analysis, plus an optional sparse-autoencoder
interpretability track. See `../facilitator_guide/day_by_day.md` for the schedule.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline order

1. **`src/build_dataset.py`** — turns `data/supervised/raw/{train,val,test}/*.fasta,*.gff`
   into labeled coding/non-coding window CSVs at `data/supervised/processed/{split}.csv`.
   Already run once; re-run if you change `--window`/`--stride`/class-balance settings:

   ```bash
   python src/build_dataset.py --raw_dir ../data/supervised/raw \
       --out_dir ../data/supervised/processed --window 200 --stride 200
   ```

2. **`src/extract_evo2_embeddings.py`** — organizer-run, before the school: calls the
   NVIDIA NIM Evo2 API on the labeled windows and writes embeddings + labels to
   `data/supervised/embeddings/{split}.npz`. Needs `NVIDIA_API_KEY` set. Students consume the
   output; they don't need API access during the week.

3. Notebooks in `notebooks/`, in order — see `facilitator_guide/day_by_day.md` for pacing:
   - `00_biology_intro_and_data_setup.ipynb`
   - `01_kmer_and_cnn_baselines.ipynb`
   - `02_evo2_embeddings_and_classifier.ipynb`
   - `03_knowledge_distillation.ipynb`
   - `04_compression_analysis_and_wrapup.ipynb`
   - `bonus_sparse_autoencoder_interpretability.ipynb` (optional/advanced)

## Layout

```
src/
  build_dataset.py           # raw FASTA/GFF -> labeled windows (stdlib only)
  extract_evo2_embeddings.py # labeled windows -> Evo2 NIM API -> embeddings+labels
  featurize.py                # k-mer frequency vectors, one-hot encoding
  data.py                      # load processed CSVs / embeddings
  embeddings.py                # load precomputed Evo2 embeddings (supervised + autoencoder)
  models/
    baselines.py               # k-mer+LR/RF, one-hot CNN
    classifier_heads.py        # MLP head on frozen Evo2 embeddings
    distillation.py            # teacher -> student, hybrid loss
    sae.py                     # minimal sparse autoencoder (bonus track)
  eval.py                       # accuracy/F1, param counts, latency benchmarking
  viz.py                         # PCA/UMAP, accuracy-vs-size-vs-speed, SAE feature plots
notebooks/                       # one per workshop day
```

## Data

`data/` lives at the repo root (not inside `code/`), one folder for everything:

```
data/
  supervised/
    raw/{train,val,test}/*.fasta,*.gff   # real bacterial genomes + CDS annotations
    processed/{train,val,test}.csv       # build_dataset.py output
    embeddings/{train,val,test}.npz      # extract_evo2_embeddings.py output
  autoencoder/
    {train,eval}/acts.dat + meta.json    # unlabeled Evo2 activations, SAE bonus track only
```

`data/` and `temp_chunk_of_code_will_be_deleted/` are gitignored — genomes and activation
dumps are multi-GB and don't belong in version control.
