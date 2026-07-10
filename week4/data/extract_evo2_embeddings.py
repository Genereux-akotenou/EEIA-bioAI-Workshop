"""
extract_evo2_embeddings.py — ORGANIZER-run, before the school.

Calls the NVIDIA NIM Evo2 "forward" API on the labeled windows produced by
build_dataset.py, mean-pools the per-token activations of one layer into a
single embedding per window, and saves embeddings + labels for the
supervised teacher classifier (notebook 02) and later distillation
(notebook 03).

Students do NOT run this during the week — the API key/quota is yours, and
this is exactly the "pre-extract before the school" step that makes the
core pipeline independent of live API availability.

Request/response format matches the working call already used in
temp_chunk_of_code_will_be_deleted/05-Evo2-Activation-Extraction.ipynb:
    POST {FORWARD_URL} json={"sequence": seq, "output_layers": [layer]}
    -> {"data": base64(npz bytes)} -> npz[layer] with shape (seq_len, hidden_dim)
    (batch dim, if present, is index 0 and gets dropped)

Usage
-----
export NVIDIA_API_KEY=nvapi-...
python extract_evo2_embeddings.py \
    --processed_dir ../../data/supervised/processed \
    --out_dir ../../data/supervised/embeddings \
    --max_per_split 3000 \
    --target_layer blocks.26
"""

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FORWARD_URL_TMPL = "https://health.api.nvidia.com/v1/biology/arc/{model}/forward"
RATE_LIMIT_WAIT = 120
MAX_RETRIES = 10


def get_activations(sequence: str, layer: str, api_key: str, model: str) -> np.ndarray:
    """Call the NIM forward endpoint, return (seq_len, hidden_dim) activations
    for `layer`, retrying on rate limits / transient errors."""
    url = FORWARD_URL_TMPL.format(model=model)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        r = requests.post(
            url, headers=headers,
            json={"sequence": sequence, "output_layers": [layer]},
            timeout=600,
        )

        if r.status_code in (429, 503, 400, 504):
            wait = RATE_LIMIT_WAIT * (attempt + 1) if r.status_code != 504 else 30 * (attempt + 1)
            print(f"  [{r.status_code}] retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        r.raise_for_status()
        payload = r.json()
        decoded = base64.b64decode(payload["data"].encode("ascii"))
        npz = np.load(io.BytesIO(decoded))
        emb = npz[layer]
        if emb.ndim == 3:
            emb = emb[0]
        return emb

    raise RuntimeError(f"Exceeded {MAX_RETRIES} retries calling Evo2 NIM API")


def load_checkpoint(ckpt_path):
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return {"next_index": 0}


def save_checkpoint(ckpt_path, next_index):
    with open(ckpt_path, "w") as f:
        json.dump({"next_index": next_index}, f)


def extract_split(df: pd.DataFrame, out_dir: Path, split: str,
                   layer: str, api_key: str, model: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{split}.npz"
    ckpt_path = out_dir / f"{split}_checkpoint.json"
    partial_path = out_dir / f"{split}_partial.npy"

    n = len(df)
    hidden_dim = None
    ckpt = load_checkpoint(ckpt_path)
    start = ckpt["next_index"]

    if start == 0:
        embeddings = None  # allocated once we know hidden_dim
    else:
        embeddings = np.load(partial_path)
        hidden_dim = embeddings.shape[1]

    for i in range(start, n):
        seq = df.iloc[i]["sequence"]
        acts = get_activations(seq, layer, api_key, model)  # (seq_len, hidden_dim)
        pooled = acts.mean(axis=0)  # mean-pool over positions -> (hidden_dim,)

        if embeddings is None:
            hidden_dim = pooled.shape[0]
            embeddings = np.zeros((n, hidden_dim), dtype=np.float32)
        embeddings[i] = pooled

        if (i + 1) % 50 == 0 or i == n - 1:
            np.save(partial_path, embeddings)
            save_checkpoint(ckpt_path, i + 1)
            print(f"  [{split}] {i + 1}/{n} embeddings extracted", end="\r")

    print()
    np.savez(
        npz_path,
        embeddings=embeddings,
        labels=df["label"].to_numpy(),
        ids=df["id"].to_numpy(),
    )
    os.remove(partial_path)
    os.remove(ckpt_path)
    print(f"[{split}] saved {embeddings.shape} embeddings -> {npz_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed_dir", required=True,
                     help="Path to data/supervised/processed (build_dataset.py output)")
    ap.add_argument("--out_dir", required=True,
                     help="Path to write <split>.npz (embeddings + labels + ids)")
    ap.add_argument("--target_layer", default="blocks.26")
    ap.add_argument("--model", default="evo2-7b")
    ap.add_argument("--max_per_split", type=int, default=None,
                     help="Class-balanced subsample per split before extraction "
                          "(recommended: a few thousand — this is per-request API cost/time)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("Set NVIDIA_API_KEY in your environment first.")

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)

    for split in args.splits:
        csv_path = processed_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[{split}] {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path, dtype={"sequence": str})
        if args.max_per_split and len(df) > args.max_per_split:
            per_class = args.max_per_split // df["label"].nunique()
            df = (
                df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_class), random_state=42))
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )

        print(f"[{split}] extracting Evo2 embeddings for {len(df)} windows ...")
        extract_split(df, out_dir, split, args.target_layer, api_key, args.model)


if __name__ == "__main__":
    main()
