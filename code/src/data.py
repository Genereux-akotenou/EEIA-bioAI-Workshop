"""
data.py — load the labeled window CSVs produced by build_dataset.py.
"""

from pathlib import Path

import pandas as pd


def load_split(processed_dir, split: str) -> pd.DataFrame:
    """Load one of train/val/test as a DataFrame (columns: see build_dataset.py)."""
    path = Path(processed_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run build_dataset.py first to generate it."
        )
    return pd.read_csv(path, dtype={"sequence": str})


def load_all(processed_dir):
    """Return {"train": df, "val": df, "test": df}."""
    return {split: load_split(processed_dir, split) for split in ("train", "val", "test")}


def subsample(df: pd.DataFrame, n: int, seed: int = 42, stratify_col: str = "label") -> pd.DataFrame:
    """Class-balanced subsample of at most n rows total (n // n_classes per class)."""
    if len(df) <= n:
        return df
    per_class = n // df[stratify_col].nunique()
    return (
        df.groupby(stratify_col, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed))
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
