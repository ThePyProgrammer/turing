"""Data preparation module for the {{PROJECT_NAME}} ML pipeline.

READ-ONLY — MEASUREMENT APPARATUS.

This file is part of the immutable evaluation infrastructure. The autoresearch
agent MUST NOT modify this file under any circumstances. Consistent data
preparation across experiments ensures that observed metric changes reflect
genuine model improvements, not data handling artifacts.

Provides:
  - load_config: Load YAML experiment configuration.
  - load_data: Load training data into a DataFrame.
  - create_splits: Stratified train/val/test split.
  - load_splits: Load pre-created split files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML experiment configuration.

    Args:
        path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    """Load training data into a DataFrame.

    Supports JSONL (.jsonl) and CSV (.csv) formats.

    Args:
        path: Path to the data file.

    Returns:
        DataFrame with training data.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If file format is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if p.suffix == ".jsonl":
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
    elif p.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format: {p.suffix}. Use .jsonl or .csv"
        )


def create_splits(
    data_path: str,
    output_dir: str,
    target_column: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, Path]:
    """Create stratified train/val/test splits from training data.

    Stratifies by target_column to preserve label distribution.

    Args:
        data_path: Path to the source data file.
        output_dir: Directory to write train.jsonl, val.jsonl, test.jsonl.
        target_column: Column to stratify on.
        test_size: Fraction of data for test set.
        val_size: Fraction of data for validation set.
        random_state: Random seed for reproducibility.

    Returns:
        Dict mapping split name to output file path.
    """
    from sklearn.model_selection import train_test_split

    df = load_data(data_path)
    if df.empty:
        raise ValueError(f"No data found in {data_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # First split: separate test set
    stratify_col = df[target_column] if target_column in df.columns else None
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Second split: separate val from train
    val_relative = val_size / (1.0 - test_size)
    stratify_col_tv = train_val[target_column] if target_column in train_val.columns else None
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        random_state=random_state,
        stratify=stratify_col_tv,
    )

    paths = {}
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for _, row in split_df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")
        paths[name] = path

    return paths


def load_splits(splits_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-created train/val/test splits.

    Args:
        splits_dir: Directory containing train.jsonl, val.jsonl, test.jsonl.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        FileNotFoundError: If any split file is missing.
    """
    splits_path = Path(splits_dir)
    train = load_data(str(splits_path / "train.jsonl"))
    val = load_data(str(splits_path / "val.jsonl"))
    test = load_data(str(splits_path / "test.jsonl"))
    return train, val, test


if __name__ == "__main__":
    config = load_config()
    data_cfg = config["data"]
    print(f"Creating splits from {data_cfg['source']}...")
    paths = create_splits(
        data_path=data_cfg["source"],
        output_dir=data_cfg["splits_dir"],
        target_column=data_cfg.get("target_column", "label"),
        test_size=data_cfg["split_ratios"]["test"],
        val_size=data_cfg["split_ratios"]["val"],
        random_state=data_cfg["random_state"],
    )
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("Done.")
