"""Tests for the data preparation module (prepare.py).

Blocker #5: The MEASUREMENT APPARATUS foundation (ADR-0002).
Every experiment comparison depends on prepare.py producing correct,
reproducible, stratified data splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from prepare import create_splits, load_config, load_data, load_splits


# --- load_data ---


def test_load_data_csv(tmp_path: Path):
    """CSV files should load into a DataFrame."""
    csv = tmp_path / "data.csv"
    csv.write_text("feature_1,label\n1.0,positive\n2.0,negative\n3.0,positive\n")
    df = load_data(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "feature_1" in df.columns
    assert "label" in df.columns


def test_load_data_jsonl(tmp_path: Path):
    """JSONL files should load into a DataFrame."""
    jsonl = tmp_path / "data.jsonl"
    records = [
        {"feature_1": 1.0, "label": "positive"},
        {"feature_1": 2.0, "label": "negative"},
    ]
    with open(jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    df = load_data(str(jsonl))
    assert len(df) == 2
    assert df.iloc[0]["feature_1"] == 1.0


def test_load_data_unsupported_format(tmp_path: Path):
    """Unsupported file formats should raise ValueError."""
    parquet = tmp_path / "data.parquet"
    parquet.write_text("not actually parquet")
    with pytest.raises(ValueError, match="Unsupported"):
        load_data(str(parquet))


def test_load_data_missing_file():
    """Missing files should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data("/nonexistent/data.csv")


def test_load_data_empty_jsonl(tmp_path: Path):
    """Empty JSONL should return empty DataFrame."""
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")
    df = load_data(str(jsonl))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# --- create_splits ---


def _make_csv(tmp_path: Path, n: int = 40) -> Path:
    """Create a sample CSV with balanced classes."""
    csv = tmp_path / "data.csv"
    lines = ["feature_1,feature_2,label"]
    for i in range(n):
        label = "positive" if i % 2 == 0 else "negative"
        lines.append(f"{i},{i*0.1},{label}")
    csv.write_text("\n".join(lines))
    return csv


def test_create_splits_produces_three_files(tmp_path: Path):
    """Should create train.jsonl, val.jsonl, test.jsonl."""
    csv = _make_csv(tmp_path)
    output_dir = str(tmp_path / "splits")
    paths = create_splits(str(csv), output_dir, target_column="label")
    assert "train" in paths
    assert "val" in paths
    assert "test" in paths
    assert Path(paths["train"]).exists()
    assert Path(paths["val"]).exists()
    assert Path(paths["test"]).exists()


def test_create_splits_no_data_overlap(tmp_path: Path):
    """Train, val, and test sets must have zero row overlap."""
    csv = _make_csv(tmp_path, n=60)
    output_dir = str(tmp_path / "splits")
    create_splits(str(csv), output_dir, target_column="label")

    train = load_data(str(tmp_path / "splits" / "train.jsonl"))
    val = load_data(str(tmp_path / "splits" / "val.jsonl"))
    test = load_data(str(tmp_path / "splits" / "test.jsonl"))

    train_idx = set(train["feature_1"].tolist())
    val_idx = set(val["feature_1"].tolist())
    test_idx = set(test["feature_1"].tolist())

    assert train_idx & val_idx == set(), "Train/val overlap"
    assert train_idx & test_idx == set(), "Train/test overlap"
    assert val_idx & test_idx == set(), "Val/test overlap"


def test_create_splits_deterministic(tmp_path: Path):
    """Same seed should produce identical splits."""
    csv = _make_csv(tmp_path, n=40)

    out1 = str(tmp_path / "splits1")
    out2 = str(tmp_path / "splits2")
    create_splits(str(csv), out1, target_column="label", random_state=42)
    create_splits(str(csv), out2, target_column="label", random_state=42)

    train1 = load_data(str(Path(out1) / "train.jsonl"))
    train2 = load_data(str(Path(out2) / "train.jsonl"))

    assert train1["feature_1"].tolist() == train2["feature_1"].tolist()


def test_create_splits_preserves_total(tmp_path: Path):
    """Total rows across splits must equal original data."""
    n = 50
    csv = _make_csv(tmp_path, n=n)
    output_dir = str(tmp_path / "splits")
    create_splits(str(csv), output_dir, target_column="label")

    train = load_data(str(tmp_path / "splits" / "train.jsonl"))
    val = load_data(str(tmp_path / "splits" / "val.jsonl"))
    test = load_data(str(tmp_path / "splits" / "test.jsonl"))

    assert len(train) + len(val) + len(test) == n


def test_create_splits_empty_data_raises(tmp_path: Path):
    """Empty data file should raise ValueError."""
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("")
    output_dir = str(tmp_path / "splits")
    with pytest.raises(ValueError, match="No data"):
        create_splits(str(empty_jsonl), output_dir)


# --- load_splits ---


def test_load_splits_returns_three_dataframes(tmp_path: Path):
    """Should return (train, val, test) tuple of DataFrames."""
    csv = _make_csv(tmp_path, n=40)
    output_dir = str(tmp_path / "splits")
    create_splits(str(csv), output_dir, target_column="label")

    train, val, test = load_splits(output_dir)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) > 0


def test_load_splits_missing_file(tmp_path: Path):
    """Missing split file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_splits(str(tmp_path / "nonexistent"))


# --- load_config ---


def test_load_config_valid(tmp_path: Path):
    """Should load and return a dict from YAML."""
    import yaml
    config = {"data": {"source": "data.csv"}, "model": {"type": "xgboost"}}
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    result = load_config(str(config_path))
    assert result["model"]["type"] == "xgboost"
