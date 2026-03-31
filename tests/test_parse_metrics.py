"""Tests for the canonical metric parser (parse_metrics.py).

ADR-0015: Single parser replacing three independent implementations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.parse_metrics import (
    metrics_to_json,
    parse_metrics_block,
    parse_run_log,
)


# --- parse_metrics_block ---


def test_parse_basic():
    """Standard --- delimited block should parse correctly."""
    text = "---\naccuracy:       0.8500\nf1_weighted:    0.8200\n---\n"
    metrics, metadata = parse_metrics_block(text)
    assert metrics["accuracy"] == 0.85
    assert metrics["f1_weighted"] == 0.82
    assert metadata == {}


def test_parse_with_metadata():
    """Metadata keys should be separated from numeric metrics."""
    text = "---\naccuracy:  0.85\nmodel_type:  xgboost\ntrain_seconds:  2.5\n---\n"
    metrics, metadata = parse_metrics_block(text)
    assert metrics["accuracy"] == 0.85
    assert metadata["model_type"] == "xgboost"
    assert metadata["train_seconds"] == "2.5"
    assert "model_type" not in metrics
    assert "train_seconds" not in metrics


def test_parse_with_surrounding_text():
    """Text before and after delimiters should be ignored."""
    text = "Training...\nEpoch 1\n---\naccuracy:  0.90\n---\nDone.\nSaved."
    metrics, metadata = parse_metrics_block(text)
    assert metrics["accuracy"] == 0.90
    assert len(metrics) == 1


def test_parse_empty():
    """Empty string should return empty dicts."""
    metrics, metadata = parse_metrics_block("")
    assert metrics == {}
    assert metadata == {}


def test_parse_no_delimiters():
    """Text without --- should return empty dicts."""
    metrics, metadata = parse_metrics_block("accuracy: 0.85\nf1: 0.80")
    assert metrics == {}


def test_parse_colon_in_value():
    """Values containing colons should parse correctly (split on first colon)."""
    text = "---\naccuracy:  0.85\nnote:  train:val ratio is 7:3\n---\n"
    metrics, metadata = parse_metrics_block(text)
    assert metrics["accuracy"] == 0.85
    assert metadata["note"] == "train:val ratio is 7:3"


def test_parse_whitespace_handling():
    """Leading/trailing whitespace in keys and values should be stripped."""
    text = "---\n  accuracy  :   0.8500   \n---\n"
    metrics, metadata = parse_metrics_block(text)
    assert metrics["accuracy"] == 0.85


def test_parse_multiple_metrics():
    """All metrics between delimiters should be parsed."""
    text = "---\naccuracy:  0.85\nf1_weighted:  0.82\nprecision:  0.88\nrecall:  0.80\n---\n"
    metrics, metadata = parse_metrics_block(text)
    assert len(metrics) == 4


def test_parse_empty_value_skipped():
    """Lines with empty values should be skipped."""
    text = "---\naccuracy:  0.85\nbad_key:\n---\n"
    metrics, metadata = parse_metrics_block(text)
    assert "bad_key" not in metrics
    assert "bad_key" not in metadata


# --- parse_run_log ---


def test_parse_run_log_file(tmp_path: Path):
    """Should read and parse a real file."""
    log = tmp_path / "run.log"
    log.write_text("Training done.\n---\naccuracy:  0.87\nmodel_type:  xgboost\n---\n")
    metrics, metadata = parse_run_log(str(log))
    assert metrics["accuracy"] == 0.87
    assert metadata["model_type"] == "xgboost"


def test_parse_run_log_missing():
    """Missing file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_run_log("/nonexistent/run.log")


# --- metrics_to_json ---


def test_metrics_to_json():
    """Should produce valid JSON strings."""
    import json
    metrics = {"accuracy": 0.85, "f1": 0.82}
    metadata = {"model_type": "xgboost"}
    m_json, c_json = metrics_to_json(metrics, metadata)
    assert json.loads(m_json) == metrics
    assert json.loads(c_json) == metadata
