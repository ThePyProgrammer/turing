"""Tests for the measurement apparatus (evaluate.py).

Tier 1: These tests verify the core invariant (ADR-0002).
The evaluation harness must produce correct, consistent metrics
across all edge cases — it is the yardstick for all experiments.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import evaluate_model, format_metrics


# --- evaluate_model ---


def test_evaluate_accuracy_basic():
    """Basic accuracy computation with known values."""
    preds = np.array([0, 1, 1, 0, 1])
    truth = np.array([0, 1, 1, 0, 0])
    config = {"evaluation": {"metrics": ["accuracy"]}}
    result = evaluate_model(preds, truth, config)
    assert result["accuracy"] == 0.8


def test_evaluate_perfect_predictions():
    """Perfect predictions should yield 1.0 for accuracy and F1."""
    preds = np.array([0, 1, 0, 1, 0])
    truth = np.array([0, 1, 0, 1, 0])
    config = {"evaluation": {"metrics": ["accuracy", "f1_weighted"]}}
    result = evaluate_model(preds, truth, config)
    assert result["accuracy"] == 1.0
    assert result["f1_weighted"] == 1.0


def test_evaluate_all_wrong():
    """Completely wrong predictions should yield 0.0 accuracy."""
    preds = np.array([1, 0, 1, 0])
    truth = np.array([0, 1, 0, 1])
    config = {"evaluation": {"metrics": ["accuracy"]}}
    result = evaluate_model(preds, truth, config)
    assert result["accuracy"] == 0.0


def test_evaluate_length_mismatch_raises():
    """Mismatched array lengths must raise ValueError."""
    preds = np.array([0, 1, 1])
    truth = np.array([0, 1])
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_model(preds, truth)


def test_evaluate_empty_arrays():
    """Empty arrays should not crash — edge case for first experiment."""
    preds = np.array([])
    truth = np.array([])
    # sklearn metrics may raise or return 0.0 — either is acceptable
    # but it must not crash with an unhandled exception
    try:
        result = evaluate_model(preds, truth, {"evaluation": {"metrics": ["accuracy"]}})
        assert isinstance(result, dict)
    except ValueError:
        pass  # sklearn raising ValueError on empty input is acceptable


def test_evaluate_unknown_metric_skipped():
    """Unknown metric names should be silently skipped, not crash."""
    preds = np.array([0, 1, 0])
    truth = np.array([0, 1, 0])
    config = {"evaluation": {"metrics": ["accuracy", "nonexistent_metric"]}}
    result = evaluate_model(preds, truth, config)
    assert "accuracy" in result
    assert "nonexistent_metric" not in result


def test_evaluate_default_metrics():
    """No config should use default metrics (accuracy, f1_weighted)."""
    preds = np.array([0, 1, 0, 1])
    truth = np.array([0, 1, 0, 0])
    result = evaluate_model(preds, truth)
    assert "accuracy" in result
    assert "f1_weighted" in result


def test_evaluate_regression_metrics():
    """MAE and MSE metrics should work for regression tasks."""
    preds = np.array([1.0, 2.0, 3.0])
    truth = np.array([1.1, 2.2, 2.8])
    config = {"evaluation": {"metrics": ["mae", "mse", "rmse"]}}
    result = evaluate_model(preds, truth, config)
    assert "mae" in result
    assert "mse" in result
    assert "rmse" in result
    assert result["mae"] > 0
    assert result["rmse"] == pytest.approx(np.sqrt(result["mse"]), abs=0.001)


def test_evaluate_metrics_are_rounded():
    """All metric values should be rounded to 4 decimal places."""
    preds = np.array([0, 1, 0, 1, 0, 1, 0])
    truth = np.array([0, 1, 1, 1, 0, 0, 0])
    config = {"evaluation": {"metrics": ["accuracy", "f1_weighted"]}}
    result = evaluate_model(preds, truth, config)
    for value in result.values():
        if isinstance(value, float):
            # Check that string representation has at most 4 decimal places
            decimal_str = str(value).split(".")[-1]
            assert len(decimal_str) <= 4


# --- format_metrics ---


def test_format_metrics_delimiter():
    """Output must be wrapped in --- delimiters for grep parsing."""
    metrics = {"accuracy": 0.85, "f1_weighted": 0.82}
    output = format_metrics(metrics)
    lines = output.split("\n")
    assert lines[0] == "---"
    assert lines[-1] == "---"


def test_format_metrics_parseable():
    """Each metric line must be key:value format parseable by grep."""
    metrics = {"accuracy": 0.85, "f1_weighted": 0.82}
    output = format_metrics(metrics)
    lines = output.split("\n")
    for line in lines[1:-1]:  # Skip delimiters
        assert ":" in line
        key, value = line.split(":", 1)
        assert key.strip()
        assert value.strip()


def test_format_metrics_metadata_after_metrics():
    """Metadata keys (model_type, train_seconds) should appear after metric keys."""
    metrics = {
        "accuracy": 0.85,
        "model_type": "xgboost",
        "train_seconds": 2.5,
    }
    output = format_metrics(metrics)
    lines = output.split("\n")
    # Find positions
    metric_lines = [l for l in lines if "accuracy" in l]
    meta_lines = [l for l in lines if "model_type" in l or "train_seconds" in l]
    assert metric_lines  # accuracy should be present
    assert meta_lines  # metadata should be present
    # Metric should come before metadata
    acc_idx = lines.index(metric_lines[0])
    for meta_line in meta_lines:
        meta_idx = lines.index(meta_line)
        assert meta_idx > acc_idx
