"""Tests for multi-run statistical comparison (statistical_compare.py).

Phase 2.1: Verifies statistics computation, metric parsing, and
Mann-Whitney U test — the functions that prevent keep/discard
decisions based on single-run noise.
"""

from __future__ import annotations

import pytest

from scripts.statistical_compare import (
    compute_statistics,
    mann_whitney_test,
    parse_metrics_from_output,
)


# --- parse_metrics_from_output ---


def test_parse_metrics_basic():
    """Should parse standard --- delimited format."""
    output = "---\naccuracy:       0.8500\nf1_weighted:    0.8200\n---\n"
    metrics = parse_metrics_from_output(output)
    assert metrics["accuracy"] == 0.85
    assert metrics["f1_weighted"] == 0.82


def test_parse_metrics_with_metadata():
    """Should separate metadata from numeric metrics."""
    output = "---\naccuracy:       0.8500\nmodel_type:     xgboost\ntrain_seconds:  2.5\n---\n"
    metrics = parse_metrics_from_output(output)
    assert metrics["accuracy"] == 0.85
    assert metrics["model_type"] == "xgboost"
    assert metrics["train_seconds"] == "2.5"


def test_parse_metrics_with_surrounding_text():
    """Should ignore text outside --- delimiters."""
    output = "Training started...\nEpoch 1 done\n---\naccuracy:  0.90\n---\nModel saved.\n"
    metrics = parse_metrics_from_output(output)
    assert metrics["accuracy"] == 0.90


def test_parse_metrics_empty():
    """Empty output should return empty dict."""
    assert parse_metrics_from_output("") == {}
    assert parse_metrics_from_output("no delimiters here") == {}


# --- compute_statistics ---


def test_statistics_basic():
    """Should compute correct mean, std, min, max."""
    results = [
        {"accuracy": 0.85},
        {"accuracy": 0.87},
        {"accuracy": 0.83},
    ]
    stats = compute_statistics(results, "accuracy")
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(0.85, abs=0.001)
    assert stats["min"] == pytest.approx(0.83, abs=0.001)
    assert stats["max"] == pytest.approx(0.87, abs=0.001)
    assert stats["std"] > 0


def test_statistics_single_value():
    """Single value should have std=0 and tight CI."""
    results = [{"accuracy": 0.85}]
    stats = compute_statistics(results, "accuracy")
    assert stats["n"] == 1
    assert stats["mean"] == 0.85
    assert stats["std"] == 0.0


def test_statistics_confidence_interval():
    """CI should contain the mean and be reasonable."""
    results = [{"accuracy": v} for v in [0.84, 0.85, 0.86, 0.85, 0.84]]
    stats = compute_statistics(results, "accuracy")
    assert stats["ci_lower"] < stats["mean"]
    assert stats["ci_upper"] > stats["mean"]
    assert stats["ci_lower"] > 0.8  # sanity
    assert stats["ci_upper"] < 0.9  # sanity


def test_statistics_missing_metric():
    """Missing metric should return n=0."""
    results = [{"f1": 0.80}]
    stats = compute_statistics(results, "accuracy")
    assert stats["n"] == 0


def test_statistics_values_preserved():
    """Raw values should be returned for inspection."""
    results = [{"accuracy": 0.85}, {"accuracy": 0.87}]
    stats = compute_statistics(results, "accuracy")
    assert len(stats["values"]) == 2


# --- mann_whitney_test ---


def test_mann_whitney_significantly_different():
    """Clearly different distributions should be significant."""
    a = [0.80, 0.81, 0.79, 0.80, 0.82]
    b = [0.90, 0.91, 0.89, 0.90, 0.92]
    result = mann_whitney_test(a, b)
    assert result["verdict"] == "significantly_different"
    assert result["p_value"] < 0.05


def test_mann_whitney_not_significant():
    """Nearly identical distributions should not be significant."""
    a = [0.850, 0.851, 0.849, 0.850, 0.852]
    b = [0.851, 0.850, 0.852, 0.849, 0.851]
    result = mann_whitney_test(a, b)
    assert result["verdict"] == "not_significant"
    assert result["p_value"] >= 0.05


def test_mann_whitney_insufficient_data():
    """Fewer than 2 values should return insufficient_data."""
    result = mann_whitney_test([0.85], [0.86])
    assert result["verdict"] == "insufficient_data"


def test_mann_whitney_returns_all_fields():
    """Result should contain all expected fields."""
    result = mann_whitney_test([0.80, 0.81, 0.82], [0.90, 0.91, 0.92])
    assert "statistic" in result
    assert "p_value" in result
    assert "verdict" in result
    assert "detail" in result
