"""Tests for Bayesian-guided suggestion engine (suggest_next.py).

Phase 2.2: Verifies surrogate model, feature extraction, and
suggestion generation — the data-driven complement to human taste.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.suggest_next import (
    extract_features_and_targets,
    features_to_matrix,
    format_suggestions,
    suggest_configurations,
)


def _make_exp(exp_id: str, accuracy: float, n_est: int = 100, depth: int = 4, lr: float = 0.1) -> dict:
    return {
        "experiment_id": exp_id,
        "status": "kept",
        "config": {
            "model_type": "xgboost",
            "hyperparams": {
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
            },
        },
        "metrics": {"accuracy": accuracy},
    }


# --- extract_features_and_targets ---


def test_extract_basic():
    exps = [_make_exp("exp-001", 0.85, 100, 4, 0.1)]
    features, targets, ids = extract_features_and_targets(exps, "accuracy")
    assert len(features) == 1
    assert targets[0] == 0.85
    assert ids[0] == "exp-001"
    assert features[0]["n_estimators"] == 100


def test_extract_skips_missing_metric():
    exps = [
        _make_exp("exp-001", 0.85),
        {"experiment_id": "exp-002", "config": {"hyperparams": {"n_estimators": 50}}, "metrics": {}},
    ]
    features, targets, ids = extract_features_and_targets(exps, "accuracy")
    assert len(features) == 1


def test_extract_empty():
    features, targets, ids = extract_features_and_targets([], "accuracy")
    assert features == []


# --- features_to_matrix ---


def test_matrix_basic():
    dicts = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
    X, cols = features_to_matrix(dicts)
    assert X.shape == (2, 2)
    assert "a" in cols
    assert "b" in cols


def test_matrix_missing_keys():
    """Missing keys should be NaN."""
    dicts = [{"a": 1.0, "b": 2.0}, {"a": 3.0}]
    X, cols = features_to_matrix(dicts)
    assert X.shape == (2, 2)
    assert np.isnan(X[1, cols.index("b")])


def test_matrix_empty():
    X, cols = features_to_matrix([])
    assert X.shape[0] == 0


# --- suggest_configurations ---


def test_suggest_insufficient_data():
    """Fewer than 3 experiments should return insufficient_data."""
    exps = [_make_exp("exp-001", 0.85)]
    result = suggest_configurations(exps, "accuracy")
    assert result[0]["reason"] == "insufficient_data"


def test_suggest_returns_configs():
    """With enough data, should return N suggestion dicts."""
    exps = [
        _make_exp("exp-001", 0.80, 50, 3, 0.01),
        _make_exp("exp-002", 0.85, 100, 4, 0.1),
        _make_exp("exp-003", 0.83, 200, 6, 0.05),
        _make_exp("exp-004", 0.87, 150, 5, 0.08),
    ]
    result = suggest_configurations(exps, "accuracy", n_suggestions=3)
    assert len(result) == 3
    assert "config" in result[0]
    assert "predicted_metric" in result[0]
    assert "uncertainty" in result[0]


def test_suggest_configs_are_numeric():
    """Suggested config values should all be numeric."""
    exps = [
        _make_exp(f"exp-{i:03d}", 0.80 + i * 0.01, 50 + i * 25, 3 + i, 0.01 + i * 0.02)
        for i in range(5)
    ]
    result = suggest_configurations(exps, "accuracy", n_suggestions=2)
    for s in result:
        for v in s["config"].values():
            assert isinstance(v, (int, float))


# --- format_suggestions ---


def test_format_suggestions_with_data():
    suggestions = [
        {"config": {"n_estimators": 150, "max_depth": 5}, "predicted_metric": 0.88, "uncertainty": 0.02, "acquisition_score": 0.91},
    ]
    output = format_suggestions(suggestions, "accuracy")
    assert "n_estimators=150" in output
    assert "0.88" in output


def test_format_suggestions_insufficient():
    suggestions = [{"reason": "insufficient_data", "detail": "Need 3 experiments", "suggestion": "Run more"}]
    output = format_suggestions(suggestions, "accuracy")
    assert "Need 3 experiments" in output
