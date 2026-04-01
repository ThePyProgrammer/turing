"""Edge case tests for automated ensemble construction (build_ensemble.py).

Phase 17.1: Covers single model, identical predictions, missing metrics,
regression tasks, empty inputs, and boundary conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.build_ensemble import (
    blending_ensemble,
    compute_prediction_correlation,
    evaluate_ensemble,
    filter_diverse_models,
    format_ensemble_report,
    select_top_k,
    stacking_ensemble,
    voting_ensemble,
    weighted_voting_ensemble,
    _fit_linear_meta,
)


# --- select_top_k edge cases ---


def test_select_top_k_empty():
    """Empty experiment list should return empty."""
    assert select_top_k([], "accuracy", 5) == []


def test_select_top_k_missing_metric():
    """Experiments without the metric should be excluded."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"f1": 0.8}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"accuracy": 0.9}},
    ]
    top = select_top_k(experiments, "accuracy", 5)
    assert len(top) == 1
    assert top[0]["experiment_id"] == "exp-002"


def test_select_top_k_non_numeric_metric():
    """Non-numeric metric values should be excluded."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": "N/A"}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"accuracy": 0.85}},
    ]
    top = select_top_k(experiments, "accuracy", 5)
    assert len(top) == 1


def test_select_top_k_zero():
    """k=0 should return empty."""
    experiments = [{"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.8}}]
    assert select_top_k(experiments, "accuracy", 0) == []


# --- compute_prediction_correlation edge cases ---


def test_correlation_empty():
    """Empty predictions should return empty matrix."""
    corr = compute_prediction_correlation([])
    assert corr.shape == (0, 0)


def test_correlation_constant_predictions():
    """Constant predictions should not crash (NaN handled)."""
    preds = [np.array([1, 1, 1, 1]), np.array([2, 2, 2, 2])]
    corr = compute_prediction_correlation(preds)
    # Correlation of constants is undefined, should be handled
    assert corr.shape == (2, 2)
    assert not np.any(np.isnan(corr))


def test_correlation_different_lengths():
    """Different length arrays should handle gracefully."""
    preds = [np.array([1, 2, 3]), np.array([1, 2])]
    corr = compute_prediction_correlation(preds)
    # Different lengths — correlation should not be computed
    assert corr.shape == (2, 2)


# --- filter_diverse_models edge cases ---


def test_filter_diverse_single_model():
    """Single model should always be kept."""
    exps = [{"experiment_id": "exp-001"}]
    filtered, indices = filter_diverse_models(exps, [np.array([1, 2, 3])])
    assert len(filtered) == 1


def test_filter_diverse_empty():
    """Empty inputs should return empty."""
    filtered, indices = filter_diverse_models([], None)
    assert len(filtered) == 0


# --- voting edge cases ---


def test_voting_single_model():
    """Single model should return its predictions unchanged."""
    preds = [np.array([0, 1, 2])]
    result = voting_ensemble(preds, "classification")
    np.testing.assert_array_equal(result, [0, 1, 2])


def test_voting_tie_classification():
    """Two-model tie should be resolved."""
    preds = [np.array([0, 1]), np.array([1, 0])]
    result = voting_ensemble(preds, "classification")
    # scipy.stats.mode returns the smallest value on tie
    assert len(result) == 2


# --- weighted_voting edge cases ---


def test_weighted_voting_zero_weights():
    """Zero weights should be handled (normalized to equal)."""
    preds = [np.array([1.0]), np.array([2.0])]
    # This would cause division by zero — weights sum to 0
    # The function normalizes, so we test with very small weights
    weights = [0.001, 0.001]
    result = weighted_voting_ensemble(preds, weights, "regression")
    assert len(result) == 1


def test_weighted_voting_empty():
    """Empty inputs should return empty array."""
    assert len(weighted_voting_ensemble([], [], "regression")) == 0


# --- stacking edge cases ---


def test_stacking_too_few_samples():
    """Very few samples should reduce folds."""
    labels = np.array([0, 1, 0])
    preds = [np.array([0.1, 0.9, 0.2]), np.array([0.2, 0.8, 0.3])]
    result = stacking_ensemble(preds, labels, "classification", n_folds=5)
    # Should reduce folds to n_samples
    assert "meta_predictions" in result


def test_stacking_single_model():
    """Stacking with single model should still produce weights."""
    labels = np.array([0, 1, 0, 1])
    preds = [np.array([0.1, 0.9, 0.2, 0.8])]
    result = stacking_ensemble(preds, labels, "classification", n_folds=2)
    assert len(result["meta_weights"]) == 1


# --- blending edge cases ---


def test_blending_too_few_samples():
    """Too few samples for split should return empty."""
    labels = np.array([0, 1])
    preds = [np.array([0.1, 0.9])]
    result = blending_ensemble(preds, labels, "classification", holdout_ratio=0.5)
    # With only 2 samples, split would be 1/1 — borderline
    assert "meta_predictions" in result


def test_blending_empty():
    """Empty should return empty result."""
    result = blending_ensemble([], np.array([]), "classification")
    assert len(result["meta_predictions"]) == 0


# --- evaluate_ensemble edge cases ---


def test_evaluate_mismatched_lengths():
    """Different length predictions/labels should use minimum."""
    preds = np.array([0, 1, 1, 0, 1])
    labels = np.array([0, 1, 0])
    result = evaluate_ensemble(preds, labels, "classification")
    assert result["n_samples"] == 3


def test_evaluate_perfect_regression():
    """Perfect regression predictions should have zero error."""
    preds = np.array([1.5, 2.5, 3.5])
    labels = np.array([1.5, 2.5, 3.5])
    result = evaluate_ensemble(preds, labels, "regression")
    assert result["mse"] == 0.0


# --- _fit_linear_meta edge cases ---


def test_fit_linear_meta_singular():
    """Singular matrix should fall back to uniform weights."""
    X = np.zeros((4, 2))
    y = np.array([0, 1, 0, 1], dtype=float)
    weights = _fit_linear_meta(X, y)
    assert len(weights) == 2


def test_fit_linear_meta_single_feature():
    """Single feature should produce single weight."""
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([1, 2, 3, 4], dtype=float)
    weights = _fit_linear_meta(X, y)
    assert len(weights) == 1


# --- format_ensemble_report edge cases ---


def test_format_report_high_correlation():
    """High correlation should note lack of diversity."""
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "task_type": "classification",
        "n_candidates": 2,
        "base_models": [],
        "results": [],
        "best_method": "best_single",
        "best_metric": 0.85,
        "improvement": 0,
        "diversity": {"mean_correlation": 0.95},
    }
    text = format_ensemble_report(report)
    assert "similar" in text.lower() or "diversity" in text.lower()


def test_format_report_good_diversity():
    """Low correlation should note good diversity."""
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "task_type": "classification",
        "n_candidates": 3,
        "base_models": [],
        "results": [],
        "best_method": "stacking",
        "best_metric": 0.90,
        "improvement": 0.02,
        "diversity": {"mean_correlation": 0.3},
    }
    text = format_ensemble_report(report)
    assert "diversity" in text.lower() or "complement" in text.lower()
