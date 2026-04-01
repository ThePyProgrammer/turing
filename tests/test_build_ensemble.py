"""Tests for automated ensemble construction (build_ensemble.py).

Phase 17.1: Verifies model selection, diversity filtering, ensemble methods
(voting, weighted voting, stacking, blending), evaluation, and report formatting.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.build_ensemble import (
    DEFAULT_METHODS,
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


# --- select_top_k ---


def test_select_top_k_basic():
    """Should select top-K by primary metric."""
    experiments = [
        {"experiment_id": f"exp-{i:03d}", "status": "kept", "metrics": {"accuracy": 0.70 + i * 0.05}}
        for i in range(5)
    ]
    top = select_top_k(experiments, "accuracy", 3)
    assert len(top) == 3
    assert abs(top[0]["metrics"]["accuracy"] - 0.90) < 1e-6  # Highest first


def test_select_top_k_lower_is_better():
    """Lower-is-better should sort ascending."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"loss": 0.3}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"loss": 0.1}},
        {"experiment_id": "exp-003", "status": "kept", "metrics": {"loss": 0.2}},
    ]
    top = select_top_k(experiments, "loss", 2, lower_is_better=True)
    assert top[0]["metrics"]["loss"] == 0.1


def test_select_top_k_fewer_than_k():
    """Should return all if fewer than K experiments."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.8}},
    ]
    top = select_top_k(experiments, "accuracy", 5)
    assert len(top) == 1


def test_select_top_k_fallback_to_all():
    """Should fall back to all experiments if none are kept."""
    experiments = [
        {"experiment_id": "exp-001", "status": "discarded", "metrics": {"accuracy": 0.8}},
    ]
    top = select_top_k(experiments, "accuracy", 5)
    assert len(top) == 1


# --- compute_prediction_correlation ---


def test_correlation_identical():
    """Identical predictions should have correlation 1.0."""
    preds = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
    corr = compute_prediction_correlation(preds)
    assert abs(corr[0, 1] - 1.0) < 1e-6


def test_correlation_uncorrelated():
    """Uncorrelated predictions should have low correlation."""
    np.random.seed(42)
    preds = [np.random.randn(100), np.random.randn(100)]
    corr = compute_prediction_correlation(preds)
    assert abs(corr[0, 1]) < 0.3


def test_correlation_single_model():
    """Single model should return identity matrix."""
    corr = compute_prediction_correlation([np.array([1, 2, 3])])
    assert corr.shape == (1, 1)
    assert corr[0, 0] == 1.0


# --- filter_diverse_models ---


def test_filter_diverse_keeps_diverse():
    """Should keep models with low correlation."""
    exps = [{"experiment_id": f"exp-{i}", "metrics": {"acc": 0.8 + i * 0.01}} for i in range(3)]
    np.random.seed(42)
    preds = [np.random.randn(50) for _ in range(3)]
    filtered, indices = filter_diverse_models(exps, preds, threshold=0.95)
    assert len(filtered) == 3  # All diverse enough


def test_filter_diverse_removes_duplicates():
    """Should remove models with very similar predictions."""
    exps = [
        {"experiment_id": "exp-001", "metrics": {"acc": 0.85}},
        {"experiment_id": "exp-002", "metrics": {"acc": 0.84}},
    ]
    preds = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
    filtered, indices = filter_diverse_models(exps, preds, threshold=0.95)
    assert len(filtered) == 1


def test_filter_diverse_no_predictions():
    """Should return all models if no predictions available."""
    exps = [{"experiment_id": "exp-001"}, {"experiment_id": "exp-002"}]
    filtered, indices = filter_diverse_models(exps, None)
    assert len(filtered) == 2


# --- voting_ensemble ---


def test_voting_classification():
    """Majority vote should pick most common class."""
    preds = [
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
        np.array([1, 1, 1, 0]),
    ]
    result = voting_ensemble(preds, "classification")
    assert result[0] == 0  # 2 votes for 0
    assert result[1] == 1  # 3 votes for 1
    assert result[3] == 0  # 3 votes for 0


def test_voting_regression():
    """Mean should be used for regression."""
    preds = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])]
    result = voting_ensemble(preds, "regression")
    np.testing.assert_array_almost_equal(result, [1.5, 2.5, 3.5])


def test_voting_empty():
    """Empty predictions should return empty array."""
    result = voting_ensemble([], "classification")
    assert len(result) == 0


# --- weighted_voting_ensemble ---


def test_weighted_voting_regression():
    """Weighted average for regression."""
    preds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    weights = [0.75, 0.25]
    result = weighted_voting_ensemble(preds, weights, "regression")
    np.testing.assert_array_almost_equal(result, [1.5, 2.5])


def test_weighted_voting_classification():
    """Weighted vote for classification."""
    preds = [
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([1, 0]),
    ]
    weights = [10.0, 1.0, 1.0]  # First model has much higher weight
    result = weighted_voting_ensemble(preds, weights, "classification")
    assert result[0] == 0  # First model dominates


# --- stacking_ensemble ---


def test_stacking_basic():
    """Stacking should produce predictions with meta-weights."""
    np.random.seed(42)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    preds = [
        np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15, 0.85, 0.25, 0.75]),
        np.array([0.2, 0.8, 0.1, 0.9, 0.25, 0.75, 0.2, 0.8, 0.3, 0.7]),
    ]
    result = stacking_ensemble(preds, labels, "classification", n_folds=2)
    assert "meta_predictions" in result
    assert "meta_weights" in result
    assert len(result["meta_predictions"]) == 10


def test_stacking_empty():
    """Empty predictions should return empty result."""
    result = stacking_ensemble([], np.array([]), "classification")
    assert len(result["meta_predictions"]) == 0


# --- blending_ensemble ---


def test_blending_basic():
    """Blending should produce holdout predictions."""
    np.random.seed(42)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    preds = [
        np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15, 0.85, 0.25, 0.75]),
        np.array([0.2, 0.8, 0.1, 0.9, 0.25, 0.75, 0.2, 0.8, 0.3, 0.7]),
    ]
    result = blending_ensemble(preds, labels, "classification")
    assert "meta_predictions" in result
    assert result["holdout_size"] > 0


# --- evaluate_ensemble ---


def test_evaluate_classification():
    """Should compute accuracy for classification."""
    preds = np.array([0, 1, 1, 0, 1])
    labels = np.array([0, 1, 0, 0, 1])
    result = evaluate_ensemble(preds, labels, "classification")
    assert result["accuracy"] == 0.8
    assert result["n_samples"] == 5


def test_evaluate_regression():
    """Should compute MSE/RMSE for regression."""
    preds = np.array([1.0, 2.0, 3.0])
    labels = np.array([1.0, 2.0, 3.0])
    result = evaluate_ensemble(preds, labels, "regression")
    assert result["mse"] == 0.0
    assert result["rmse"] == 0.0


def test_evaluate_empty():
    """Empty inputs should return empty dict."""
    assert evaluate_ensemble(np.array([]), np.array([]), "classification") == {}


# --- _fit_linear_meta ---


def test_fit_linear_meta():
    """Should fit linear weights."""
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
    y = np.array([1, 0, 1, 0], dtype=float)
    weights = _fit_linear_meta(X, y)
    assert len(weights) == 2


# --- format_ensemble_report ---


def test_format_report_basic():
    """Should produce readable markdown."""
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "task_type": "classification",
        "n_candidates": 3,
        "base_models": [
            {"experiment_id": "exp-001", "model_type": "xgboost", "accuracy": 0.85},
        ],
        "results": [
            {"method": "best_single", "metric_value": 0.85, "delta": 0.0},
            {"method": "stacking", "metric_value": 0.87, "delta": 0.02},
        ],
        "best_method": "stacking",
        "best_metric": 0.87,
        "improvement": 0.02,
        "diversity": {"mean_correlation": 0.6},
    }
    text = format_ensemble_report(report)
    assert "Ensemble Results" in text
    assert "stacking" in text
    assert "+0.0200" in text


def test_format_report_error():
    """Error should show error message."""
    text = format_ensemble_report({"error": "Not enough models"})
    assert "ERROR" in text


def test_format_report_no_improvement():
    """Should note when no ensemble beats the single model."""
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "task_type": "classification",
        "n_candidates": 3,
        "base_models": [],
        "results": [{"method": "best_single", "metric_value": 0.85, "delta": 0.0}],
        "best_method": "best_single",
        "best_metric": 0.85,
        "improvement": 0.0,
        "diversity": {},
    }
    text = format_ensemble_report(report)
    assert "No ensemble method improved" in text
