"""Tests for automatic baseline generation (generate_baselines.py).

Phase 20.2: Verifies baseline methods, evaluation, report formatting.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.generate_baselines import (
    evaluate_predictions,
    format_baseline_report,
    generate_baselines,
    majority_or_mean_baseline,
    random_baseline,
    stratified_or_median_baseline,
    _method_display_name,
)


# --- random_baseline ---

def test_random_classification():
    y = np.array([0, 1, 0, 1, 0])
    preds = random_baseline(y, "classification")
    assert len(preds) == 5
    assert all(p in [0, 1] for p in preds)

def test_random_regression():
    y = np.array([1.0, 2.0, 3.0])
    preds = random_baseline(y, "regression")
    assert len(preds) == 3

# --- majority_or_mean_baseline ---

def test_majority_classification():
    y = np.array([0, 0, 0, 1, 1])
    preds = majority_or_mean_baseline(y, "classification")
    assert all(p == 0 for p in preds)

def test_mean_regression():
    y = np.array([2.0, 4.0, 6.0])
    preds = majority_or_mean_baseline(y, "regression")
    np.testing.assert_almost_equal(preds[0], 4.0)

# --- stratified_or_median_baseline ---

def test_stratified_classification():
    y = np.array([0, 0, 0, 1, 1])
    np.random.seed(42)
    preds = stratified_or_median_baseline(y, "classification")
    assert len(preds) == 5
    assert all(p in [0, 1] for p in preds)

def test_median_regression():
    y = np.array([1.0, 3.0, 5.0])
    preds = stratified_or_median_baseline(y, "regression")
    assert all(p == 3.0 for p in preds)

# --- evaluate_predictions ---

def test_eval_classification():
    preds = np.array([0, 1, 1, 0])
    labels = np.array([0, 1, 0, 0])
    result = evaluate_predictions(preds, labels, "classification")
    assert result["accuracy"] == 0.75

def test_eval_regression():
    preds = np.array([1.0, 2.0, 3.0])
    labels = np.array([1.0, 2.0, 3.0])
    result = evaluate_predictions(preds, labels, "regression")
    assert result["mse"] == 0.0

def test_eval_mismatched_lengths():
    preds = np.array([0, 1, 1, 0, 1])
    labels = np.array([0, 1, 0])
    result = evaluate_predictions(preds, labels, "classification")
    assert result["n_samples"] == 3

# --- _method_display_name ---

def test_display_name_classification():
    assert _method_display_name("majority_or_mean", "classification") == "Majority class"

def test_display_name_regression():
    assert _method_display_name("majority_or_mean", "regression") == "Mean predictor"

def test_display_name_linear():
    assert _method_display_name("linear", "classification") == "Logistic Regression"

# --- generate_baselines ---

def test_generate_no_data():
    """Without data, should produce plan."""
    report = generate_baselines(methods="simple")
    assert "baselines" in report
    assert report.get("note") is not None  # No data note

def test_generate_all_methods():
    """All methods should be listed."""
    report = generate_baselines(methods="all")
    assert len(report["baselines"]) == 5

def test_generate_simple_methods():
    report = generate_baselines(methods="simple")
    assert len(report["baselines"]) == 2

# --- format_baseline_report ---

def test_format_report():
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "baselines": [
            {"method": "Majority class", "metric_value": 0.627, "notes": "Naive floor"},
            {"method": "Logistic Reg.", "metric_value": 0.814, "notes": "Linear ceiling"},
        ],
        "current_best": 0.872,
        "improvement_over_linear": 0.058,
    }
    text = format_baseline_report(report)
    assert "Baselines" in text
    assert "0.627" in text

def test_format_error():
    text = format_baseline_report({"error": "Failed"})
    assert "ERROR" in text

def test_format_no_data():
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "task_type": "classification",
        "primary_metric": "accuracy",
        "baselines": [{"method": "Random", "metric_value": None, "notes": "Requires data"}],
        "note": "No data loaded",
    }
    text = format_baseline_report(report)
    assert "N/A" in text
