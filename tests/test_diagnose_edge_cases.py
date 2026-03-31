"""Edge case tests for diagnose_errors.py."""

from __future__ import annotations

import pytest

from scripts.diagnose_errors import (
    analyze_feature_ranges,
    analyze_regression_errors,
    compute_confusion_matrix,
    identify_failure_modes,
)


def test_confusion_single_class():
    """Should handle single-class prediction."""
    y_true = ["A", "A", "A"]
    y_pred = ["A", "A", "A"]
    result = compute_confusion_matrix(y_true, y_pred)
    assert result["total_errors"] == 0
    assert len(result["classes"]) == 1


def test_confusion_all_wrong():
    """Should handle all-wrong predictions."""
    y_true = ["A", "A", "B", "B"]
    y_pred = ["B", "B", "A", "A"]
    result = compute_confusion_matrix(y_true, y_pred)
    assert result["error_rate"] == 1.0


def test_confusion_numeric_classes():
    """Should handle numeric class labels."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 1, 0, 2]
    result = compute_confusion_matrix(y_true, y_pred)
    assert len(result["classes"]) == 3
    assert result["total_errors"] == 2


def test_regression_single_sample():
    """Should handle single sample."""
    result = analyze_regression_errors([1.0], [1.5])
    assert result["mean_absolute_error"] == 0.5
    assert result["worst_predictions"][0]["residual"] == 0.5


def test_regression_large_residuals():
    """Should handle very large residuals without overflow."""
    y_true = [0.0] * 10
    y_pred = [1e6] * 10
    result = analyze_regression_errors(y_true, y_pred)
    assert result["mean_absolute_error"] == 1e6


def test_feature_ranges_insufficient_data():
    """Should skip features with too few samples."""
    y_true = [1.0, 2.0]
    y_pred = [1.5, 2.5]
    residuals = [0.5, 0.5]
    features = [{"x": 1}, {"x": 2}]
    result = analyze_feature_ranges(y_true, y_pred, residuals, features)
    assert result == []  # Too few samples for quartile analysis


def test_feature_ranges_nan_values():
    """Should skip NaN feature values."""
    import math
    y_true = [1.0] * 10
    y_pred = [1.1] * 10
    residuals = [0.1] * 10
    features = [{"x": float("nan") if i < 5 else float(i)} for i in range(10)]
    result = analyze_feature_ranges(y_true, y_pred, residuals, features)
    # Should not crash


def test_failure_modes_empty_inputs():
    """Should return empty list for empty inputs."""
    modes = identify_failure_modes()
    assert modes == []


def test_failure_modes_respects_top_n():
    """Should respect top_n limit."""
    confusion = {
        "most_confused": [
            {"true_class": f"A{i}", "predicted_class": f"B{i}", "count": 10 - i, "pct_of_true": 20}
            for i in range(10)
        ],
        "per_class": {},
        "total_errors": 100,
    }
    modes = identify_failure_modes(confusion_data=confusion, top_n=3)
    assert len(modes) <= 3
