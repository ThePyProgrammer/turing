"""Edge case tests for scaling law estimator (scaling_estimator.py).

Phase 18.1: Covers single point, perfect fit, constant values,
negative scales, zero values, and boundary conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.scaling_estimator import (
    analyze_scaling,
    compute_verdict,
    extrapolate,
    fit_power_law,
    format_ascii_plot,
    format_scaling_report,
    generate_scale_points,
)


# --- fit_power_law edge cases ---


def test_fit_empty():
    """Empty inputs should return error."""
    fit = fit_power_law([], [])
    assert "error" in fit


def test_fit_constant_values():
    """Constant metric values should handle gracefully."""
    fit = fit_power_law([0.1, 0.5, 1.0], [0.85, 0.85, 0.85])
    assert "a" in fit
    # R² may be low or NaN for constant data


def test_fit_negative_scales_filtered():
    """Negative scales should be filtered out."""
    fit = fit_power_law([-0.1, 0.25, 0.5, 1.0], [0.5, 0.7, 0.8, 0.9])
    assert "error" not in fit  # Should work with 3 positive points


def test_fit_identical_scales():
    """Identical scale values should handle division by zero."""
    fit = fit_power_law([0.5, 0.5], [0.8, 0.82])
    # Denominator is zero — should not crash
    assert "a" in fit


def test_fit_very_small_values():
    """Very small metric values should not produce NaN."""
    fit = fit_power_law([0.1, 0.5, 1.0], [1e-6, 2e-6, 3e-6])
    assert "a" in fit
    assert not any(np.isnan(v) for v in [fit["a"], fit["b"]])


def test_fit_large_exponent():
    """Steep scaling curve should produce large exponent."""
    scales = [0.1, 0.25, 0.5, 1.0]
    values = [s ** 2 for s in scales]  # Quadratic
    fit = fit_power_law(scales, values)
    assert fit["b"] > 1.0  # Exponent > 1 for quadratic


# --- extrapolate edge cases ---


def test_extrapolate_negative_scale():
    """Negative scale should produce None."""
    fit = {"a": 1.0, "b": 0.5, "c": 0.0}
    preds = extrapolate(fit, [-1.0])
    assert preds[0]["predicted_value"] is None


def test_extrapolate_very_large_scale():
    """Very large scale should still produce a value."""
    fit = {"a": 1.0, "b": 0.5, "c": 0.0}
    preds = extrapolate(fit, [1000.0])
    assert preds[0]["predicted_value"] is not None
    assert preds[0]["predicted_value"] > 0


def test_extrapolate_empty():
    """Empty target list should return empty."""
    assert extrapolate({"a": 1, "b": 1, "c": 0}, []) == []


# --- generate_scale_points edge cases ---


def test_scale_points_empty_fractions():
    """Empty fractions should produce empty points."""
    points = generate_scale_points("data", fractions=[])
    assert len(points) == 0


def test_scale_points_no_config():
    """No config should use defaults."""
    points = generate_scale_points("compute")
    assert len(points) == 4
    # Default n_estimators=100, 10% = 10
    assert points[0]["config_overrides"]["n_estimators"] == 10


def test_scale_points_single_fraction():
    """Single fraction should work."""
    points = generate_scale_points("data", fractions=[0.5])
    assert len(points) == 1
    assert points[0]["percentage"] == "50%"


def test_scale_points_params_minimum():
    """Params axis should have minimum of 1 for depth/estimators."""
    config = {"model": {"hyperparams": {"n_estimators": 5, "max_depth": 2}}}
    points = generate_scale_points("params", fractions=[0.1], config=config)
    assert points[0]["config_overrides"]["n_estimators"] >= 1
    assert points[0]["config_overrides"]["max_depth"] >= 1


# --- compute_verdict edge cases ---


def test_verdict_zero_metric_value():
    """Zero metric value should not divide by zero."""
    observed = [{"fraction": 0.75, "metric_value": 0.0}]
    predictions = [{"scale": 1.0, "predicted_value": 0.01}]
    verdict = compute_verdict(observed, predictions, "accuracy")
    assert "verdict" in verdict


def test_verdict_negative_gain():
    """Predicted degradation should still produce a verdict."""
    observed = [{"fraction": 0.75, "metric_value": 0.90}]
    predictions = [{"scale": 1.0, "predicted_value": 0.85}]
    verdict = compute_verdict(observed, predictions, "accuracy")
    # Negative gain is still a small absolute change
    assert "verdict" in verdict


def test_verdict_none_prediction():
    """None predicted value should handle gracefully."""
    observed = [{"fraction": 0.5, "metric_value": 0.8}]
    predictions = [{"scale": 1.0, "predicted_value": None}]
    verdict = compute_verdict(observed, predictions, "accuracy")
    assert verdict["verdict"] == "no_prediction"


# --- analyze_scaling edge cases ---


def test_analyze_two_points():
    """Should work with minimum 2 points."""
    results = [
        {"fraction": 0.25, "metric_value": 0.70},
        {"fraction": 0.75, "metric_value": 0.85},
    ]
    report = analyze_scaling(results, "accuracy")
    assert "predictions" in report
    assert "error" not in report


def test_analyze_with_std():
    """Should handle results with std deviation."""
    results = [
        {"fraction": 0.25, "metric_value": 0.70, "std": 0.01},
        {"fraction": 0.50, "metric_value": 0.80, "std": 0.008},
    ]
    report = analyze_scaling(results, "accuracy")
    assert "error" not in report


# --- format edge cases ---


def test_format_report_with_std():
    """Report with std should show ± notation."""
    report = {
        "analyzed_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "scale_points": [
            {"fraction": 0.25, "metric_value": 0.78, "std": 0.01},
        ],
        "power_law_fit": {"a": 0.9, "b": 0.1, "c": 0, "r_squared": 0.99},
        "predictions": [],
        "verdict": {"verdict": "diminishing_returns", "reason": "Small gain"},
    }
    text = format_scaling_report(report)
    assert "±" in text


def test_ascii_plot_single_point():
    """Single point should still render."""
    results = [{"fraction": 0.5, "metric_value": 0.8}]
    plot = format_ascii_plot(results, [], "accuracy")
    assert "o" in plot
