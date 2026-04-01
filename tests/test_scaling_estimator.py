"""Tests for scaling law estimator (scaling_estimator.py).

Phase 18.1: Verifies power law fitting, extrapolation, scale point
generation, verdict logic, and report formatting.
"""

from __future__ import annotations

import math

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


# --- fit_power_law ---


def test_fit_power_law_basic():
    """Should fit a reasonable power law to synthetic data."""
    # Generate data from known power law: y = 2 * x^0.5
    scales = [0.1, 0.25, 0.5, 0.75, 1.0]
    values = [2 * (s ** 0.5) for s in scales]
    fit = fit_power_law(scales, values)
    assert "a" in fit
    assert "b" in fit
    assert fit["r_squared"] > 0.95


def test_fit_power_law_linear():
    """Linear data should have b close to 1."""
    scales = [0.1, 0.2, 0.5, 1.0]
    values = [s * 10 for s in scales]  # y = 10x
    fit = fit_power_law(scales, values)
    assert abs(fit["b"] - 1.0) < 0.1


def test_fit_power_law_two_points():
    """Should work with minimum 2 points."""
    fit = fit_power_law([0.25, 1.0], [0.5, 1.0])
    assert "error" not in fit
    assert fit["r_squared"] >= 0


def test_fit_power_law_single_point():
    """Single point should return error."""
    fit = fit_power_law([0.5], [0.8])
    assert "error" in fit


def test_fit_power_law_residuals():
    """Should include residuals for each point."""
    scales = [0.1, 0.25, 0.5, 1.0]
    values = [0.5, 0.6, 0.7, 0.8]
    fit = fit_power_law(scales, values)
    assert len(fit["residuals"]) == 4


# --- extrapolate ---


def test_extrapolate_basic():
    """Should produce predictions at target scales."""
    fit = {"a": 2.0, "b": 0.5, "c": 0.0}
    preds = extrapolate(fit, [1.0, 2.0])
    assert len(preds) == 2
    assert preds[0]["scale"] == 1.0
    assert abs(preds[0]["predicted_value"] - 2.0) < 0.01  # 2 * 1^0.5


def test_extrapolate_zero_scale():
    """Zero scale should produce None."""
    fit = {"a": 2.0, "b": 0.5, "c": 0.0}
    preds = extrapolate(fit, [0.0])
    assert preds[0]["predicted_value"] is None


def test_extrapolate_with_offset():
    """Should include c offset."""
    fit = {"a": 1.0, "b": 1.0, "c": 0.5}
    preds = extrapolate(fit, [1.0])
    assert abs(preds[0]["predicted_value"] - 1.5) < 0.01  # 1*1^1 + 0.5


# --- generate_scale_points ---


def test_scale_points_data_axis():
    """Data axis should set data_fraction."""
    points = generate_scale_points("data")
    assert len(points) == 4
    assert points[0]["fraction"] == 0.10
    assert "data_fraction" in points[0]["config_overrides"]


def test_scale_points_compute_axis():
    """Compute axis should set n_estimators."""
    config = {"model": {"hyperparams": {"n_estimators": 100}}}
    points = generate_scale_points("compute", config=config)
    assert points[0]["config_overrides"]["n_estimators"] == 10  # 10% of 100


def test_scale_points_params_axis():
    """Params axis should scale model size."""
    config = {"model": {"hyperparams": {"n_estimators": 200, "max_depth": 10}}}
    points = generate_scale_points("params", config=config)
    assert points[0]["config_overrides"]["n_estimators"] == 20  # 10% of 200
    assert points[0]["config_overrides"]["max_depth"] == 1  # 10% of 10


def test_scale_points_custom_fractions():
    """Should respect custom fractions."""
    points = generate_scale_points("data", fractions=[0.5, 1.0])
    assert len(points) == 2
    assert points[0]["fraction"] == 0.5


# --- compute_verdict ---


def test_verdict_diminishing_returns():
    """Small predicted gain should produce diminishing returns."""
    observed = [
        {"fraction": 0.5, "metric_value": 0.870},
        {"fraction": 0.75, "metric_value": 0.873},
    ]
    predictions = [{"scale": 1.0, "predicted_value": 0.875}]
    verdict = compute_verdict(observed, predictions, "accuracy")
    assert verdict["verdict"] == "diminishing_returns"


def test_verdict_worth_scaling():
    """Large predicted gain should say worth scaling."""
    observed = [
        {"fraction": 0.25, "metric_value": 0.70},
        {"fraction": 0.50, "metric_value": 0.80},
    ]
    predictions = [{"scale": 1.0, "predicted_value": 0.90}]
    verdict = compute_verdict(observed, predictions, "accuracy")
    assert verdict["verdict"] == "worth_scaling"


def test_verdict_marginal():
    """Medium gain should produce marginal_gains."""
    observed = [
        {"fraction": 0.75, "metric_value": 0.860},
    ]
    predictions = [{"scale": 1.0, "predicted_value": 0.872}]
    verdict = compute_verdict(observed, predictions, "accuracy")
    assert verdict["verdict"] == "marginal_gains"


def test_verdict_empty_data():
    """Empty data should produce insufficient_data."""
    verdict = compute_verdict([], [], "accuracy")
    assert verdict["verdict"] == "insufficient_data"


def test_verdict_no_full_prediction():
    """Missing full-scale prediction should note it."""
    observed = [{"fraction": 0.5, "metric_value": 0.8}]
    predictions = [{"scale": 2.0, "predicted_value": 0.9}]
    verdict = compute_verdict(observed, predictions, "accuracy")
    assert verdict["verdict"] == "no_prediction"


# --- analyze_scaling ---


def test_analyze_scaling_basic():
    """Should produce complete analysis from scale results."""
    results = [
        {"fraction": 0.1, "metric_value": 0.70},
        {"fraction": 0.25, "metric_value": 0.78},
        {"fraction": 0.5, "metric_value": 0.84},
        {"fraction": 0.75, "metric_value": 0.87},
    ]
    report = analyze_scaling(results, "accuracy")
    assert "power_law_fit" in report
    assert "predictions" in report
    assert "verdict" in report
    assert report["power_law_fit"]["r_squared"] > 0.9


def test_analyze_scaling_empty():
    """Empty results should return error."""
    report = analyze_scaling([], "accuracy")
    assert "error" in report


# --- format_scaling_report ---


def test_format_report_basic():
    """Should produce readable markdown."""
    report = {
        "analyzed_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "scale_points": [
            {"fraction": 0.25, "metric_value": 0.78},
            {"fraction": 0.50, "metric_value": 0.84},
        ],
        "power_law_fit": {"a": 0.9, "b": 0.1, "c": 0, "r_squared": 0.99},
        "predictions": [{"scale": 1.0, "predicted_value": 0.88}],
        "verdict": {"verdict": "worth_scaling", "reason": "Significant improvement expected"},
    }
    text = format_scaling_report(report)
    assert "Scaling Analysis" in text
    assert "Power Law Fit" in text
    assert "WORTH SCALING" in text


def test_format_report_error():
    """Error should show error message."""
    text = format_scaling_report({"error": "No data"})
    assert "ERROR" in text


# --- format_ascii_plot ---


def test_ascii_plot_basic():
    """Should produce non-empty plot."""
    results = [{"fraction": 0.25, "metric_value": 0.78}, {"fraction": 0.5, "metric_value": 0.84}]
    predictions = [{"scale": 1.0, "predicted_value": 0.88}]
    plot = format_ascii_plot(results, predictions, "accuracy")
    assert "o" in plot  # observed
    assert "*" in plot  # predicted


def test_ascii_plot_empty():
    """Empty data should produce placeholder."""
    plot = format_ascii_plot([], [], "accuracy")
    assert "no data" in plot
