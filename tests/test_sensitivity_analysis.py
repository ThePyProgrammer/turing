"""Tests for hyperparameter sensitivity analysis (sensitivity_analysis.py).

Phase 21.2: Verifies sweep generation, sensitivity scoring, ranking, recommendations, reporting.
"""

from __future__ import annotations

import pytest

from scripts.sensitivity_analysis import (
    compute_sensitivity,
    extract_tunable_params,
    format_sensitivity_report,
    generate_recommendations,
    generate_sweep,
    rank_sensitivities,
    sensitivity_analysis,
    _check_monotonicity,
)


# --- generate_sweep ---

def test_sweep_basic():
    points = generate_sweep("lr", 0.1)
    assert len(points) == 5
    assert any(p["is_current"] for p in points)

def test_sweep_integer():
    points = generate_sweep("n_estimators", 100)
    assert all(isinstance(p["value"], int) for p in points)
    assert points[0]["value"] == 50  # 0.5x

def test_sweep_minimum():
    points = generate_sweep("max_depth", 1)
    assert all(p["value"] >= 1 for p in points)

# --- extract_tunable_params ---

def test_extract_params():
    config = {"model": {"hyperparams": {"learning_rate": 0.1, "max_depth": 6, "seed": 42}}}
    params = extract_tunable_params(config)
    assert "learning_rate" in params
    assert "seed" not in params

def test_extract_empty():
    assert extract_tunable_params({}) == {}

# --- compute_sensitivity ---

def test_sensitivity_high():
    results = [{"value": 0.01, "metric_value": 0.75},
               {"value": 0.1, "metric_value": 0.87, "is_current": True},
               {"value": 0.5, "metric_value": 0.82}]
    s = compute_sensitivity("lr", results, "accuracy")
    assert s["level"] == "HIGH"
    assert s["sensitivity"] > 0

def test_sensitivity_none():
    results = [{"value": 1, "metric_value": 0.870},
               {"value": 5, "metric_value": 0.871},
               {"value": 10, "metric_value": 0.872}]
    s = compute_sensitivity("min_child", results, "accuracy")
    assert s["level"] in ("NONE", "LOW")

def test_sensitivity_insufficient():
    s = compute_sensitivity("lr", [{"value": 0.1, "metric_value": 0.85}], "accuracy")
    assert s["level"] == "NONE"

def test_sensitivity_best_value():
    results = [{"value": 0.01, "metric_value": 0.75},
               {"value": 0.1, "metric_value": 0.87},
               {"value": 0.5, "metric_value": 0.80}]
    s = compute_sensitivity("lr", results, "accuracy")
    assert s["best_value"] == 0.1

# --- _check_monotonicity ---

def test_monotonic_increasing():
    assert _check_monotonicity([1, 2, 3, 4]) == "increasing"

def test_monotonic_decreasing():
    assert _check_monotonicity([4, 3, 2, 1]) == "decreasing"

def test_non_monotonic():
    assert _check_monotonicity([1, 3, 2, 4]) == "non_monotonic"

def test_monotonic_single():
    assert _check_monotonicity([5]) == "unknown"

# --- rank_sensitivities ---

def test_rank():
    sensitivities = [
        {"param": "a", "sensitivity": 0.01},
        {"param": "b", "sensitivity": 0.05},
        {"param": "c", "sensitivity": 0.001},
    ]
    ranked = rank_sensitivities(sensitivities)
    assert ranked[0]["param"] == "b"
    assert ranked[-1]["param"] == "c"

# --- generate_recommendations ---

def test_recommendations():
    ranked = [
        {"param": "lr", "level": "HIGH", "sensitivity": 0.05, "monotonic": "non_monotonic", "best_value": 0.1},
        {"param": "depth", "level": "NONE", "sensitivity": 0.001, "monotonic": "increasing"},
    ]
    recs = generate_recommendations(ranked)
    assert any("lr" in r for r in recs)
    assert any("depth" in r and "Stop" in r for r in recs)

def test_recommendations_empty():
    assert generate_recommendations([]) == []

# --- sensitivity_analysis ---

def test_analysis_plan():
    """Without sweep data, should return plan."""
    report = sensitivity_analysis()
    # With default empty config, should error about no params
    assert "error" in report or report.get("action") == "plan"

def test_analysis_with_data():
    sweep_data = {
        "lr": [{"value": 0.01, "metric_value": 0.75}, {"value": 0.1, "metric_value": 0.87}],
        "depth": [{"value": 3, "metric_value": 0.86}, {"value": 6, "metric_value": 0.87}],
    }
    report = sensitivity_analysis(sweep_data=sweep_data)
    assert "sensitivities" in report
    assert len(report["sensitivities"]) == 2

# --- format_sensitivity_report ---

def test_format_analysis():
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "experiment_id": "exp-042",
        "sensitivities": [
            {"param": "lr", "current_value": 0.1, "sensitivity": 0.04, "level": "HIGH",
             "metric_min": 0.83, "metric_max": 0.87, "metric_range": 0.04},
        ],
        "recommendations": ["Focus tuning on lr"],
    }
    text = format_sensitivity_report(report)
    assert "Sensitivity" in text
    assert "lr" in text
    assert "HIGH" in text

def test_format_plan():
    report = {"action": "plan", "sweep_plans": {"lr": [{"value": 0.05}, {"value": 0.1}]},
              "n_experiments_needed": 10}
    text = format_sensitivity_report(report)
    assert "Plan" in text

def test_format_error():
    assert "ERROR" in format_sensitivity_report({"error": "No params"})
