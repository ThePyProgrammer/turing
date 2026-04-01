"""Tests for experiment outcome simulator (experiment_simulator.py).

Phase 27.3: Verifies surrogate building, prediction, ranking, budget savings.
"""

from __future__ import annotations

import pytest

from scripts.experiment_simulator import (
    extract_config_features,
    experiment_to_features,
    build_surrogate,
    predict_with_surrogate,
    simulate_experiments,
    format_simulation_report,
    _config_distance,
    _flatten,
    _summarize_config,
    MIN_HISTORY_FOR_SURROGATE,
)


# --- extract_config_features ---

def test_extract_flat():
    features = extract_config_features({"learning_rate": 0.1, "max_depth": 6})
    assert features["learning_rate"] == 0.1
    assert features["max_depth"] == 6.0

def test_extract_nested():
    features = extract_config_features({"model": {"hyperparams": {"lr": 0.05}}})
    assert "model.hyperparams.lr" in features

def test_extract_filters_non_numeric():
    features = extract_config_features({"name": "xgboost", "lr": 0.1, "verbose": True})
    assert "lr" in features
    assert "name" not in features
    assert "verbose" not in features

def test_extract_empty():
    assert extract_config_features({}) == {}


# --- experiment_to_features ---

def test_exp_features_from_config():
    exp = {"config": {"hyperparams": {"learning_rate": 0.1, "max_depth": 6}}}
    features = experiment_to_features(exp)
    assert "learning_rate" in features

def test_exp_features_top_level():
    exp = {"learning_rate": 0.05, "config": {}}
    features = experiment_to_features(exp)
    assert features["learning_rate"] == 0.05

def test_exp_features_empty():
    assert experiment_to_features({}) == {}


# --- build_surrogate ---

def _make_experiments(n: int, metric_base: float = 0.8) -> list[dict]:
    """Generate N dummy experiments with varying configs and metrics."""
    exps = []
    for i in range(n):
        exps.append({
            "config": {"hyperparams": {"learning_rate": 0.01 * (i + 1), "max_depth": 3 + i}},
            "metrics": {"accuracy": metric_base + 0.01 * i},
        })
    return exps

def test_build_surrogate_ready():
    exps = _make_experiments(MIN_HISTORY_FOR_SURROGATE)
    surrogate = build_surrogate(exps, "accuracy")
    assert surrogate["status"] == "ready"
    assert surrogate["n_points"] == MIN_HISTORY_FOR_SURROGATE

def test_build_surrogate_insufficient():
    exps = _make_experiments(2)
    surrogate = build_surrogate(exps, "accuracy")
    assert surrogate["status"] == "insufficient"

def test_build_surrogate_missing_metric():
    exps = [{"config": {"hyperparams": {"lr": 0.1}}, "metrics": {}}] * 10
    surrogate = build_surrogate(exps, "accuracy")
    assert surrogate["status"] == "insufficient"


# --- predict_with_surrogate ---

def test_predict_basic():
    exps = _make_experiments(10)
    surrogate = build_surrogate(exps, "accuracy")
    config_features = {"learning_rate": 0.05, "max_depth": 6.0}
    pred = predict_with_surrogate(surrogate, config_features)
    assert pred["predicted"] is not None
    assert pred["uncertainty"] is not None
    assert pred["predicted"] > 0

def test_predict_not_ready():
    pred = predict_with_surrogate({"status": "insufficient"}, {})
    assert pred["predicted"] is None

def test_predict_exact_match():
    """Config that matches a training point should have low uncertainty."""
    exps = _make_experiments(10)
    surrogate = build_surrogate(exps, "accuracy")
    # Use exact features from first experiment
    features = experiment_to_features(exps[0])
    pred = predict_with_surrogate(surrogate, features)
    assert pred["predicted"] is not None


# --- _config_distance ---

def test_distance_identical():
    a = {"lr": 0.1, "depth": 6.0}
    dist = _config_distance(a, a, ["lr", "depth"])
    assert dist == 0.0

def test_distance_different():
    a = {"lr": 0.01}
    b = {"lr": 0.1}
    dist = _config_distance(a, b, ["lr"])
    assert dist > 0

def test_distance_no_overlap():
    a = {"lr": 0.1}
    b = {"depth": 6}
    dist = _config_distance(a, b, ["lr", "depth"])
    assert dist == float("inf")


# --- simulate_experiments ---

def test_simulate_basic():
    exps = _make_experiments(10)
    proposed = [
        {"learning_rate": 0.05, "max_depth": 6},
        {"learning_rate": 0.15, "max_depth": 10},
        {"learning_rate": 0.01, "max_depth": 3},
    ]
    result = simulate_experiments(proposed, exps, "accuracy", top_k=2)
    assert result["total_proposed"] == 3
    assert result["run_count"] <= 2
    assert "predictions" in result
    assert result["predictions"][0]["rank"] == 1

def test_simulate_empty_configs():
    result = simulate_experiments([], [], "accuracy")
    assert "error" in result

def test_simulate_insufficient_history():
    result = simulate_experiments([{"lr": 0.1}], _make_experiments(2), "accuracy")
    assert "error" in result

def test_simulate_budget_savings():
    exps = _make_experiments(10, metric_base=0.85)
    proposed = [{"learning_rate": 0.01 * i, "max_depth": 3 + i} for i in range(20)]
    result = simulate_experiments(proposed, exps, "accuracy", top_k=5)
    assert result["budget_savings_pct"] > 0
    assert result["skip_count"] > 0

def test_simulate_lower_is_better():
    exps = []
    for i in range(10):
        exps.append({
            "config": {"hyperparams": {"learning_rate": 0.01 * (i + 1)}},
            "metrics": {"loss": 1.0 - 0.05 * i},
        })
    proposed = [{"learning_rate": 0.05}, {"learning_rate": 0.2}]
    result = simulate_experiments(proposed, exps, "loss", lower_is_better=True)
    assert "predictions" in result


# --- _summarize_config ---

def test_summarize_basic():
    s = _summarize_config({"lr": 0.1, "depth": 6})
    assert "lr=0.1" in s
    assert "depth=6" in s

def test_summarize_truncated():
    config = {f"param_{i}": i for i in range(10)}
    s = _summarize_config(config, max_items=3)
    assert "..." in s

def test_summarize_empty():
    assert "empty" in _summarize_config({})


# --- format_simulation_report ---

def test_format_report():
    report = {
        "current_best": 0.89,
        "total_proposed": 10,
        "run_count": 3,
        "skip_count": 7,
        "budget_savings_pct": 70.0,
        "predictions": [
            {"rank": 1, "config_summary": "lr=0.05", "predicted_metric": 0.91,
             "uncertainty": 0.003, "uncertainty_level": "LOW", "verdict": "RUN"},
        ],
        "generated_at": "2026-04-01T00:00:00Z",
    }
    text = format_simulation_report(report)
    assert "Experiment Simulation" in text
    assert "70.0%" in text

def test_format_error():
    report = {"error": "not enough data", "suggestion": "run more"}
    text = format_simulation_report(report)
    assert "not enough data" in text
    assert "run more" in text
