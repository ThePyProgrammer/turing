"""Edge case tests for experiment outcome simulator (experiment_simulator.py).

Phase 27.3: No history, single experiment, identical configs, extreme values.
"""

from __future__ import annotations

import pytest
import yaml

from scripts.experiment_simulator import (
    extract_config_features,
    experiment_to_features,
    build_surrogate,
    predict_with_surrogate,
    simulate_experiments,
    save_simulation_report,
    format_simulation_report,
    _config_distance,
    _summarize_config,
    MIN_HISTORY_FOR_SURROGATE,
)


def _make_experiments(n, metric_base=0.8):
    exps = []
    for i in range(n):
        exps.append({
            "config": {"hyperparams": {"learning_rate": 0.01 * (i + 1), "max_depth": 3 + i}},
            "metrics": {"accuracy": metric_base + 0.01 * i},
        })
    return exps


# --- Config feature edge cases ---

def test_extract_deeply_nested():
    config = {"a": {"b": {"c": {"d": 42.0}}}}
    features = extract_config_features(config)
    assert any(42.0 == v for v in features.values())

def test_extract_mixed_types():
    config = {"lr": 0.1, "name": "xgb", "layers": [1, 2, 3], "nested": {"n": 5}}
    features = extract_config_features(config)
    assert "lr" in features
    assert "nested.n" in features
    # lists are skipped
    assert not any("layers" in k for k in features)

def test_experiment_features_nested_config():
    exp = {"config": {"model": {"hyperparams": {"lr": 0.01, "depth": 3}}}}
    features = experiment_to_features(exp)
    assert "lr" in features


# --- Surrogate edge cases ---

def test_surrogate_all_same_metric():
    """All experiments have identical metrics."""
    exps = [
        {"config": {"hyperparams": {"lr": 0.01 * (i + 1)}}, "metrics": {"accuracy": 0.85}}
        for i in range(MIN_HISTORY_FOR_SURROGATE)
    ]
    surrogate = build_surrogate(exps, "accuracy")
    assert surrogate["status"] == "ready"

def test_surrogate_no_features():
    """Experiments with metrics but no extractable hyperparameters."""
    exps = [{"metrics": {"accuracy": 0.85}} for _ in range(10)]
    surrogate = build_surrogate(exps, "accuracy")
    assert surrogate["status"] == "insufficient"


# --- Prediction edge cases ---

def test_predict_empty_features():
    exps = _make_experiments(10)
    surrogate = build_surrogate(exps, "accuracy")
    pred = predict_with_surrogate(surrogate, {})
    # No overlap → infinite distance
    assert pred.get("predicted") is not None or pred.get("error") is not None

def test_predict_novel_config():
    """Config very far from training distribution should have high novelty."""
    exps = _make_experiments(10)
    surrogate = build_surrogate(exps, "accuracy")
    novel_config = {"learning_rate": 999.0, "max_depth": 999.0}
    pred = predict_with_surrogate(surrogate, novel_config)
    assert pred["novelty_score"] > 0

def test_predict_k_larger_than_data():
    exps = _make_experiments(MIN_HISTORY_FOR_SURROGATE)
    surrogate = build_surrogate(exps, "accuracy")
    pred = predict_with_surrogate(surrogate, {"learning_rate": 0.05}, k=100)
    # Should use all available points
    assert pred["n_neighbors"] <= MIN_HISTORY_FOR_SURROGATE


# --- Distance edge cases ---

def test_distance_zero_values():
    a = {"lr": 0.0}
    b = {"lr": 0.0}
    dist = _config_distance(a, b, ["lr"])
    assert dist == 0.0

def test_distance_negative_values():
    a = {"temp": -5.0}
    b = {"temp": 5.0}
    dist = _config_distance(a, b, ["temp"])
    assert dist > 0

def test_distance_single_feature():
    a = {"lr": 0.01}
    b = {"lr": 0.1}
    dist = _config_distance(a, b, ["lr"])
    assert 0 < dist < float("inf")


# --- Simulation edge cases ---

def test_simulate_single_config():
    exps = _make_experiments(10)
    result = simulate_experiments([{"learning_rate": 0.05}], exps, "accuracy")
    assert result["total_proposed"] == 1

def test_simulate_all_identical():
    """All proposed configs are identical."""
    exps = _make_experiments(10)
    proposed = [{"learning_rate": 0.05}] * 5
    result = simulate_experiments(proposed, exps, "accuracy")
    metrics = [p["predicted_metric"] for p in result["predictions"]]
    assert len(set(metrics)) == 1  # all same prediction

def test_simulate_top_k_zero():
    exps = _make_experiments(10)
    proposed = [{"learning_rate": 0.05}]
    result = simulate_experiments(proposed, exps, "accuracy", top_k=0)
    assert result["run_count"] == 0
    assert result["budget_savings_pct"] == 100.0

def test_simulate_threshold_high():
    """Very high threshold → everything is SKIP."""
    exps = _make_experiments(10)
    proposed = [{"learning_rate": 0.05}]
    result = simulate_experiments(proposed, exps, "accuracy", improvement_threshold=10.0)
    assert all(p["verdict"] == "SKIP" for p in result["predictions"])


# --- Save report ---

def test_save_report(tmp_path):
    report = {
        "current_best": 0.89,
        "total_proposed": 5,
        "predictions": [],
        "generated_at": "2026-04-01",
    }
    path = save_simulation_report(report, str(tmp_path / "sims"))
    assert path.exists()
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data["current_best"] == 0.89


# --- Format edge cases ---

def test_format_empty_predictions():
    report = {
        "current_best": 0.85,
        "total_proposed": 0,
        "run_count": 0,
        "skip_count": 0,
        "budget_savings_pct": 0,
        "predictions": [],
        "generated_at": "2026-04-01",
    }
    text = format_simulation_report(report)
    assert "Experiment Simulation" in text

def test_format_many_predictions():
    preds = []
    for i in range(20):
        preds.append({
            "rank": i + 1,
            "config_summary": f"lr={0.01 * i}",
            "predicted_metric": 0.85 + 0.001 * i,
            "uncertainty": 0.005,
            "uncertainty_level": "MED",
            "verdict": "RUN" if i < 5 else "SKIP",
        })
    report = {
        "current_best": 0.85,
        "total_proposed": 20,
        "run_count": 5,
        "skip_count": 15,
        "budget_savings_pct": 75.0,
        "predictions": preds,
        "generated_at": "2026-04-01",
    }
    text = format_simulation_report(report)
    assert "75.0%" in text
    assert "RUN" in text
    assert "SKIP" in text


# --- _summarize_config edge cases ---

def test_summarize_nested_config():
    config = {"model": {"lr": 0.1}}
    s = _summarize_config(config)
    assert "0.1" in s

def test_summarize_single_param():
    s = _summarize_config({"depth": 6})
    assert "depth=6" in s
