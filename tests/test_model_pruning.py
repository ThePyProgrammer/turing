"""Tests for weight pruning (model_pruning.py). Phase 23.1."""

from __future__ import annotations
import pytest
from scripts.model_pruning import (
    compute_pruning_plan, estimate_size_reduction, estimate_speedup,
    find_knee_point, format_pruning_report, plan_sparsity_sweep, analyze_pruning,
)

def test_sparsity_sweep_default():
    levels = plan_sparsity_sweep()
    assert len(levels) == 5
    assert levels[0]["sparsity"] == 0.0

def test_sparsity_sweep_custom():
    levels = plan_sparsity_sweep([0.5, 0.9])
    assert len(levels) == 2

def test_plan_tree():
    plan = compute_pruning_plan("xgboost", {"n_estimators": 100}, "magnitude", 0.5)
    assert plan["config_changes"]["n_estimators"] == 50
    assert plan["strategy"] == "reduce_estimators"

def test_plan_magnitude():
    plan = compute_pruning_plan("pytorch", {}, "magnitude", 0.75)
    assert plan["strategy"] == "zero_small_weights"

def test_plan_structured():
    plan = compute_pruning_plan("mlp", {}, "structured", 0.5)
    assert plan["strategy"] == "remove_neurons"

def test_plan_lottery():
    plan = compute_pruning_plan("nn", {}, "lottery", 0.9)
    assert plan["strategy"] == "iterative_magnitude_with_rewind"

def test_knee_point_basic():
    results = [
        {"sparsity": 0.0, "accuracy": 0.872},
        {"sparsity": 0.5, "accuracy": 0.871},
        {"sparsity": 0.75, "accuracy": 0.868},
        {"sparsity": 0.9, "accuracy": 0.859},
        {"sparsity": 0.95, "accuracy": 0.831},
    ]
    knee = find_knee_point(results)
    assert knee is not None
    assert knee["sparsity"] == 0.9  # Biggest drop is 0.9→0.95

def test_knee_point_too_few():
    assert find_knee_point([{"sparsity": 0.0, "accuracy": 0.85}]) is None

def test_speedup():
    assert estimate_speedup(0.0) == 1.0
    assert estimate_speedup(0.5) > 1.0
    assert estimate_speedup(0.9) > estimate_speedup(0.5)

def test_size_reduction():
    assert estimate_size_reduction(0.75) == 75.0
    assert estimate_size_reduction(0.0) == 0.0

def test_analyze_with_results():
    results = [
        {"sparsity": 0.0, "accuracy": 0.872},
        {"sparsity": 0.5, "accuracy": 0.871},
        {"sparsity": 0.75, "accuracy": 0.868},
    ]
    report = analyze_pruning(sweep_results=results, exp_id="exp-042")
    assert "sweep_results" in report
    assert report["sweep_results"][0].get("speedup") is not None

def test_format_results():
    report = {
        "experiment_id": "exp-042", "method": "magnitude", "primary_metric": "accuracy",
        "sweep_results": [
            {"sparsity": 0.0, "accuracy": 0.872, "speedup": 1.0, "size_reduction_pct": 0},
            {"sparsity": 0.75, "accuracy": 0.868, "speedup": 1.8, "size_reduction_pct": 75},
        ],
        "knee_point": {"sparsity": 0.9, "drop_at_knee": 0.013},
        "recommended": {"sparsity": 0.75, "speedup": 1.8},
    }
    text = format_pruning_report(report)
    assert "Pruning" in text
    assert "75%" in text

def test_format_plan():
    report = {"action": "plan", "model_type": "xgboost", "method": "magnitude",
              "plans": [{"sparsity": 0.5, "strategy": "reduce_estimators"}]}
    text = format_pruning_report(report)
    assert "Plan" in text

def test_format_error():
    assert "ERROR" in format_pruning_report({"error": "fail"})
