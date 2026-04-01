"""Tests for model merging (model_merger.py). Phase 23.3."""

from __future__ import annotations
import pytest
from scripts.model_merger import (
    check_compatibility, compare_merge_methods, format_merge_report,
    plan_dare_merge, plan_greedy_merge, plan_ties_merge, plan_uniform_merge,
)

EXPS = [
    {"experiment_id": "exp-042", "config": {"model_type": "xgboost"}, "metrics": {"accuracy": 0.872}},
    {"experiment_id": "exp-053", "config": {"model_type": "xgboost"}, "metrics": {"accuracy": 0.878}},
    {"experiment_id": "exp-067", "config": {"model_type": "xgboost"}, "metrics": {"accuracy": 0.869}},
]

def test_compatibility_same():
    result = check_compatibility(EXPS)
    assert result["compatible"] is True

def test_compatibility_mixed():
    mixed = [{"config": {"model_type": "xgboost"}}, {"config": {"model_type": "lightgbm"}}]
    result = check_compatibility(mixed)
    assert result["compatible"] is False

def test_uniform_merge():
    plan = plan_uniform_merge(EXPS, "accuracy")
    assert plan["method"] == "uniform"
    assert len(plan["weights"]) == 3
    assert abs(sum(plan["weights"]) - 1.0) < 0.01

def test_greedy_merge():
    plan = plan_greedy_merge(EXPS, "accuracy")
    assert plan["method"] == "greedy"
    assert plan["included"][0] == "exp-053"  # Best first

def test_ties_merge():
    plan = plan_ties_merge(EXPS)
    assert plan["method"] == "ties"
    assert len(plan["steps"]) == 4

def test_dare_merge():
    plan = plan_dare_merge(EXPS)
    assert plan["method"] == "dare"
    assert plan["drop_rate"] == 0.5

def test_compare_with_results():
    results = {"uniform": {"metric_value": 0.881}, "greedy": {"metric_value": 0.883}}
    comparison = compare_merge_methods(results, EXPS, "accuracy")
    assert comparison["best_method"] == "greedy"
    assert comparison["improvement"] > 0

def test_compare_no_improvement():
    results = {"uniform": {"metric_value": 0.870}}
    comparison = compare_merge_methods(results, EXPS, "accuracy")
    assert comparison["best_method"] == "best_single"  # Single model still best

def test_compare_no_experiments():
    result = compare_merge_methods(experiments=[])
    assert "error" in result

def test_format_basic():
    report = {
        "generated_at": "2026-01-01T00:00:00", "primary_metric": "accuracy",
        "compatibility": {"compatible": True, "reason": "Same arch"},
        "base_models": [{"exp_id": "exp-042", "model_type": "xgboost", "accuracy": 0.872}],
        "plans": {"uniform": {"description": "Simple average"}, "greedy": {"description": "Iterative add"}},
        "comparison": {
            "results": [{"method": "best_single", "metric_value": 0.878, "delta": 0.0},
                       {"method": "greedy", "metric_value": 0.883, "delta": 0.005}],
            "best_method": "greedy", "improvement": 0.005,
        },
    }
    text = format_merge_report(report)
    assert "Model Merge" in text
    assert "greedy" in text
    assert "+0.005" in text or "BEST" in text

def test_format_error():
    assert "ERROR" in format_merge_report({"error": "fail"})
