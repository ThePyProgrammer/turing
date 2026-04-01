"""Tests for experiment replay (experiment_replay.py). Phase 24.7."""
from __future__ import annotations
import pytest
from scripts.experiment_replay import (
    find_experiment, compare_metrics, format_replay_report,
)

EXPS = [
    {"experiment_id": "exp-023", "config": {"model_type": "xgboost", "hyperparams": {"max_depth": 6}},
     "metrics": {"accuracy": 0.834}, "git_commit": "abc123", "timestamp": "2026-03-01T00:00:00"},
]

def test_find_experiment():
    exp = find_experiment(EXPS, "exp-023")
    assert exp["experiment_id"] == "exp-023"

def test_find_missing():
    assert find_experiment(EXPS, "exp-999") is None

def test_find_empty():
    assert find_experiment([], "exp-001") is None

def test_compare_metrics():
    original = {"accuracy": 0.834, "f1": 0.81}
    replayed = {"accuracy": 0.856, "f1": 0.83}
    comparison = compare_metrics(original, replayed, "accuracy")
    assert "comparisons" in comparison or "accuracy" in comparison

def test_compare_empty():
    comparison = compare_metrics({}, {}, "accuracy")
    assert isinstance(comparison, dict)

def test_format_error():
    assert "ERROR" in format_replay_report({"error": "Not found"})
