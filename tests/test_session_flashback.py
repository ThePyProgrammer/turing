"""Tests for session context restoration (session_flashback.py). Phase 24.2."""
from __future__ import annotations
import json
import pytest
from scripts.session_flashback import (
    find_current_best, filter_recent_experiments, load_annotations,
    format_flashback_report,
)

EXPS = [
    {"experiment_id": "exp-087", "status": "kept", "metrics": {"accuracy": 0.879},
     "config": {"model_type": "xgboost"}, "timestamp": "2026-03-25T10:00:00"},
    {"experiment_id": "exp-088", "status": "kept", "metrics": {"accuracy": 0.881},
     "config": {"model_type": "xgboost"}, "timestamp": "2026-03-25T11:00:00"},
    {"experiment_id": "exp-089", "status": "kept", "metrics": {"accuracy": 0.883},
     "config": {"model_type": "lightgbm"}, "timestamp": "2026-03-25T12:00:00"},
    {"experiment_id": "exp-090", "status": "discarded", "metrics": {"accuracy": 0.861},
     "config": {"model_type": "mlp"}, "timestamp": "2026-03-25T13:00:00"},
]

def test_find_current_best():
    best = find_current_best(EXPS, "accuracy", False)
    assert best["experiment_id"] == "exp-089"

def test_find_current_best_empty():
    assert find_current_best([], "accuracy", False) is None

def test_recent_experiments():
    recent = filter_recent_experiments(EXPS, last_n=2)
    assert len(recent) == 2

def test_recent_experiments_empty():
    assert filter_recent_experiments([], last_n=5) == []

def test_load_annotations_missing(tmp_path):
    result = load_annotations(str(tmp_path / "missing.yaml"))
    assert result == [] or result == {}

def test_load_annotations(tmp_path):
    import yaml
    ann = [{"experiment_id": "exp-042", "text": "fragile result", "date": "2026-04-01"}]
    with open(tmp_path / "annotations.yaml", "w") as f:
        yaml.dump(ann, f)
    result = load_annotations(str(tmp_path / "annotations.yaml"))
    assert len(result) >= 1

def test_format_report():
    report = {
        "generated_at": "2026-04-01T00:00:00",
        "current_best": {"experiment_id": "exp-089", "accuracy": 0.883, "model_type": "lightgbm"},
        "recent_experiments": [{"experiment_id": "exp-089", "status": "kept", "metrics": {"accuracy": 0.883}}],
        "pending_hypotheses": [{"id": "hyp-1", "description": "Try CatBoost"}],
        "annotations": [],
    }
    text = format_flashback_report(report)
    assert "exp-089" in text

def test_format_error():
    assert "ERROR" in format_flashback_report({"error": "fail"})
