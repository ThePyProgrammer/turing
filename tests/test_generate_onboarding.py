"""Tests for generate_onboarding.py. Phase 26.1."""
from __future__ import annotations
import pytest
from scripts.generate_onboarding import _family_summary, _find_best, format_onboarding_report

EXPS = [
    {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.85}, "config": {"model_type": "xgboost"}},
    {"experiment_id": "exp-002", "status": "discarded", "metrics": {"accuracy": 0.78}, "config": {"model_type": "xgboost"}},
    {"experiment_id": "exp-003", "status": "kept", "metrics": {"accuracy": 0.90}, "config": {"model_type": "lightgbm"}},
]

def test_family_summary():
    result = _family_summary(EXPS[:2], "accuracy", False)
    assert result["total"] == 2
    assert result["kept"] == 1

def test_find_best():
    best = _find_best(EXPS, "accuracy", False)
    assert best["experiment_id"] == "exp-003"

def test_find_best_empty():
    assert _find_best([], "accuracy", False) is None

def test_format_report_import():
    from scripts.generate_onboarding import format_onboarding_report
    assert callable(format_onboarding_report)
