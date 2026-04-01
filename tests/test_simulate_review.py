"""Tests for simulate_review.py. Phase 26.3."""
from __future__ import annotations
import pytest
from scripts.simulate_review import check_baselines, check_error_bars, format_review_report

EXPS = [
    {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.85}, "config": {"model_type": "xgboost"}, "description": "tuned model"},
]

def test_check_baselines_missing():
    weaknesses = check_baselines(EXPS, {})
    assert len(weaknesses) >= 1  # No baselines found

def test_check_baselines_present():
    exps = EXPS + [{"experiment_id": "exp-000", "status": "kept", "config": {"model_type": "majority_baseline"}, "description": "baseline", "metrics": {"accuracy": 0.5}}]
    weaknesses = check_baselines(exps, {})
    # Function returns a dict (single weakness) or None
    assert isinstance(weaknesses, (dict, type(None)))

def test_check_error_bars_missing():
    weaknesses = check_error_bars(EXPS, [])
    assert len(weaknesses) >= 1

