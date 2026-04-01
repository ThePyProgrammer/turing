"""Tests for experiment search (experiment_search.py). Phase 24.5."""
from __future__ import annotations
import pytest
from scripts.experiment_search import (
    parse_query, apply_filters, rank_by_keywords, format_search_report, _compare,
)

EXPS = [
    {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.85},
     "config": {"model_type": "xgboost"}, "description": "XGBoost baseline"},
    {"experiment_id": "exp-002", "status": "discarded", "metrics": {"accuracy": 0.78},
     "config": {"model_type": "lightgbm"}, "description": "LightGBM test"},
    {"experiment_id": "exp-003", "status": "kept", "metrics": {"accuracy": 0.90},
     "config": {"model_type": "xgboost"}, "description": "XGBoost tuned"},
]

def test_parse_basic():
    result = parse_query("xgboost high accuracy")
    assert "keywords" in result
    assert len(result["keywords"]) > 0

def test_parse_with_filter():
    result = parse_query("xgboost accuracy>0.85")
    assert "filters" in result or "metric_filters" in result

def test_apply_filters_status():
    filtered = apply_filters(EXPS, {"status": "kept"})
    assert all(e["status"] == "kept" for e in filtered)

def test_apply_filters_empty():
    filtered = apply_filters(EXPS, {})
    assert len(filtered) == len(EXPS)

def test_rank_by_keywords():
    ranked = rank_by_keywords(EXPS, ["xgboost"])
    assert len(ranked) > 0
    # XGBoost experiments should rank higher
    top = ranked[0]
    assert "xgboost" in str(top[0].get("config", {})).lower() or "xgboost" in str(top[0].get("description", "")).lower()

def test_rank_empty():
    assert rank_by_keywords([], ["test"]) == []

def test_compare():
    assert _compare(0.85, ">", 0.80) is True
    assert _compare(0.85, "<", 0.80) is False
    assert _compare(0.85, ">=", 0.85) is True

def test_format_search():
    results = [(EXPS[0], 1.0), (EXPS[2], 0.8)]
    text = format_search_report(results, {}, "xgboost")
    assert "exp-001" in text or "exp-003" in text
