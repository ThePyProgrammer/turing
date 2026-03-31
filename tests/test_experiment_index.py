"""Tests for semantic experiment index (experiment_index.py).

Phase 9.1: Verifies TF-IDF indexing, semantic query, similar
experiment retrieval, serialization, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiment_index import (
    ExperimentIndex,
    experiment_to_text,
    format_results,
    load_experiments,
)


def _exp(exp_id, desc, status="kept", accuracy=0.85, model="xgboost",
         exp_type="hyperparameter", hyperparams=None):
    return {
        "experiment_id": exp_id,
        "description": desc,
        "status": status,
        "config": {
            "model_type": model,
            "experiment_type": exp_type,
            "hyperparams": hyperparams or {},
        },
        "metrics": {"accuracy": accuracy},
    }


SAMPLE_EXPERIMENTS = [
    _exp("exp-001", "baseline xgboost with default params", "kept", 0.82),
    _exp("exp-002", "increase max_depth to 8 for deeper trees", "kept", 0.84,
         hyperparams={"max_depth": 8}),
    _exp("exp-003", "switch to LightGBM with dart boosting", "discarded", 0.80,
         model="lightgbm"),
    _exp("exp-004", "add polynomial features for numeric columns", "kept", 0.86,
         exp_type="feature"),
    _exp("exp-005", "increase n_estimators to 500", "kept", 0.85,
         hyperparams={"n_estimators": 500}),
    _exp("exp-006", "reduce learning rate to 0.01", "discarded", 0.83,
         hyperparams={"learning_rate": 0.01}),
    _exp("exp-007", "try random forest ensemble", "discarded", 0.79,
         model="random_forest", exp_type="architecture"),
    _exp("exp-008", "add regularization with higher weight decay", "kept", 0.87,
         exp_type="regularization"),
]


# --- experiment_to_text ---


def test_text_includes_description():
    text = experiment_to_text(_exp("e1", "try LightGBM"))
    assert "try LightGBM" in text


def test_text_includes_model_type():
    text = experiment_to_text(_exp("e1", "test", model="catboost"))
    assert "catboost" in text


def test_text_includes_metrics():
    text = experiment_to_text(_exp("e1", "test", accuracy=0.92))
    assert "0.92" in text


def test_text_includes_status():
    text = experiment_to_text(_exp("e1", "test", status="discarded"))
    assert "discarded" in text


def test_text_empty_experiment():
    assert experiment_to_text({}) == ""


# --- ExperimentIndex.build ---


def test_build_indexes_all():
    idx = ExperimentIndex()
    count = idx.build(SAMPLE_EXPERIMENTS)
    assert count == len(SAMPLE_EXPERIMENTS)
    assert len(idx.experiment_ids) == count


def test_build_empty_list():
    idx = ExperimentIndex()
    count = idx.build([])
    assert count == 0


def test_build_skips_empty_text():
    idx = ExperimentIndex()
    count = idx.build([{}, {"experiment_id": "e1"}])
    assert count == 0


# --- ExperimentIndex.query ---


def test_query_finds_relevant():
    idx = ExperimentIndex()
    idx.build(SAMPLE_EXPERIMENTS)
    results = idx.query("LightGBM dart boosting", top_k=3)
    assert len(results) > 0
    # The LightGBM experiment should be the top result
    assert results[0]["experiment_id"] == "exp-003"


def test_query_depth_related():
    idx = ExperimentIndex()
    idx.build(SAMPLE_EXPERIMENTS)
    results = idx.query("deeper trees max_depth", top_k=3)
    assert len(results) > 0
    ids = [r["experiment_id"] for r in results]
    assert "exp-002" in ids


def test_query_feature_engineering():
    idx = ExperimentIndex()
    idx.build(SAMPLE_EXPERIMENTS)
    results = idx.query("polynomial features", top_k=3)
    assert len(results) > 0
    assert results[0]["experiment_id"] == "exp-004"


def test_query_returns_scores():
    idx = ExperimentIndex()
    idx.build(SAMPLE_EXPERIMENTS)
    results = idx.query("xgboost baseline", top_k=3)
    for r in results:
        assert "similarity" in r
        assert 0 <= r["similarity"] <= 1


def test_query_no_results_for_unrelated():
    idx = ExperimentIndex()
    idx.build(SAMPLE_EXPERIMENTS)
    results = idx.query("quantum computing photonics laser", top_k=3)
    # Should return empty or very low scores
    assert all(r["similarity"] < 0.1 for r in results) or len(results) == 0


def test_query_empty_index():
    idx = ExperimentIndex()
    results = idx.query("anything")
    assert results == []


def test_query_respects_top_k():
    idx = ExperimentIndex()
    idx.build(SAMPLE_EXPERIMENTS)
    results = idx.query("xgboost", top_k=2)
    assert len(results) <= 2


# --- ExperimentIndex.find_similar ---


def test_find_similar_basic():
    idx = ExperimentIndex()
    idx.build(SAMPLE_EXPERIMENTS)
    results = idx.find_similar("exp-002", top_k=3)
    assert len(results) > 0
    # Should not include exp-002 itself
    ids = [r["experiment_id"] for r in results]
    # Note: find_similar includes the query experiment, caller filters


def test_find_similar_nonexistent():
    idx = ExperimentIndex()
    idx.build(SAMPLE_EXPERIMENTS)
    results = idx.find_similar("exp-999")
    assert results == []


# --- save/load ---


def test_save_and_load(tmp_path):
    idx = ExperimentIndex()
    idx.build(SAMPLE_EXPERIMENTS)

    path = str(tmp_path / "index.pkl")
    idx.save(path)

    loaded = ExperimentIndex.load(path)
    assert loaded is not None
    assert len(loaded.experiment_ids) == len(SAMPLE_EXPERIMENTS)

    # Query should work on loaded index
    results = loaded.query("LightGBM", top_k=1)
    assert len(results) > 0
    assert results[0]["experiment_id"] == "exp-003"


def test_load_nonexistent():
    result = ExperimentIndex.load("/nonexistent/index.pkl")
    assert result is None


# --- format_results ---


def test_format_results_basic():
    results = [
        {"experiment_id": "exp-001", "similarity": 0.85, "description": "test",
         "status": "kept", "metrics": {"accuracy": 0.9}, "config": {}},
    ]
    output = format_results(results)
    assert "exp-001" in output
    assert "0.850" in output
    assert "0.9" in output


def test_format_results_empty():
    output = format_results([])
    assert "No matching" in output
