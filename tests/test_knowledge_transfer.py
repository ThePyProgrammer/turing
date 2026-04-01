"""Tests for cross-project knowledge transfer (knowledge_transfer.py).

Phase 19.1: Verifies signature extraction, similarity matching,
recommendation generation, index management, and report formatting.
"""

from __future__ import annotations

import json
import os

import pytest

from scripts.knowledge_transfer import (
    compute_similarity,
    extract_project_signature,
    format_transfer_report,
    generate_recommendations,
    index_project,
    load_project_index,
    save_project_index,
    _extract_insights,
)


# --- extract_project_signature ---


def test_extract_signature_basic(tmp_path):
    """Should extract signature from config and log."""
    import yaml

    config = {"evaluation": {"primary_metric": "accuracy"}, "task": {"type": "classification"}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))

    log = tmp_path / "experiments"
    log.mkdir()
    with open(log / "log.jsonl", "w") as f:
        f.write(json.dumps({"experiment_id": "exp-001", "status": "kept", "config": {"model_type": "xgboost"}, "metrics": {"accuracy": 0.85}}) + "\n")

    sig = extract_project_signature(str(tmp_path / "config.yaml"), str(log / "log.jsonl"))
    assert sig["primary_metric"] == "accuracy"
    assert sig["total_experiments"] == 1
    assert sig["best_experiment"]["model_type"] == "xgboost"


def test_extract_signature_empty(tmp_path):
    """Empty project should produce signature with no best."""
    import yaml
    (tmp_path / "config.yaml").write_text(yaml.dump({}))
    (tmp_path / "log.jsonl").write_text("")

    sig = extract_project_signature(str(tmp_path / "config.yaml"), str(tmp_path / "log.jsonl"))
    assert sig["total_experiments"] == 0
    assert sig["best_experiment"] is None


# --- _extract_insights ---


def test_insights_best_family():
    """Should identify best model family."""
    stats = {"xgboost": {"kept": 8, "discarded": 2, "total": 10}, "rf": {"kept": 2, "discarded": 8, "total": 10}}
    insights = _extract_insights([], stats, "accuracy")
    assert any("xgboost" in i.lower() for i in insights)


def test_insights_few_experiments():
    """Few experiments should note limited exploration."""
    experiments = [{"experiment_id": f"exp-{i}"} for i in range(3)]
    insights = _extract_insights(experiments, {}, "accuracy")
    assert any("limited" in i.lower() for i in insights)


def test_insights_many_experiments():
    """Many experiments should note extensive search."""
    experiments = [{"experiment_id": f"exp-{i}"} for i in range(25)]
    insights = _extract_insights(experiments, {}, "accuracy")
    assert any("extensive" in i.lower() for i in insights)


# --- compute_similarity ---


def test_similarity_identical():
    """Identical signatures should have high similarity."""
    sig = {"dataset": {"task_type": "classification", "n_samples": 1000, "n_features": 50, "feature_types": "tabular"}}
    sim = compute_similarity(sig, sig)
    assert sim > 0.9


def test_similarity_different_task():
    """Different task types should have low similarity."""
    sig_a = {"dataset": {"task_type": "classification"}}
    sig_b = {"dataset": {"task_type": "regression"}}
    sim = compute_similarity(sig_a, sig_b)
    assert sim < 0.5


def test_similarity_same_task_different_size():
    """Same task but very different size should have moderate similarity."""
    sig_a = {"dataset": {"task_type": "classification", "n_samples": 100}}
    sig_b = {"dataset": {"task_type": "classification", "n_samples": 1000000}}
    sim = compute_similarity(sig_a, sig_b)
    assert 0.3 < sim < 0.9


def test_similarity_empty():
    """Empty signatures should return 0."""
    assert compute_similarity({}, {}) == 0.0


def test_similarity_partial():
    """Partial signatures should still compute."""
    sig_a = {"dataset": {"task_type": "classification"}}
    sig_b = {"dataset": {"task_type": "classification", "n_samples": 500}}
    sim = compute_similarity(sig_a, sig_b)
    assert sim > 0.5


# --- Project Index ---


def test_index_save_load(tmp_path):
    """Should roundtrip through save/load."""
    idx_path = str(tmp_path / "index.yaml")
    index = [{"path": "/proj/a", "signature": {"dataset": {"task_type": "classification"}}}]
    save_project_index(index, idx_path)
    loaded = load_project_index(idx_path)
    assert len(loaded) == 1
    assert loaded[0]["path"] == "/proj/a"


def test_index_missing_file(tmp_path):
    """Missing index should return empty list."""
    assert load_project_index(str(tmp_path / "missing.yaml")) == []


def test_index_project_upsert(tmp_path):
    """Indexing same path twice should update, not duplicate."""
    idx_path = str(tmp_path / "index.yaml")
    index_project("/proj/a", {"v": 1}, idx_path)
    index_project("/proj/a", {"v": 2}, idx_path)
    loaded = load_project_index(idx_path)
    assert len(loaded) == 1
    assert loaded[0]["signature"]["v"] == 2


def test_index_multiple_projects(tmp_path):
    """Should store multiple projects."""
    idx_path = str(tmp_path / "index.yaml")
    index_project("/proj/a", {"v": 1}, idx_path)
    index_project("/proj/b", {"v": 2}, idx_path)
    loaded = load_project_index(idx_path)
    assert len(loaded) == 2


# --- generate_recommendations ---


def test_recommendations_basic():
    """Should generate recommendations from similar projects."""
    current = {"dataset": {"task_type": "classification"}}
    similar = [
        {
            "path": "/proj/fraud",
            "similarity": 0.85,
            "signature": {
                "dataset": {"task_type": "classification"},
                "best_experiment": {"model_type": "lightgbm", "primary_metric": "accuracy", "metric_value": 0.92, "hyperparams": {"max_depth": 6}},
                "total_experiments": 30,
                "insights": ["lightgbm best"],
            },
        },
    ]
    recs = generate_recommendations(current, similar)
    assert len(recs) == 1
    assert recs[0]["similarity"] == 0.85
    assert "lightgbm" in recs[0]["hypothesis"]


def test_recommendations_empty():
    """No similar projects should produce empty list."""
    recs = generate_recommendations({}, [])
    assert recs == []


def test_recommendations_top_k():
    """Should respect top_k limit."""
    similar = [{"path": f"/p/{i}", "similarity": 0.9 - i * 0.1, "signature": {"insights": []}} for i in range(5)]
    recs = generate_recommendations({}, similar, top_k=2)
    assert len(recs) == 2


# --- format_transfer_report ---


def test_format_report_with_results():
    """Should produce readable markdown with recommendations."""
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "current_project": "/proj/current",
        "similar_projects_found": 1,
        "recommendations": [
            {
                "project_path": "/proj/fraud",
                "similarity": 0.85,
                "task_type": "classification",
                "total_experiments": 30,
                "winner": {"model_type": "lightgbm", "metric_name": "accuracy", "metric_value": 0.92},
                "hypothesis": "Try lightgbm with max_depth=6",
                "insights": ["lightgbm best"],
            },
        ],
    }
    text = format_transfer_report(report)
    assert "Knowledge Transfer" in text
    assert "fraud" in text
    assert "lightgbm" in text


def test_format_report_no_projects():
    """No projects should show helpful message."""
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "similar_projects_found": 0,
        "recommendations": [],
    }
    text = format_transfer_report(report)
    assert "No similar" in text


def test_format_report_error():
    text = format_transfer_report({"error": "No config found"})
    assert "ERROR" in text
