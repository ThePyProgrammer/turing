"""Edge case tests for cross-project knowledge transfer (knowledge_transfer.py).

Phase 19.1: Covers no prior projects, identical projects, missing configs,
empty logs, self-reference, and boundary conditions.
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
    scan_for_projects,
    _extract_insights,
)


# --- compute_similarity edge cases ---


def test_similarity_same_task_no_features():
    """Same task type with no feature info should still match."""
    sig_a = {"dataset": {"task_type": "classification"}}
    sig_b = {"dataset": {"task_type": "classification"}}
    sim = compute_similarity(sig_a, sig_b)
    assert sim > 0.5


def test_similarity_none_values():
    """None sample sizes should not crash."""
    sig_a = {"dataset": {"task_type": "classification", "n_samples": None}}
    sig_b = {"dataset": {"task_type": "classification", "n_samples": 1000}}
    sim = compute_similarity(sig_a, sig_b)
    assert sim >= 0


def test_similarity_zero_samples():
    """Zero samples should not crash (log(0) guard)."""
    sig_a = {"dataset": {"task_type": "classification", "n_samples": 0}}
    sig_b = {"dataset": {"task_type": "classification", "n_samples": 100}}
    sim = compute_similarity(sig_a, sig_b)
    assert sim >= 0


def test_similarity_class_balance_match():
    """Matching class balance should boost similarity."""
    sig_a = {"dataset": {"task_type": "classification", "class_balance": "imbalanced"}}
    sig_b = {"dataset": {"task_type": "classification", "class_balance": "imbalanced"}}
    sim = compute_similarity(sig_a, sig_b)
    assert sim > 0.7


# --- extract_project_signature edge cases ---


def test_signature_lower_is_better(tmp_path):
    """Should find best experiment when lower is better."""
    import yaml
    config = {"evaluation": {"primary_metric": "loss", "lower_is_better": True}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    log = tmp_path / "experiments"
    log.mkdir()
    with open(log / "log.jsonl", "w") as f:
        f.write(json.dumps({"experiment_id": "exp-001", "status": "kept", "config": {"model_type": "a"}, "metrics": {"loss": 0.5}}) + "\n")
        f.write(json.dumps({"experiment_id": "exp-002", "status": "kept", "config": {"model_type": "b"}, "metrics": {"loss": 0.2}}) + "\n")
    sig = extract_project_signature(str(tmp_path / "config.yaml"), str(log / "log.jsonl"))
    assert sig["best_experiment"]["experiment_id"] == "exp-002"


def test_signature_no_kept(tmp_path):
    """All discarded should produce no best experiment."""
    import yaml
    (tmp_path / "config.yaml").write_text(yaml.dump({}))
    log = tmp_path / "experiments"
    log.mkdir()
    with open(log / "log.jsonl", "w") as f:
        f.write(json.dumps({"experiment_id": "exp-001", "status": "discarded", "config": {}, "metrics": {}}) + "\n")
    sig = extract_project_signature(str(tmp_path / "config.yaml"), str(log / "log.jsonl"))
    assert sig["best_experiment"] is None


# --- _extract_insights edge cases ---


def test_insights_empty():
    """No stats should produce minimal insights."""
    insights = _extract_insights([], {}, "accuracy")
    assert isinstance(insights, list)


def test_insights_single_model():
    """Single model family with 1 experiment should not report rates."""
    stats = {"xgboost": {"kept": 1, "discarded": 0, "total": 1}}
    insights = _extract_insights([], stats, "accuracy")
    # total < 2, so no rate insights
    assert not any("rate" in i.lower() for i in insights)


# --- scan_for_projects edge cases ---


def test_scan_nonexistent_roots():
    """Non-existent roots should not crash."""
    projects = scan_for_projects(["/nonexistent/path"], max_depth=1)
    assert projects == []


def test_scan_empty_dir(tmp_path):
    """Empty directory should find no projects."""
    projects = scan_for_projects([str(tmp_path)], max_depth=2)
    assert projects == []


def test_scan_finds_project(tmp_path):
    """Should find a valid project."""
    proj = tmp_path / "my_project"
    proj.mkdir()
    (proj / "config.yaml").write_text("metric: accuracy")
    (proj / "experiments").mkdir()
    (proj / "experiments" / "log.jsonl").write_text("")
    projects = scan_for_projects([str(tmp_path)], max_depth=2)
    assert len(projects) == 1


def test_scan_ignores_dotdirs(tmp_path):
    """Should skip hidden directories."""
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "config.yaml").write_text("x: 1")
    (hidden / "experiments").mkdir()
    (hidden / "experiments" / "log.jsonl").write_text("")
    projects = scan_for_projects([str(tmp_path)], max_depth=2)
    assert len(projects) == 0


# --- generate_recommendations edge cases ---


def test_recommendations_no_winner():
    """Project with no best experiment should still recommend."""
    similar = [{"path": "/proj/a", "similarity": 0.7, "signature": {"insights": ["explored 20 models"], "total_experiments": 20}}]
    recs = generate_recommendations({}, similar)
    assert len(recs) == 1
    assert "hypothesis" not in recs[0] or recs[0].get("hypothesis") is None


def test_recommendations_with_hyperparams():
    """Should include key hyperparams in hypothesis."""
    similar = [{
        "path": "/proj/a",
        "similarity": 0.8,
        "signature": {
            "best_experiment": {
                "model_type": "xgboost",
                "primary_metric": "accuracy",
                "metric_value": 0.90,
                "hyperparams": {"max_depth": 8, "n_estimators": 500, "learning_rate": 0.05},
            },
            "insights": [],
            "total_experiments": 15,
        },
    }]
    recs = generate_recommendations({}, similar)
    assert "max_depth=8" in recs[0]["hypothesis"]
    assert "n_estimators=500" in recs[0]["hypothesis"]


# --- format_transfer_report edge cases ---


def test_format_report_auto_queued():
    """Should show auto-queued hypotheses."""
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "similar_projects_found": 1,
        "recommendations": [{"project_path": "/p/a", "similarity": 0.8, "task_type": "c", "total_experiments": 10, "insights": [], "hypothesis": "Try xgboost"}],
        "auto_queued": ["Try xgboost (transferred from a)"],
    }
    text = format_transfer_report(report)
    assert "Auto-Queued" in text
