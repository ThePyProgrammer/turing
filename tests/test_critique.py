"""Tests for hypothesis critique engine (critique_hypothesis.py).

Phase 8.1: Verifies novelty scoring, feasibility scoring, impact
scoring, and overall critique with verdict classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.critique_hypothesis import (
    critique_hypothesis,
    format_critique,
    score_expected_impact,
    score_feasibility,
    score_novelty,
)


def _exp(exp_id, desc, status="kept", accuracy=0.85, exp_type="hyperparameter"):
    return {
        "experiment_id": exp_id,
        "description": desc,
        "status": status,
        "config": {"model_type": "xgboost", "experiment_type": exp_type},
        "metrics": {"accuracy": accuracy},
    }


# --- score_novelty ---


def test_novelty_no_history():
    result = score_novelty("try something new", [])
    assert result["score"] >= 8


def test_novelty_near_duplicate():
    exps = [_exp("exp-001", "increase max_depth to 8", "discarded")]
    result = score_novelty("increase max_depth to 8 for better accuracy", exps)
    assert result["score"] <= 4
    assert len(result["similar"]) > 0


def test_novelty_duplicate_with_guard_result():
    result = score_novelty("whatever", [], novelty_result={"classification": "duplicate_run"})
    assert result["score"] == 0


def test_novelty_repeat_failure_with_guard():
    result = score_novelty("whatever", [], novelty_result={"classification": "repeat_failure"})
    assert result["score"] == 1


def test_novelty_incremental_with_guard():
    result = score_novelty("whatever", [], novelty_result={"classification": "incremental_followup"})
    assert result["score"] == 5


def test_novelty_builds_on_success():
    exps = [_exp("exp-001", "switch to LightGBM with default params", "kept")]
    result = score_novelty("switch to LightGBM with tuned params", exps)
    assert result["score"] >= 5
    assert result["score"] <= 8


# --- score_feasibility ---


def test_feasibility_simple_change():
    result = score_feasibility("increase max_depth to 8", [], {"model": {"type": "xgboost"}})
    assert result["score"] >= 8
    assert len(result["concerns"]) == 0


def test_feasibility_exotic_model():
    result = score_feasibility("switch to a transformer architecture", [], {"model": {"type": "xgboost"}})
    assert result["score"] < 8
    assert len(result["concerns"]) > 0


def test_feasibility_gpu_requirement():
    result = score_feasibility("train on multi-gpu setup with CUDA", [], {"model": {"type": "xgboost"}})
    assert result["score"] < 8
    assert any("gpu" in c.lower() or "cuda" in c.lower() for c in result["concerns"])


def test_feasibility_external_data():
    result = score_feasibility("download pretrained embeddings from huggingface", [], {})
    assert len(result["concerns"]) >= 1


def test_feasibility_multiple_concerns():
    result = score_feasibility(
        "switch to a transformer with GPU training and download pretrained weights from huggingface",
        [], {"model": {"type": "xgboost"}},
    )
    assert result["score"] <= 5
    assert len(result["concerns"]) >= 2


# --- score_expected_impact ---


def test_impact_no_history():
    result = score_expected_impact("try something", [], {})
    assert result["score"] == 5  # neutral


def test_impact_successful_type():
    exps = [
        _exp("e1", "tune lr", "kept", 0.85, "hyperparameter"),
        _exp("e2", "tune depth", "kept", 0.87, "hyperparameter"),
        _exp("e3", "tune estimators", "discarded", 0.84, "hyperparameter"),
    ]
    result = score_expected_impact("change learning rate to 0.05", exps, {})
    assert result["score"] >= 7  # 2/3 kept = 67%


def test_impact_failed_type():
    exps = [
        _exp("e1", "switch to lightgbm", "discarded", 0.80, "architecture"),
        _exp("e2", "switch to catboost", "discarded", 0.81, "architecture"),
        _exp("e3", "switch to random forest", "discarded", 0.79, "architecture"),
    ]
    result = score_expected_impact("switch to a different model type", exps, {})
    assert result["score"] <= 4  # 0/3 kept = 0%


def test_impact_diminishing_returns():
    exps = [
        _exp("e1", "a", "kept", 0.850),
        _exp("e2", "b", "kept", 0.851),
        _exp("e3", "c", "kept", 0.8512),
        _exp("e4", "d", "kept", 0.8515),
    ]
    config = {"evaluation": {"primary_metric": "accuracy"}}
    result = score_expected_impact("another small tweak", exps, config)
    assert result["score"] <= 5


# --- critique_hypothesis (integration) ---


def test_critique_full_proceed(tmp_path):
    log_path = tmp_path / "log.jsonl"
    config_path = tmp_path / "config.yaml"
    log_path.touch()
    with open(config_path, "w") as f:
        yaml.dump({"model": {"type": "xgboost"}, "evaluation": {"primary_metric": "accuracy"}}, f)

    result = critique_hypothesis("increase n_estimators to 200", str(log_path), str(config_path))
    assert result["overall_score"] >= 0
    assert result["verdict"] in ("proceed", "modify", "reject")
    assert "novelty" in result
    assert "feasibility" in result
    assert "impact" in result


def test_critique_rejects_duplicate(tmp_path):
    log_path = tmp_path / "log.jsonl"
    config_path = tmp_path / "config.yaml"
    with open(log_path, "w") as f:
        f.write(json.dumps(_exp("e1", "increase depth to 8", "discarded")) + "\n")
        f.write(json.dumps(_exp("e2", "increase depth to 8", "discarded")) + "\n")
    with open(config_path, "w") as f:
        yaml.dump({"model": {"type": "xgboost"}}, f)

    result = critique_hypothesis(
        "increase depth to 8",
        str(log_path), str(config_path),
        novelty_result={"classification": "repeat_failure"},
    )
    assert result["overall_score"] < 5
    assert result["verdict"] in ("modify", "reject")


def test_critique_weighted_scoring():
    """Verify the weighting: novelty 30%, feasibility 30%, impact 40%."""
    # With known scores, verify the weighted average
    result = critique_hypothesis("test hypothesis", "/nonexistent/log.jsonl", "/nonexistent/config.yaml")
    n = result["novelty"]["score"]
    f = result["feasibility"]["score"]
    i = result["impact"]["score"]
    expected = round(n * 0.3 + f * 0.3 + i * 0.4, 1)
    assert result["overall_score"] == expected


# --- format_critique ---


def test_format_critique_basic():
    result = {
        "description": "try LightGBM",
        "overall_score": 7.5,
        "verdict": "proceed",
        "novelty": {"score": 8, "rationale": "Novel approach", "similar": []},
        "feasibility": {"score": 7, "rationale": "One concern", "concerns": ["may need install"]},
        "impact": {"score": 7, "rationale": "Unknown impact", "evidence": []},
    }
    output = format_critique(result)
    assert "7.5/10" in output
    assert "PROCEED" in output
    assert "may need install" in output


def test_format_critique_reject():
    result = {
        "description": "bad idea",
        "overall_score": 2.0,
        "verdict": "reject",
        "novelty": {"score": 1, "rationale": "Duplicate", "similar": []},
        "feasibility": {"score": 3, "rationale": "Hard", "concerns": []},
        "impact": {"score": 2, "rationale": "Low", "evidence": []},
    }
    output = format_critique(result)
    assert "REJECT" in output
    assert "different approach" in output.lower()
