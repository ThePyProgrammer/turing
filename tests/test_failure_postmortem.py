"""Tests for failure postmortem (failure_postmortem.py).

Phase 29.1: Verifies streak detection, diagnosis categories, recommendations, reporting.
"""

from __future__ import annotations

import pytest

from scripts.failure_postmortem import (
    detect_failure_streak,
    diagnose_search_space_exhaustion,
    diagnose_systematic_config_error,
    diagnose_data_issue,
    diagnose_metric_ceiling,
    diagnose_noise_floor,
    run_postmortem,
    format_postmortem_report,
    _generate_recommendations,
)


def _make_streak(n, base_metric=0.85, config=None):
    """Generate N experiments that don't improve."""
    exps = [{"experiment_id": f"exp-{i:03d}",
             "metrics": {"accuracy": base_metric + 0.001 * (i % 3 - 1)},
             "config": config or {"model_type": "xgboost", "hyperparams": {"lr": 0.1, "depth": 6}},
             "status": "kept"} for i in range(n)]
    return exps


def _make_improving(n):
    """Generate N experiments that steadily improve."""
    return [{"experiment_id": f"exp-{i:03d}",
             "metrics": {"accuracy": 0.80 + 0.01 * i},
             "config": {"model_type": "xgboost", "hyperparams": {"lr": 0.1}},
             "status": "kept"} for i in range(n)]


# --- detect_failure_streak ---

def test_streak_basic():
    exps = _make_improving(5) + _make_streak(5, base_metric=0.84)
    result = detect_failure_streak(exps, "accuracy")
    assert result["streak_length"] >= 2  # Some stagnant exps detected

def test_streak_all_improving():
    result = detect_failure_streak(_make_improving(5), "accuracy")
    assert result["streak_length"] == 0

def test_streak_empty():
    result = detect_failure_streak([], "accuracy")
    assert result["streak_length"] == 0

def test_streak_single():
    result = detect_failure_streak([{"metrics": {"accuracy": 0.85}, "status": "kept"}], "accuracy")
    assert result["streak_length"] == 0


# --- diagnose_search_space_exhaustion ---

def test_exhaustion_detected():
    # All experiments with nearly identical configs
    exps = _make_streak(8, config={"hyperparams": {"lr": 0.1, "depth": 6}})
    result = diagnose_search_space_exhaustion(exps, "accuracy")
    assert result["score"] > 0

def test_exhaustion_diverse_configs():
    """Diverse configs but identical metrics still triggers metric clustering."""
    exps = [{"metrics": {"accuracy": 0.80 + 0.01 * i}, "config": {"hyperparams": {"lr": 0.01 * i, "depth": i}}} for i in range(1, 9)]
    result = diagnose_search_space_exhaustion(exps, "accuracy")
    assert result["score"] < 0.5  # Different metrics + different configs = low score

def test_exhaustion_too_few():
    result = diagnose_search_space_exhaustion(_make_streak(2), "accuracy")
    assert result["score"] == 0


# --- diagnose_systematic_config_error ---

def test_config_error_detected():
    exps = _make_streak(5, config={"hyperparams": {"lr": 0.5, "depth": 20}})
    result = diagnose_systematic_config_error(exps, "accuracy", 0.90)
    assert result["score"] > 0
    assert len(result["common_params"]) > 0

def test_config_error_varied():
    exps = [{"metrics": {"accuracy": 0.85}, "config": {"hyperparams": {"lr": 0.01 * i}}} for i in range(1, 6)]
    result = diagnose_systematic_config_error(exps, "accuracy", 0.90)
    assert result["score"] < 0.8


# --- diagnose_data_issue ---

def test_data_issue_detected():
    exps = [
        {"metrics": {"accuracy": 0.851}, "config": {"model_type": "xgboost"}},
        {"metrics": {"accuracy": 0.849}, "config": {"model_type": "lightgbm"}},
        {"metrics": {"accuracy": 0.852}, "config": {"model_type": "random_forest"}},
        {"metrics": {"accuracy": 0.850}, "config": {"model_type": "mlp"}},
    ]
    result = diagnose_data_issue(exps, "accuracy")
    assert result["score"] > 0

def test_data_issue_single_model():
    exps = _make_streak(5)  # All same model type
    result = diagnose_data_issue(exps, "accuracy")
    assert result["score"] == 0


# --- diagnose_metric_ceiling ---

def test_ceiling_detected():
    exps = _make_streak(5, base_metric=0.97)
    result = diagnose_metric_ceiling(exps, "accuracy", 0.97)
    assert result["score"] > 0

def test_ceiling_low_metric():
    result = diagnose_metric_ceiling(_make_streak(5, 0.70), "accuracy", 0.70)
    assert result["score"] < 0.5


# --- diagnose_noise_floor ---

def test_noise_with_seed_data(tmp_path):
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "exp-001-seeds.yaml").write_text("std: 0.005\n")

    exps = _make_streak(5, base_metric=0.85)
    result = diagnose_noise_floor(exps, "accuracy", str(seed_dir))
    assert result["seed_variance"] == 0.005

def test_noise_no_seed_data():
    result = diagnose_noise_floor(_make_streak(5), "accuracy", "/nonexistent")
    assert result["seed_variance"] is None


# --- run_postmortem ---

def test_postmortem_no_experiments(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text("")
    result = run_postmortem(config_path=str(config), log_path=str(log))
    assert "error" in result

def test_postmortem_no_streak(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    lines = [f'{{"experiment_id": "exp-{i:03d}", "status": "kept", "metrics": {{"accuracy": {0.80 + 0.01 * i}}}, "config": {{"hyperparams": {{"lr": 0.1}}}}}}\n' for i in range(5)]
    log.write_text("".join(lines))
    result = run_postmortem(config_path=str(config), log_path=str(log))
    assert "message" in result

def test_postmortem_with_streak(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    # 3 improving + 6 stagnant
    lines = []
    for i in range(3):
        lines.append(f'{{"experiment_id": "exp-{i:03d}", "status": "kept", "metrics": {{"accuracy": {0.80 + 0.02 * i}}}, "config": {{"model_type": "xgboost", "hyperparams": {{"lr": 0.1, "depth": 6}}}}}}\n')
    for i in range(3, 9):
        lines.append(f'{{"experiment_id": "exp-{i:03d}", "status": "kept", "metrics": {{"accuracy": 0.84}}, "config": {{"model_type": "xgboost", "hyperparams": {{"lr": 0.1, "depth": 6}}}}}}\n')
    log.write_text("".join(lines))
    result = run_postmortem(config_path=str(config), log_path=str(log))
    assert result["streak_length"] >= 4
    assert result["primary_diagnosis"] in ["search_space_exhaustion", "systematic_config_error", "metric_ceiling", "noise_floor", "data_issue"]
    assert len(result["recommendations"]) > 0


# --- _generate_recommendations ---

def test_recs_exhaustion():
    recs = _generate_recommendations("search_space_exhaustion", {}, 5)
    assert len(recs) >= 2
    assert any("feature" in r.lower() for r in recs)

def test_recs_data_issue():
    recs = _generate_recommendations("data_issue", {}, 5)
    assert any("leak" in r.lower() or "sanity" in r.lower() for r in recs)


# --- format_postmortem_report ---

def test_format_error():
    assert "ERROR" in format_postmortem_report({"error": "no experiments"})

def test_format_no_streak():
    text = format_postmortem_report({"message": "all good", "best_metric": 0.90})
    assert "all good" in text

def test_format_full():
    report = {
        "streak_length": 8,
        "primary_diagnosis": "search_space_exhaustion",
        "diagnosis_score": 0.7,
        "diagnosis_evidence": ["Config variance LOW"],
        "all_diagnoses": {"search_space_exhaustion": {"score": 0.7}, "data_issue": {"score": 0.1}},
        "recommendations": ["Switch to feature engineering"],
        "generated_at": "2026-04-01",
    }
    text = format_postmortem_report(report)
    assert "SEARCH SPACE EXHAUSTION" in text
    assert "70%" in text
    assert "Switch to feature" in text
