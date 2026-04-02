"""Edge case tests for failure postmortem (failure_postmortem.py).

Phase 29.1: No experiments, all kept, single experiment, missing metrics.
"""

from __future__ import annotations

import pytest
import yaml

from scripts.failure_postmortem import (
    detect_failure_streak,
    diagnose_search_space_exhaustion,
    diagnose_systematic_config_error,
    diagnose_data_issue,
    diagnose_metric_ceiling,
    diagnose_noise_floor,
    run_postmortem,
    save_postmortem_report,
    format_postmortem_report,
)


# --- Streak edge cases ---

def test_streak_no_metrics():
    exps = [{"experiment_id": "exp-001", "config": {}, "status": "kept"}] * 5
    result = detect_failure_streak(exps, "accuracy")
    assert result["best_metric"] is None

def test_streak_mixed_status():
    exps = [
        {"metrics": {"accuracy": 0.80}, "status": "kept"},
        {"metrics": {"accuracy": 0.75}, "status": "discarded"},
        {"metrics": {"accuracy": 0.82}, "status": "kept"},
    ]
    result = detect_failure_streak(exps, "accuracy")
    assert result["best_metric"] is not None

def test_streak_lower_is_better():
    exps = [
        {"metrics": {"loss": 0.20}, "status": "kept"},
        {"metrics": {"loss": 0.18}, "status": "kept"},  # improvement
        {"metrics": {"loss": 0.19}, "status": "kept"},
        {"metrics": {"loss": 0.19}, "status": "kept"},
    ]
    result = detect_failure_streak(exps, "loss", lower_is_better=True)
    assert result["streak_length"] >= 1


# --- Diagnosis edge cases ---

def test_exhaustion_no_hyperparams():
    exps = [{"metrics": {"accuracy": 0.85}, "config": {}}] * 5
    result = diagnose_search_space_exhaustion(exps, "accuracy")
    assert result["score"] >= 0

def test_config_error_no_best():
    exps = [{"metrics": {"accuracy": 0.85}, "config": {"hyperparams": {"lr": 0.1}}}] * 5
    result = diagnose_systematic_config_error(exps, "accuracy", None)
    assert result["score"] >= 0

def test_data_issue_too_few():
    result = diagnose_data_issue([{"metrics": {"accuracy": 0.85}}], "accuracy")
    assert result["score"] == 0

def test_ceiling_no_metric():
    result = diagnose_metric_ceiling([], "accuracy", None)
    assert result["score"] == 0

def test_ceiling_low_metric_tight_range():
    exps = [{"metrics": {"accuracy": 0.501}}, {"metrics": {"accuracy": 0.502}}, {"metrics": {"accuracy": 0.503}}]
    result = diagnose_metric_ceiling(exps, "accuracy", 0.503)
    assert "score" in result

def test_noise_empty():
    result = diagnose_noise_floor([], "accuracy")
    assert result["score"] == 0

def test_noise_single_experiment():
    result = diagnose_noise_floor([{"metrics": {"accuracy": 0.85}}], "accuracy")
    assert result["score"] == 0


# --- run_postmortem edge cases ---

def test_postmortem_missing_config(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"metrics": {"accuracy": 0.85}, "status": "kept", "config": {}}\n')
    result = run_postmortem(config_path=str(tmp_path / "missing.yaml"), log_path=str(log))
    # Should still work with default config
    assert "error" not in result or "message" in result

def test_postmortem_window_larger_than_log(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.85}, "config": {}}\n')
    result = run_postmortem(window=100, config_path=str(config), log_path=str(log))
    assert "error" not in result


# --- save_postmortem_report ---

def test_save_report(tmp_path):
    report = {"streak_length": 5, "primary_diagnosis": "exhaustion", "generated_at": "2026-04-01"}
    path = save_postmortem_report(report, str(tmp_path / "postmortems"))
    assert path.exists()
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data["streak_length"] == 5


# --- Format edge cases ---

def test_format_all_diagnoses():
    report = {
        "streak_length": 10,
        "primary_diagnosis": "data_issue",
        "diagnosis_score": 0.6,
        "diagnosis_evidence": ["Multiple models fail similarly"],
        "all_diagnoses": {
            "search_space_exhaustion": {"score": 0.2},
            "systematic_config_error": {"score": 0.1},
            "data_issue": {"score": 0.6},
            "metric_ceiling": {"score": 0.0},
            "noise_floor": {"score": 0.1},
        },
        "recommendations": ["Run leak check", "Run sanity"],
        "generated_at": "2026-04-01",
    }
    text = format_postmortem_report(report)
    assert "DATA ISSUE" in text
    assert "60%" in text
    assert "◀" in text  # Primary diagnosis marker
