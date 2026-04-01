"""Edge case tests for pre-submission methodology audit (methodology_audit.py).

Phase 19.2: Covers no experiments, all pass, all fail, strict mode,
empty config, and boundary conditions.
"""

from __future__ import annotations

import json

import pytest

from scripts.methodology_audit import (
    check_baseline,
    check_hyperparameter_budget,
    check_seed_sensitivity,
    format_audit_report,
    get_venue_checks,
    run_audit,
)


# --- check_seed_sensitivity edge cases ---


def test_seed_partial_coverage(tmp_path):
    """Some but not all best experiments studied should warn."""
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "exp-001-seeds.yaml").write_text("ok: true")
    experiments = [
        {"experiment_id": "exp-001", "status": "kept"},
        {"experiment_id": "exp-002", "status": "kept"},
        {"experiment_id": "exp-003", "status": "kept"},
    ]
    result = check_seed_sensitivity(experiments, str(seed_dir))
    assert result["status"] == "warn"


def test_seed_many_experiments_few_kept(tmp_path):
    """Should only check best kept experiments."""
    experiments = [
        {"experiment_id": f"exp-{i:03d}", "status": "discarded"} for i in range(20)
    ]
    experiments.append({"experiment_id": "exp-best", "status": "kept"})
    result = check_seed_sensitivity(experiments, str(tmp_path / "none"))
    assert result["status"] == "fail"  # No seed study for the one kept


# --- check_baseline edge cases ---


def test_baseline_multiple_keywords():
    """Should detect various baseline keywords."""
    for kw in ["majority", "random", "mean", "dummy", "naive"]:
        experiments = [{"experiment_id": "exp-001", "config": {"model_type": f"{kw}_predictor"}, "description": ""}]
        result = check_baseline(experiments)
        assert result["status"] == "pass", f"Failed to detect baseline keyword: {kw}"


def test_baseline_empty_experiments():
    """Empty experiment list should fail."""
    result = check_baseline([])
    assert result["status"] == "fail"


def test_baseline_no_config_model_type():
    """Experiments without model_type should not crash."""
    experiments = [{"experiment_id": "exp-001", "config": {}, "description": "some experiment"}]
    result = check_baseline(experiments)
    assert result["status"] == "fail"


# --- check_hyperparameter_budget edge cases ---


def test_budget_no_train_seconds():
    """Experiments without train_seconds should still count."""
    experiments = [{"metrics": {"accuracy": 0.85}}, {"metrics": {"accuracy": 0.87}}]
    result = check_hyperparameter_budget(experiments)
    assert result["status"] == "pass"
    assert result["detail"]["n_experiments"] == 2
    assert result["detail"]["total_hours"] == 0


def test_budget_mixed_metrics():
    """Some experiments with and without train_seconds."""
    experiments = [
        {"metrics": {"train_seconds": 3600}},
        {"metrics": {"accuracy": 0.85}},
        {"metrics": {"train_seconds": "invalid"}},
    ]
    result = check_hyperparameter_budget(experiments)
    assert result["detail"]["total_hours"] == 1.0  # Only first


# --- get_venue_checks edge cases ---


def test_venue_icml():
    """ICML should have checks."""
    checks = get_venue_checks("icml")
    assert len(checks) >= 1


def test_venue_iclr():
    """ICLR should have checks."""
    checks = get_venue_checks("iclr")
    assert len(checks) >= 1


def test_venue_case_insensitive():
    """Should be case-insensitive."""
    checks = get_venue_checks("NeurIPS")
    assert len(checks) >= 2


# --- run_audit edge cases ---


def test_audit_strict_mode(tmp_path, monkeypatch):
    """Strict mode should still produce a report."""
    import yaml
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.dump({}))
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "log.jsonl").write_text("")
    report = run_audit(strict=True, config_path=str(tmp_path / "config.yaml"), log_path=str(tmp_path / "experiments" / "log.jsonl"))
    assert report["strict_mode"] is True


def test_audit_with_venue(tmp_path, monkeypatch):
    """Venue checks should be added."""
    import yaml
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.dump({}))
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "log.jsonl").write_text("")
    report = run_audit(venue="neurips", config_path=str(tmp_path / "config.yaml"), log_path=str(tmp_path / "experiments" / "log.jsonl"))
    assert report["venue"] == "neurips"
    assert report["score"]["manual"] >= 2


def test_audit_all_pass_scenario(tmp_path, monkeypatch):
    """Scenario where most checks pass should produce pass verdict."""
    import yaml
    monkeypatch.chdir(tmp_path)
    config = {"evaluation": {"cv_strategy": "stratified_kfold"}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    (tmp_path / "prepare.py").write_text("# data prep")
    (tmp_path / "evaluate.py").write_text("# eval")

    # Create artifacts
    for d in ["experiments", "experiments/seed_studies", "experiments/ablations", "experiments/reproductions", "experiments/regressions"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    # Seed study
    (tmp_path / "experiments" / "seed_studies" / "exp-001-seeds.yaml").write_text("metric: accuracy")
    # Ablation
    (tmp_path / "experiments" / "ablations" / "ablation.yaml").write_text("components: [a]")
    # Reproduction
    repro = {"experiment_id": "exp-001", "verdict": "reproducible"}
    with open(tmp_path / "experiments" / "reproductions" / "exp-001-repro.yaml", "w") as f:
        yaml.dump(repro, f)
    # Regression check
    (tmp_path / "experiments" / "regressions" / "check-2026-01-01.yaml").write_text("verdict: pass")

    # Experiment log with baseline
    with open(tmp_path / "experiments" / "log.jsonl", "w") as f:
        f.write(json.dumps({"experiment_id": "exp-001", "status": "kept", "config": {"model_type": "majority_baseline"}, "description": "", "metrics": {"train_seconds": 60}}) + "\n")

    report = run_audit(config_path=str(tmp_path / "config.yaml"), log_path=str(tmp_path / "experiments" / "log.jsonl"))
    assert report["verdict"] in ("pass", "pass_with_warnings")
    assert report["score"]["fail"] == 0


# --- format_audit_report edge cases ---


def test_format_report_all_skip():
    """All skipped should still render."""
    report = {
        "audited_at": "2026-01-01T00:00:00",
        "strict_mode": False,
        "venue": None,
        "checks": [{"check": "seed_sensitivity", "status": "skip", "reason": "No experiments", "severity": "high"}],
        "score": {"pass": 0, "fail": 0, "warn": 0, "skip": 1, "manual": 0, "total": 1, "checkable": 0},
        "verdict": "pass",
        "actions": [],
    }
    text = format_audit_report(report)
    assert "SKIP" in text


def test_format_report_manual_checks():
    """Manual checks should show count."""
    report = {
        "audited_at": "2026-01-01T00:00:00",
        "strict_mode": False,
        "venue": "neurips",
        "checks": [{"check": "broader_impact", "status": "manual", "reason": "Check required", "severity": "medium"}],
        "score": {"pass": 0, "fail": 0, "warn": 0, "skip": 0, "manual": 1, "total": 1, "checkable": 0},
        "verdict": "pass",
        "actions": [],
    }
    text = format_audit_report(report)
    assert "manual" in text.lower()
