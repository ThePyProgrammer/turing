"""Tests for pre-submission methodology audit (methodology_audit.py).

Phase 19.2: Verifies each check type, verdict logic, scoring,
venue checklists, and report formatting.
"""

from __future__ import annotations

import json

import pytest

from scripts.methodology_audit import (
    check_ablation,
    check_baseline,
    check_cv_strategy,
    check_data_leakage,
    check_hyperparameter_budget,
    check_regression_stability,
    check_reproducibility,
    check_seed_sensitivity,
    format_audit_report,
    get_venue_checks,
    run_audit,
)


# --- check_seed_sensitivity ---


def test_seed_pass(tmp_path):
    """Should pass when seed studies exist for best experiments."""
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "exp-001-seeds.yaml").write_text("metric: accuracy\nmean: 0.85")
    experiments = [{"experiment_id": "exp-001", "status": "kept"}]
    result = check_seed_sensitivity(experiments, str(seed_dir))
    assert result["status"] == "pass"


def test_seed_fail(tmp_path):
    """Should fail when no seed studies exist."""
    experiments = [{"experiment_id": "exp-001", "status": "kept"}]
    result = check_seed_sensitivity(experiments, str(tmp_path / "none"))
    assert result["status"] == "fail"


def test_seed_skip_no_experiments():
    """Should skip when no kept experiments."""
    result = check_seed_sensitivity([])
    assert result["status"] == "skip"


# --- check_ablation ---


def test_ablation_pass(tmp_path):
    """Should pass when ablation files exist."""
    abl_dir = tmp_path / "ablations"
    abl_dir.mkdir()
    (abl_dir / "ablation-exp-001.yaml").write_text("components: [a, b]")
    result = check_ablation([], str(abl_dir))
    assert result["status"] == "pass"


def test_ablation_fail(tmp_path):
    """Should fail when no ablation files."""
    result = check_ablation([], str(tmp_path / "none"))
    assert result["status"] == "fail"
    assert result.get("fix") == "/turing:ablate"


# --- check_baseline ---


def test_baseline_pass():
    """Should pass when baseline experiments exist."""
    experiments = [
        {"experiment_id": "exp-001", "config": {"model_type": "majority_baseline"}, "description": ""},
    ]
    result = check_baseline(experiments)
    assert result["status"] == "pass"


def test_baseline_from_description():
    """Should detect baselines from description field."""
    experiments = [
        {"experiment_id": "exp-001", "config": {"model_type": "xgboost"}, "description": "random baseline comparison"},
    ]
    result = check_baseline(experiments)
    assert result["status"] == "pass"


def test_baseline_fail():
    """Should fail when no baselines found."""
    experiments = [
        {"experiment_id": "exp-001", "config": {"model_type": "xgboost"}, "description": "tuned model"},
    ]
    result = check_baseline(experiments)
    assert result["status"] == "fail"


# --- check_reproducibility ---


def test_repro_pass(tmp_path):
    """Should pass when reproduction passed."""
    import yaml
    repro_dir = tmp_path / "repros"
    repro_dir.mkdir()
    report = {"experiment_id": "exp-001", "verdict": "reproducible"}
    with open(repro_dir / "exp-001-repro.yaml", "w") as f:
        yaml.dump(report, f)
    result = check_reproducibility([], str(repro_dir))
    assert result["status"] == "pass"


def test_repro_fail(tmp_path):
    """Should fail when no reproduction reports."""
    result = check_reproducibility([], str(tmp_path / "none"))
    assert result["status"] == "fail"


def test_repro_warn_no_pass(tmp_path):
    """Should warn when reproductions exist but none passed."""
    import yaml
    repro_dir = tmp_path / "repros"
    repro_dir.mkdir()
    report = {"experiment_id": "exp-001", "verdict": "not_reproducible"}
    with open(repro_dir / "exp-001-repro.yaml", "w") as f:
        yaml.dump(report, f)
    result = check_reproducibility([], str(repro_dir))
    assert result["status"] == "warn"


# --- check_hyperparameter_budget ---


def test_budget_pass():
    """Should pass and report stats."""
    experiments = [
        {"metrics": {"train_seconds": 3600}},
        {"metrics": {"train_seconds": 1800}},
    ]
    result = check_hyperparameter_budget(experiments)
    assert result["status"] == "pass"
    assert result["detail"]["total_hours"] == 1.5


def test_budget_empty():
    """Empty experiments should warn."""
    result = check_hyperparameter_budget([])
    assert result["status"] == "warn"


# --- check_data_leakage ---


def test_leakage_pass(tmp_path, monkeypatch):
    """Should pass when prepare.py and evaluate.py exist."""
    (tmp_path / "prepare.py").write_text("# data prep")
    (tmp_path / "evaluate.py").write_text("# eval")
    monkeypatch.chdir(tmp_path)
    result = check_data_leakage({})
    assert result["status"] == "pass"


def test_leakage_warn_no_files(tmp_path, monkeypatch):
    """Should warn when files missing."""
    monkeypatch.chdir(tmp_path)
    result = check_data_leakage({})
    assert result["status"] == "warn"


# --- check_cv_strategy ---


def test_cv_pass():
    """Should pass when CV strategy specified."""
    config = {"evaluation": {"cv_strategy": "stratified_kfold"}}
    result = check_cv_strategy(config)
    assert result["status"] == "pass"


def test_cv_warn():
    """Should warn when no CV strategy."""
    result = check_cv_strategy({})
    assert result["status"] == "warn"


# --- check_regression_stability ---


def test_regression_pass(tmp_path):
    """Should pass when regression checks exist."""
    reg_dir = tmp_path / "regs"
    reg_dir.mkdir()
    (reg_dir / "check-2026-01-01.yaml").write_text("verdict: pass")
    result = check_regression_stability(str(reg_dir))
    assert result["status"] == "pass"


def test_regression_warn(tmp_path):
    """Should warn when no checks."""
    result = check_regression_stability(str(tmp_path / "none"))
    assert result["status"] == "warn"


# --- get_venue_checks ---


def test_venue_neurips():
    """NeurIPS should add specific checks."""
    checks = get_venue_checks("neurips")
    assert len(checks) >= 2
    assert all(c["status"] == "manual" for c in checks)


def test_venue_none():
    """No venue should return empty."""
    assert get_venue_checks(None) == []


def test_venue_unknown():
    """Unknown venue should return empty."""
    assert get_venue_checks("unknown_venue") == []


# --- run_audit ---


def test_audit_basic(tmp_path, monkeypatch):
    """Should run full audit and produce report."""
    import yaml
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(yaml.dump({"evaluation": {"primary_metric": "accuracy"}}))
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "log.jsonl").write_text("")

    report = run_audit(config_path=str(tmp_path / "config.yaml"), log_path=str(tmp_path / "experiments" / "log.jsonl"))
    assert "checks" in report
    assert "score" in report
    assert "verdict" in report


# --- format_audit_report ---


def test_format_report_pass():
    """Pass verdict should show PASS."""
    report = {
        "audited_at": "2026-01-01T00:00:00",
        "strict_mode": False,
        "venue": None,
        "checks": [
            {"check": "seed_sensitivity", "status": "pass", "reason": "5-seed study", "severity": "high"},
        ],
        "score": {"pass": 1, "fail": 0, "warn": 0, "skip": 0, "manual": 0, "total": 1, "checkable": 1},
        "verdict": "pass",
        "actions": [],
    }
    text = format_audit_report(report)
    assert "PASS" in text
    assert "1/1" in text


def test_format_report_fail():
    """Fail verdict should show required actions."""
    report = {
        "audited_at": "2026-01-01T00:00:00",
        "strict_mode": False,
        "venue": None,
        "checks": [
            {"check": "baseline_comparison", "status": "fail", "reason": "No baselines", "severity": "high", "fix": "/turing:try 'baseline'"},
        ],
        "score": {"pass": 0, "fail": 1, "warn": 0, "skip": 0, "manual": 0, "total": 1, "checkable": 1},
        "verdict": "needs_work",
        "actions": [{"check": "baseline_comparison", "fix": "/turing:try 'baseline'", "severity": "high"}],
    }
    text = format_audit_report(report)
    assert "NEEDS WORK" in text
    assert "Required Actions" in text


def test_format_report_error():
    text = format_audit_report({"error": "Bad config"})
    assert "ERROR" in text


def test_format_report_venue():
    """Should show venue name."""
    report = {
        "audited_at": "2026-01-01T00:00:00",
        "strict_mode": False,
        "venue": "neurips",
        "checks": [],
        "score": {"pass": 0, "fail": 0, "warn": 0, "skip": 0, "manual": 0, "total": 0, "checkable": 0},
        "verdict": "pass",
        "actions": [],
    }
    text = format_audit_report(report)
    assert "neurips" in text
