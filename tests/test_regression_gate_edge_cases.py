"""Edge case tests for performance regression gate (regression_gate.py).

Phase 16.3: Covers no experiments, quick mode, identical metrics,
environment-only changes, zero values, mixed verdicts, and boundary conditions.
"""

from __future__ import annotations

import pytest

from scripts.regression_gate import (
    determine_verdict,
    diff_environments,
    find_best_experiment,
    format_regression_report,
)


# --- find_best_experiment edge cases ---


def test_find_best_all_discarded():
    """All discarded experiments should still return one (fallback)."""
    experiments = [
        {"experiment_id": "exp-001", "status": "discarded", "metrics": {"accuracy": 0.70}},
        {"experiment_id": "exp-002", "status": "discarded", "metrics": {"accuracy": 0.75}},
    ]
    best = find_best_experiment(experiments, "accuracy")
    assert best["experiment_id"] == "exp-002"


def test_find_best_single_experiment():
    """Single experiment should be returned."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.85}},
    ]
    best = find_best_experiment(experiments, "accuracy")
    assert best["experiment_id"] == "exp-001"


def test_find_best_tie():
    """Tied metrics should return one deterministically."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.85}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"accuracy": 0.85}},
    ]
    best = find_best_experiment(experiments, "accuracy")
    assert best is not None


# --- determine_verdict edge cases ---


def test_verdict_identical_metrics():
    """Identical metrics should always pass."""
    result = determine_verdict(
        {"accuracy": 0.85},
        [{"accuracy": 0.85}],
        "accuracy", 0.01,
    )
    assert result["verdict"] == "pass"
    assert result["per_metric"]["accuracy"]["delta"] == 0


def test_verdict_zero_original():
    """Zero original value should not crash."""
    result = determine_verdict(
        {"accuracy": 0.0},
        [{"accuracy": 0.01}],
        "accuracy", 0.01,
    )
    assert "verdict" in result
    assert result["per_metric"]["accuracy"]["degraded"] is False  # Improvement from 0


def test_verdict_empty_new_metrics():
    """No matching metrics should produce empty per_metric."""
    result = determine_verdict(
        {"accuracy": 0.85},
        [{"f1": 0.80}],  # Different metric
        "accuracy", 0.01,
    )
    assert result["verdict"] == "pass"
    assert len(result["per_metric"]) == 0


def test_verdict_mixed_pass_and_fail():
    """One passing and one failing metric should fail overall."""
    result = determine_verdict(
        {"accuracy": 0.85, "f1": 0.83},
        [{"accuracy": 0.86, "f1": 0.75}],
        "accuracy", 0.01,
    )
    assert result["verdict"] == "fail"
    assert result["per_metric"]["accuracy"]["verdict"] == "pass"
    assert result["per_metric"]["f1"]["verdict"] == "fail"


def test_verdict_mixed_pass_and_warning():
    """Warning + pass should be warning overall."""
    result = determine_verdict(
        {"accuracy": 0.85, "f1": 0.83},
        [{"accuracy": 0.86, "f1": 0.82}],  # f1 degraded ~1.2%
        "accuracy", 0.01,
    )
    assert result["verdict"] in ("pass", "warning")


def test_verdict_very_tight_tolerance():
    """Very tight tolerance should catch tiny changes."""
    result = determine_verdict(
        {"accuracy": 0.850000},
        [{"accuracy": 0.849000}],  # 0.12% degradation
        "accuracy", 0.001,  # 0.1% tolerance
    )
    assert result["per_metric"]["accuracy"]["degraded"] is True
    assert result["per_metric"]["accuracy"]["verdict"] in ("warning", "fail")


def test_verdict_very_loose_tolerance():
    """Very loose tolerance should pass large changes."""
    result = determine_verdict(
        {"accuracy": 0.85},
        [{"accuracy": 0.80}],  # 5.9% degradation
        "accuracy", 0.10,  # 10% tolerance
    )
    assert result["verdict"] == "pass"


def test_verdict_multiple_metrics_all_improve():
    """All improving metrics should pass."""
    result = determine_verdict(
        {"accuracy": 0.85, "f1": 0.83, "precision": 0.90},
        [{"accuracy": 0.87, "f1": 0.85, "precision": 0.92}],
        "accuracy", 0.01,
    )
    assert result["verdict"] == "pass"
    for m in result["per_metric"].values():
        assert m["degraded"] is False


def test_verdict_non_numeric_filtered():
    """Non-numeric metrics should be skipped."""
    result = determine_verdict(
        {"accuracy": 0.85, "model_type": "xgboost"},
        [{"accuracy": 0.85, "model_type": "lightgbm"}],
        "accuracy", 0.01,
    )
    assert "model_type" not in result["per_metric"]


# --- diff_environments edge cases ---


def test_env_diff_empty_packages():
    """Both empty packages should produce no diffs."""
    env = {"python_version": "3.10", "packages": {}}
    assert diff_environments(env, env) == []


def test_env_diff_new_package():
    """New package (not in original) should not be flagged (only changes)."""
    orig = {"python_version": "3.10", "packages": {}}
    curr = {"python_version": "3.10", "packages": {"new_pkg": "1.0"}}
    diffs = diff_environments(orig, curr)
    # Only flags changes/removals, not additions
    assert len(diffs) == 0


def test_env_diff_multiple_critical():
    """Multiple critical package changes should all be flagged."""
    orig = {"python_version": "3.10", "packages": {"numpy": "1.24", "scipy": "1.10"}}
    curr = {"python_version": "3.10", "packages": {"numpy": "1.26", "scipy": "1.12"}}
    diffs = diff_environments(orig, curr)
    assert len(diffs) == 2
    assert all(d["severity"] == "critical" for d in diffs)


# --- format_regression_report edge cases ---


def test_format_report_no_per_metric():
    """Report with no per-metric data should still render."""
    report = {
        "baseline_id": "exp-001",
        "checked_at": "2026-01-01T00:00:00",
        "verdict": "pass",
        "primary_metric": "accuracy",
        "tolerance": 0.01,
        "mode": "quick",
        "n_runs": 1,
        "failed_runs": 0,
        "per_metric": {},
        "environment_diffs": [],
    }
    text = format_regression_report(report)
    assert "PASS" in text
    assert "Metric Comparison" in text


def test_format_report_with_std():
    """Multi-run report should show run details."""
    report = {
        "baseline_id": "exp-001",
        "checked_at": "2026-01-01T00:00:00",
        "verdict": "pass",
        "primary_metric": "accuracy",
        "tolerance": 0.01,
        "mode": "full",
        "n_runs": 3,
        "failed_runs": 1,
        "per_metric": {
            "accuracy": {
                "original": 0.85, "new_mean": 0.852,
                "delta": 0.002, "relative_diff": 0.002,
                "degraded": False, "verdict": "pass",
                "new_std": 0.005,
            },
        },
        "environment_diffs": [],
    }
    text = format_regression_report(report)
    assert "Successful runs" in text
    assert "Failed runs" in text
    assert "std" in text
