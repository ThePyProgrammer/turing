"""Tests for performance regression gate (regression_gate.py).

Phase 16.3: Verifies verdict logic, environment diff, tolerance checking,
best experiment finding, and report formatting.
"""

from __future__ import annotations

import pytest

from scripts.regression_gate import (
    DEFAULT_TOLERANCE,
    determine_verdict,
    diff_environments,
    find_best_experiment,
    format_regression_report,
)


# --- find_best_experiment ---


def test_find_best_higher_is_better():
    """Should find experiment with highest primary metric."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.80}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"accuracy": 0.90}},
        {"experiment_id": "exp-003", "status": "kept", "metrics": {"accuracy": 0.85}},
    ]
    best = find_best_experiment(experiments, "accuracy", lower_is_better=False)
    assert best["experiment_id"] == "exp-002"


def test_find_best_lower_is_better():
    """Should find experiment with lowest primary metric."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"loss": 0.30}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"loss": 0.20}},
    ]
    best = find_best_experiment(experiments, "loss", lower_is_better=True)
    assert best["experiment_id"] == "exp-002"


def test_find_best_no_kept():
    """Should fall back to all experiments if none are kept."""
    experiments = [
        {"experiment_id": "exp-001", "status": "discarded", "metrics": {"accuracy": 0.80}},
    ]
    best = find_best_experiment(experiments, "accuracy")
    assert best["experiment_id"] == "exp-001"


def test_find_best_empty():
    """Should return None for empty experiment list."""
    assert find_best_experiment([], "accuracy") is None


def test_find_best_no_metric():
    """Experiment without metric should still be returned (caller checks)."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"f1": 0.80}},
    ]
    # find_best_experiment returns the experiment; caller handles missing metric
    best = find_best_experiment(experiments, "accuracy")
    assert best is not None
    assert "accuracy" not in best.get("metrics", {})


# --- determine_verdict ---


def test_verdict_pass_no_degradation():
    """No degradation should pass."""
    result = determine_verdict(
        {"accuracy": 0.85, "f1": 0.83},
        [{"accuracy": 0.86, "f1": 0.84}],
        "accuracy", 0.01,
    )
    assert result["verdict"] == "pass"


def test_verdict_pass_within_tolerance():
    """Small degradation within tolerance should pass."""
    result = determine_verdict(
        {"accuracy": 0.85},
        [{"accuracy": 0.846}],  # 0.47% degradation < 1% tolerance
        "accuracy", 0.01,
    )
    assert result["verdict"] == "pass"


def test_verdict_warning():
    """Degradation within 2x tolerance should warn."""
    result = determine_verdict(
        {"accuracy": 0.85},
        [{"accuracy": 0.835}],  # ~1.8% degradation, within 2x 1% tolerance
        "accuracy", 0.01,
    )
    assert result["verdict"] == "warning"


def test_verdict_fail():
    """Large degradation should fail."""
    result = determine_verdict(
        {"accuracy": 0.85},
        [{"accuracy": 0.80}],  # ~5.9% degradation > 2% tolerance
        "accuracy", 0.01,
    )
    assert result["verdict"] == "fail"


def test_verdict_multi_run():
    """Multiple runs should use mean."""
    result = determine_verdict(
        {"accuracy": 0.85},
        [{"accuracy": 0.84}, {"accuracy": 0.85}, {"accuracy": 0.86}],
        "accuracy", 0.01,
    )
    assert result["verdict"] == "pass"
    acc = result["per_metric"]["accuracy"]
    assert abs(acc["new_mean"] - 0.85) < 0.01


def test_verdict_per_metric_details():
    """Should include per-metric breakdown."""
    result = determine_verdict(
        {"accuracy": 0.85, "f1": 0.83},
        [{"accuracy": 0.86, "f1": 0.80}],
        "accuracy", 0.01,
    )
    assert "accuracy" in result["per_metric"]
    assert "f1" in result["per_metric"]
    f1 = result["per_metric"]["f1"]
    assert f1["degraded"] is True


def test_verdict_lower_is_better():
    """Lower-is-better metrics should treat increases as degradation."""
    result = determine_verdict(
        {"loss": 0.20},
        [{"loss": 0.30}],  # Loss went up = degradation
        "loss", 0.01, lower_is_better=True,
    )
    assert result["verdict"] == "fail"
    assert result["per_metric"]["loss"]["degraded"] is True


def test_verdict_improvement_lower_is_better():
    """Lower-is-better improvement should pass."""
    result = determine_verdict(
        {"loss": 0.30},
        [{"loss": 0.20}],  # Loss went down = improvement
        "loss", 0.01, lower_is_better=True,
    )
    assert result["verdict"] == "pass"
    assert result["per_metric"]["loss"]["degraded"] is False


def test_verdict_multi_run_std():
    """Multiple runs should include std."""
    result = determine_verdict(
        {"accuracy": 0.85},
        [{"accuracy": 0.84}, {"accuracy": 0.85}, {"accuracy": 0.86}],
        "accuracy", 0.01,
    )
    acc = result["per_metric"]["accuracy"]
    assert "new_std" in acc


def test_verdict_skips_metadata():
    """Should skip metadata keys like train_seconds."""
    result = determine_verdict(
        {"accuracy": 0.85, "train_seconds": 100},
        [{"accuracy": 0.85, "train_seconds": 200}],
        "accuracy", 0.01,
    )
    assert "train_seconds" not in result["per_metric"]


# --- diff_environments ---


def test_env_diff_no_original():
    """Missing original should return info message."""
    diffs = diff_environments(None, {"python_version": "3.10"})
    assert len(diffs) == 1
    assert "No original" in diffs[0]["detail"]


def test_env_diff_identical():
    """Identical environments should return empty."""
    env = {"python_version": "3.10", "packages": {"numpy": "1.24"}}
    assert diff_environments(env, env) == []


def test_env_diff_critical_package():
    """Critical package change should have critical severity."""
    orig = {"python_version": "3.10", "packages": {"numpy": "1.24"}}
    curr = {"python_version": "3.10", "packages": {"numpy": "1.26"}}
    diffs = diff_environments(orig, curr)
    assert len(diffs) == 1
    assert diffs[0]["severity"] == "critical"


def test_env_diff_python_version():
    """Python version change should be warning."""
    orig = {"python_version": "3.10", "packages": {}}
    curr = {"python_version": "3.11", "packages": {}}
    diffs = diff_environments(orig, curr)
    assert len(diffs) == 1
    assert diffs[0]["severity"] == "warning"


def test_env_diff_non_critical():
    """Non-critical package should be info severity."""
    orig = {"python_version": "3.10", "packages": {"requests": "2.28"}}
    curr = {"python_version": "3.10", "packages": {"requests": "2.31"}}
    diffs = diff_environments(orig, curr)
    assert diffs[0]["severity"] == "info"


# --- format_regression_report ---


def test_format_report_pass():
    """Pass verdict should show PASS."""
    report = {
        "baseline_id": "exp-042",
        "checked_at": "2026-01-01T00:00:00",
        "verdict": "pass",
        "primary_metric": "accuracy",
        "tolerance": 0.01,
        "mode": "full",
        "n_runs": 3,
        "failed_runs": 0,
        "per_metric": {
            "accuracy": {
                "original": 0.85, "new_mean": 0.852,
                "delta": 0.002, "relative_diff": 0.002,
                "degraded": False, "verdict": "pass",
            },
        },
        "environment_diffs": [],
    }
    text = format_regression_report(report)
    assert "PASS" in text
    assert "exp-042" in text


def test_format_report_fail():
    """Fail verdict should show FAIL and REGRESSION."""
    report = {
        "baseline_id": "exp-042",
        "checked_at": "2026-01-01T00:00:00",
        "verdict": "fail",
        "primary_metric": "accuracy",
        "tolerance": 0.01,
        "mode": "full",
        "n_runs": 3,
        "failed_runs": 0,
        "per_metric": {
            "accuracy": {
                "original": 0.85, "new_mean": 0.80,
                "delta": -0.05, "relative_diff": 0.059,
                "degraded": True, "verdict": "fail",
            },
        },
        "environment_diffs": [
            {"severity": "critical", "detail": "numpy: 1.24 -> 1.26"},
        ],
    }
    text = format_regression_report(report)
    assert "FAIL" in text
    assert "REGRESSION" in text
    assert "numpy" in text


def test_format_report_error():
    """Error should show error message."""
    text = format_regression_report({"error": "No experiments found"})
    assert "ERROR" in text


def test_format_report_warning():
    """Warning verdict should show WARNING."""
    report = {
        "baseline_id": "exp-001",
        "checked_at": "2026-01-01T00:00:00",
        "verdict": "warning",
        "primary_metric": "accuracy",
        "tolerance": 0.01,
        "mode": "quick",
        "n_runs": 1,
        "failed_runs": 0,
        "per_metric": {
            "accuracy": {
                "original": 0.85, "new_mean": 0.837,
                "delta": -0.013, "relative_diff": 0.015,
                "degraded": True, "verdict": "warning",
            },
        },
        "environment_diffs": [],
    }
    text = format_regression_report(report)
    assert "WARNING" in text
