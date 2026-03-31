"""Tests for reproducibility verification (reproduce_experiment.py).

Phase 10.2: Verifies verdict logic, environment comparison,
tolerance checking, and report formatting.
"""

from __future__ import annotations

import pytest

from scripts.reproduce_experiment import (
    FLOAT_TOLERANCE,
    compare_environments,
    determine_verdict,
    find_experiment,
    format_reproduction_report,
)


# --- find_experiment ---


def test_find_experiment_by_id():
    """Should find experiment by exact ID."""
    experiments = [
        {"experiment_id": "exp-001", "metrics": {"accuracy": 0.80}},
        {"experiment_id": "exp-002", "metrics": {"accuracy": 0.85}},
    ]
    result = find_experiment(experiments, "exp-002")
    assert result["experiment_id"] == "exp-002"


def test_find_experiment_missing():
    """Should return None for missing ID."""
    experiments = [{"experiment_id": "exp-001"}]
    assert find_experiment(experiments, "exp-999") is None


def test_find_experiment_empty_list():
    """Should return None for empty list."""
    assert find_experiment([], "exp-001") is None


# --- determine_verdict ---


def test_verdict_exact_match():
    """Identical values should be reproducible."""
    result = determine_verdict(0.85, [0.85], tolerance=0.02, strict=True)
    assert result["verdict"] == "reproducible"


def test_verdict_strict_fail():
    """Strict mode should fail on any meaningful difference."""
    result = determine_verdict(0.85, [0.84], tolerance=0.02, strict=True)
    assert result["verdict"] == "not_reproducible"


def test_verdict_approximate_within_tolerance():
    """Within tolerance should be approximately reproducible."""
    result = determine_verdict(0.85, [0.84, 0.845, 0.843], tolerance=0.02, strict=False)
    assert result["verdict"] == "approximately_reproducible"


def test_verdict_not_reproducible():
    """Large difference should fail."""
    result = determine_verdict(0.85, [0.70, 0.71, 0.69], tolerance=0.02, strict=False)
    assert result["verdict"] == "not_reproducible"


def test_verdict_within_ci():
    """Original within CI of new distribution should pass."""
    # These values are close enough that 0.85 should fall in CI
    result = determine_verdict(0.85, [0.845, 0.855, 0.850, 0.848, 0.852], tolerance=0.001, strict=False)
    assert result["verdict"] == "approximately_reproducible"


def test_verdict_fields():
    """Result should contain all expected fields."""
    result = determine_verdict(0.85, [0.84, 0.85, 0.86], tolerance=0.02, strict=False)
    assert "verdict" in result
    assert "reason" in result
    assert "original_value" in result
    assert "new_mean" in result
    assert "n_runs" in result


def test_verdict_single_run_not_strict():
    """Single non-strict run within tolerance should pass."""
    result = determine_verdict(0.85, [0.849], tolerance=0.02, strict=False)
    # With single run and not strict, falls through to tolerance check
    assert result["verdict"] in ("reproducible", "approximately_reproducible")


def test_verdict_zero_original():
    """Should handle zero original value without division error."""
    result = determine_verdict(0.0, [0.001, 0.002, 0.001], tolerance=0.02, strict=False)
    assert "verdict" in result


def test_verdict_statistical_details():
    """Multi-run verdict should include CI."""
    result = determine_verdict(0.85, [0.84, 0.85, 0.86], tolerance=0.02, strict=False)
    assert "ci_95" in result
    assert len(result["ci_95"]) == 2
    assert "new_std" in result


# --- compare_environments ---


def test_env_compare_no_original():
    """Missing original env should return info-level message."""
    diffs = compare_environments(None, {"python_version": "3.10.0"})
    assert len(diffs) == 1
    assert diffs[0]["severity"] == "info"


def test_env_compare_identical():
    """Identical environments should return no diffs."""
    env = {
        "python_version": "3.10.0",
        "packages": {"numpy": "1.24.0", "scipy": "1.10.0"},
    }
    diffs = compare_environments(env, env)
    assert len(diffs) == 0


def test_env_compare_python_version_change():
    """Python version change should be a warning."""
    orig = {"python_version": "3.10.0", "packages": {}}
    curr = {"python_version": "3.11.0", "packages": {}}
    diffs = compare_environments(orig, curr)
    py_diffs = [d for d in diffs if d["field"] == "python_version"]
    assert len(py_diffs) == 1
    assert py_diffs[0]["severity"] == "warning"


def test_env_compare_critical_package():
    """Change in critical ML package should be critical severity."""
    orig = {"python_version": "3.10.0", "packages": {"numpy": "1.24.0"}}
    curr = {"python_version": "3.10.0", "packages": {"numpy": "1.26.0"}}
    diffs = compare_environments(orig, curr)
    np_diffs = [d for d in diffs if "numpy" in d["field"]]
    assert len(np_diffs) == 1
    assert np_diffs[0]["severity"] == "critical"


def test_env_compare_non_critical_package():
    """Change in non-critical package should be info severity."""
    orig = {"python_version": "3.10.0", "packages": {"requests": "2.28.0"}}
    curr = {"python_version": "3.10.0", "packages": {"requests": "2.31.0"}}
    diffs = compare_environments(orig, curr)
    req_diffs = [d for d in diffs if "requests" in d["field"]]
    assert len(req_diffs) == 1
    assert req_diffs[0]["severity"] == "info"


def test_env_compare_missing_package():
    """Package removed from environment should be flagged."""
    orig = {"python_version": "3.10.0", "packages": {"scikit-learn": "1.2.0"}}
    curr = {"python_version": "3.10.0", "packages": {}}
    diffs = compare_environments(orig, curr)
    sk_diffs = [d for d in diffs if "scikit-learn" in d["field"]]
    assert len(sk_diffs) == 1
    assert "missing" in sk_diffs[0]["detail"]


# --- format_reproduction_report ---


def test_format_report_reproducible():
    """Reproducible verdict should show PASS."""
    report = {
        "experiment_id": "exp-042",
        "verdict": "reproducible",
        "reason": "Exact match within float tolerance",
        "original_metrics": {"accuracy": 0.8500},
        "new_metrics": {"accuracy": 0.8500},
        "statistical_details": {"n_runs": 1, "original_value": 0.85, "new_mean": 0.85},
        "strictness": "exact (1e-6)",
        "environment_diffs": [],
    }
    text = format_reproduction_report(report)
    assert "PASS" in text
    assert "exp-042" in text


def test_format_report_not_reproducible():
    """Failed reproduction should show FAIL."""
    report = {
        "experiment_id": "exp-001",
        "verdict": "not_reproducible",
        "reason": "Difference exceeds tolerance",
        "original_metrics": {"accuracy": 0.85},
        "new_metrics": {"accuracy": 0.70},
        "statistical_details": {
            "n_runs": 3,
            "original_value": 0.85,
            "new_mean": 0.70,
            "new_values": [0.69, 0.70, 0.71],
            "new_std": 0.01,
            "ci_95": [0.68, 0.72],
        },
        "strictness": "tolerance=0.02",
        "environment_diffs": [],
    }
    text = format_reproduction_report(report)
    assert "FAIL" in text


def test_format_report_environment_changed():
    """Environment changed verdict should show WARN."""
    report = {
        "experiment_id": "exp-001",
        "verdict": "environment_changed",
        "reason": "Metrics diverge and environment differs",
        "original_metrics": {"accuracy": 0.85},
        "new_metrics": {"accuracy": 0.80},
        "statistical_details": {"n_runs": 3, "original_value": 0.85, "new_mean": 0.80},
        "strictness": "tolerance=0.02",
        "environment_diffs": [
            {"field": "package:numpy", "severity": "critical",
             "detail": "numpy: 1.24.0 -> 1.26.0", "original": "1.24.0", "current": "1.26.0"},
        ],
    }
    text = format_reproduction_report(report)
    assert "WARN" in text
    assert "numpy" in text


def test_format_report_error():
    """Error should show error message."""
    report = {"error": "Experiment not found", "experiment_id": "exp-999"}
    text = format_reproduction_report(report)
    assert "ERROR" in text


def test_format_report_approx():
    """Approximately reproducible should show PASS (approx)."""
    report = {
        "experiment_id": "exp-001",
        "verdict": "approximately_reproducible",
        "reason": "Within 2.0% tolerance",
        "original_metrics": {"accuracy": 0.85},
        "new_metrics": {"accuracy": 0.845},
        "statistical_details": {
            "n_runs": 3,
            "original_value": 0.85,
            "new_mean": 0.845,
            "new_values": [0.84, 0.845, 0.85],
            "new_std": 0.005,
            "ci_95": [0.83, 0.86],
            "relative_difference": 0.006,
        },
        "strictness": "tolerance=0.02",
        "environment_diffs": [],
    }
    text = format_reproduction_report(report)
    assert "PASS" in text
    assert "approx" in text


# --- Constants ---


def test_float_tolerance_small():
    """Float tolerance should be very small for exact match."""
    assert FLOAT_TOLERANCE < 0.001
