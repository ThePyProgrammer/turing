"""Tests for multi-seed experiment runner (seed_runner.py).

Phase 10.1: Verifies seed statistics computation, sensitivity detection,
CI calculation, report formatting, and YAML output structure.
"""

from __future__ import annotations

import pytest

from scripts.seed_runner import (
    CV_THRESHOLD,
    DEFAULT_SEEDS,
    compute_seed_statistics,
    format_seed_report,
    get_experiment_config,
)


# --- compute_seed_statistics ---


def test_statistics_basic():
    """Should compute correct mean, std, CI for stable values."""
    values = [0.872, 0.868, 0.871, 0.855, 0.873]
    seeds = [42, 123, 456, 789, 1024]
    stats = compute_seed_statistics(values, seeds)

    assert stats["mean"] == pytest.approx(0.8678, abs=0.001)
    assert stats["std"] > 0
    assert len(stats["ci_95"]) == 2
    assert stats["ci_95"][0] < stats["mean"]
    assert stats["ci_95"][1] > stats["mean"]
    assert stats["cv_percent"] < CV_THRESHOLD
    assert stats["seed_sensitive"] is False


def test_statistics_seed_sensitive():
    """High variance should flag as seed-sensitive (CV > 5%)."""
    values = [0.90, 0.70, 0.85, 0.60, 0.80]
    seeds = [42, 123, 456, 789, 1024]
    stats = compute_seed_statistics(values, seeds)

    assert stats["cv_percent"] > CV_THRESHOLD
    assert stats["seed_sensitive"] is True


def test_statistics_identical_values():
    """Identical values should have zero std and CV."""
    values = [0.85, 0.85, 0.85]
    seeds = [42, 123, 456]
    stats = compute_seed_statistics(values, seeds)

    assert stats["std"] == 0.0
    assert stats["cv_percent"] == 0.0
    assert stats["seed_sensitive"] is False


def test_statistics_best_worst_seeds():
    """Should correctly identify best and worst seeds."""
    values = [0.80, 0.90, 0.85]
    seeds = [42, 123, 456]
    stats = compute_seed_statistics(values, seeds)

    assert stats["best_seed"] == 123
    assert stats["best_value"] == pytest.approx(0.90, abs=0.001)
    assert stats["worst_seed"] == 42
    assert stats["worst_value"] == pytest.approx(0.80, abs=0.001)


def test_statistics_range():
    """Range should be max - min."""
    values = [0.80, 0.85, 0.90]
    seeds = [42, 123, 456]
    stats = compute_seed_statistics(values, seeds)

    assert stats["range"] == pytest.approx(0.10, abs=0.001)


def test_statistics_two_values():
    """Should work with minimum viable N=2."""
    values = [0.80, 0.90]
    seeds = [42, 123]
    stats = compute_seed_statistics(values, seeds)

    assert stats["mean"] == pytest.approx(0.85, abs=0.001)
    assert stats["std"] > 0
    assert len(stats["ci_95"]) == 2


def test_statistics_ci_contains_mean():
    """95% CI must always contain the mean."""
    for values in [
        [0.84, 0.85, 0.86, 0.85, 0.84],
        [0.70, 0.90, 0.80, 0.75, 0.85],
        [0.99, 0.98, 0.97, 0.99, 0.98],
    ]:
        seeds = list(range(len(values)))
        stats = compute_seed_statistics(values, seeds)
        assert stats["ci_95"][0] <= stats["mean"] <= stats["ci_95"][1]


# --- get_experiment_config ---


def test_get_experiment_by_id():
    """Should find experiment by exact ID match."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.80}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"accuracy": 0.85}},
    ]
    result = get_experiment_config(experiments, "exp-002", "accuracy", False)
    assert result["experiment_id"] == "exp-002"


def test_get_experiment_best_when_no_id():
    """Should return best kept experiment when no ID specified."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.80}},
        {"experiment_id": "exp-002", "status": "discarded", "metrics": {"accuracy": 0.90}},
        {"experiment_id": "exp-003", "status": "kept", "metrics": {"accuracy": 0.85}},
    ]
    result = get_experiment_config(experiments, None, "accuracy", False)
    assert result["experiment_id"] == "exp-003"


def test_get_experiment_lower_is_better():
    """Should respect lower_is_better for best experiment selection."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"mse": 0.10}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"mse": 0.05}},
    ]
    result = get_experiment_config(experiments, None, "mse", True)
    assert result["experiment_id"] == "exp-002"


def test_get_experiment_not_found():
    """Should return None for missing ID."""
    experiments = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.80}},
    ]
    assert get_experiment_config(experiments, "exp-999", "accuracy", False) is None


def test_get_experiment_empty_log():
    """Should return None for empty experiment list."""
    assert get_experiment_config([], None, "accuracy", False) is None


# --- format_seed_report ---


def test_format_report_stable():
    """Stable study should show STABLE verdict."""
    study = {
        "experiment_id": "exp-042",
        "metric": "accuracy",
        "lower_is_better": False,
        "seeds_run": [42, 123, 456, 789, 1024],
        "results": [0.872, 0.868, 0.871, 0.855, 0.873],
        "mean": 0.8678,
        "std": 0.0071,
        "ci_95": [0.859, 0.877],
        "cv_percent": 0.82,
        "seed_sensitive": False,
        "best_seed": 1024,
        "best_value": 0.873,
        "worst_seed": 789,
        "worst_value": 0.855,
        "range": 0.018,
    }
    report = format_seed_report(study)
    assert "STABLE" in report
    assert "exp-042" in report
    assert "0.8678" in report


def test_format_report_sensitive():
    """Seed-sensitive study should show SEED-SENSITIVE verdict."""
    study = {
        "experiment_id": "exp-001",
        "metric": "accuracy",
        "lower_is_better": False,
        "seeds_run": [42, 123, 456],
        "results": [0.90, 0.70, 0.80],
        "mean": 0.80,
        "std": 0.10,
        "ci_95": [0.55, 1.05],
        "cv_percent": 12.5,
        "seed_sensitive": True,
        "best_seed": 42,
        "best_value": 0.90,
        "worst_seed": 123,
        "worst_value": 0.70,
        "range": 0.20,
    }
    report = format_seed_report(study)
    assert "SEED-SENSITIVE" in report


def test_format_report_error():
    """Error study should show error message."""
    study = {"error": "No experiment found", "experiment_id": "exp-999"}
    report = format_seed_report(study)
    assert "ERROR" in report


def test_format_report_with_failed_seeds():
    """Should mention failed seeds in report."""
    study = {
        "experiment_id": "exp-001",
        "metric": "accuracy",
        "lower_is_better": False,
        "seeds_run": [42, 123],
        "results": [0.85, 0.86],
        "mean": 0.855,
        "std": 0.007,
        "ci_95": [0.80, 0.91],
        "cv_percent": 0.82,
        "seed_sensitive": False,
        "best_seed": 123,
        "best_value": 0.86,
        "worst_seed": 42,
        "worst_value": 0.85,
        "range": 0.01,
        "failed_seeds": [456, 789],
    }
    report = format_seed_report(study)
    assert "failed" in report.lower()
    assert "456" in report


# --- DEFAULT_SEEDS ---


def test_default_seeds_count():
    """Should have at least 10 default seeds."""
    assert len(DEFAULT_SEEDS) >= 10


def test_default_seeds_unique():
    """Default seeds should all be unique."""
    assert len(DEFAULT_SEEDS) == len(set(DEFAULT_SEEDS))


def test_cv_threshold_positive():
    """CV threshold must be positive."""
    assert CV_THRESHOLD > 0
