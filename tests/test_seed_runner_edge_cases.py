"""Edge case tests for seed_runner.py.

Tests boundary conditions, error handling, and YAML serialization.
"""

from __future__ import annotations

import yaml
import pytest

from scripts.seed_runner import (
    compute_seed_statistics,
    format_seed_report,
    save_seed_study,
)


def test_statistics_single_value():
    """Single value should have zero std and tight CI."""
    stats = compute_seed_statistics([0.85], [42])
    assert stats["std"] == 0.0
    assert stats["mean"] == pytest.approx(0.85, abs=0.001)
    assert stats["best_seed"] == 42
    assert stats["worst_seed"] == 42
    assert stats["seed_sensitive"] is False


def test_statistics_negative_values():
    """Should handle metrics that can be negative (e.g., log-loss)."""
    values = [-0.5, -0.6, -0.55, -0.58, -0.52]
    seeds = [42, 123, 456, 789, 1024]
    stats = compute_seed_statistics(values, seeds)
    assert stats["mean"] < 0
    assert stats["std"] > 0
    assert stats["ci_95"][0] < stats["ci_95"][1]


def test_statistics_very_large_values():
    """Should handle large metric values without overflow."""
    values = [1e6, 1e6 + 1, 1e6 - 1, 1e6 + 0.5, 1e6 - 0.5]
    seeds = [42, 123, 456, 789, 1024]
    stats = compute_seed_statistics(values, seeds)
    assert stats["mean"] == pytest.approx(1e6, rel=0.001)
    assert stats["cv_percent"] < 1.0  # Very low CV for tight values


def test_statistics_all_zeros():
    """All zeros should have zero mean/std, inf CV."""
    values = [0.0, 0.0, 0.0]
    seeds = [42, 123, 456]
    stats = compute_seed_statistics(values, seeds)
    assert stats["mean"] == 0.0
    assert stats["std"] == 0.0
    assert stats["cv_percent"] == float("inf")


def test_save_seed_study_creates_file(tmp_path):
    """Should save YAML file and return correct path."""
    study = {
        "experiment_id": "exp-042",
        "timestamp": "2026-04-01T00:00:00Z",
        "metric": "accuracy",
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
    }
    filepath = save_seed_study(study, str(tmp_path))
    assert filepath.exists()
    assert filepath.name == "exp-042-seeds.yaml"

    # Verify YAML roundtrip
    with open(filepath) as f:
        loaded = yaml.safe_load(f)
    assert loaded["experiment_id"] == "exp-042"
    assert loaded["mean"] == 0.855


def test_save_seed_study_creates_directory(tmp_path):
    """Should create output directory if it doesn't exist."""
    nested = tmp_path / "deep" / "nested"
    study = {"experiment_id": "exp-001", "mean": 0.85}
    filepath = save_seed_study(study, str(nested))
    assert filepath.exists()


def test_format_report_lower_is_better():
    """Report should indicate lower is better for loss metrics."""
    study = {
        "experiment_id": "exp-001",
        "metric": "mse",
        "lower_is_better": True,
        "seeds_run": [42, 123, 456],
        "results": [0.10, 0.11, 0.09],
        "mean": 0.10,
        "std": 0.01,
        "ci_95": [0.07, 0.13],
        "cv_percent": 10.0,
        "seed_sensitive": True,
        "best_seed": 456,
        "best_value": 0.09,
        "worst_seed": 123,
        "worst_value": 0.11,
        "range": 0.02,
    }
    report = format_seed_report(study)
    assert "lower" in report.lower()


def test_format_report_table_has_all_seeds():
    """Report table should include all seeds."""
    n = 7
    study = {
        "experiment_id": "exp-001",
        "metric": "accuracy",
        "lower_is_better": False,
        "seeds_run": list(range(n)),
        "results": [0.80 + i * 0.01 for i in range(n)],
        "mean": 0.83,
        "std": 0.02,
        "ci_95": [0.80, 0.86],
        "cv_percent": 2.4,
        "seed_sensitive": False,
        "best_seed": n - 1,
        "best_value": 0.80 + (n - 1) * 0.01,
        "worst_seed": 0,
        "worst_value": 0.80,
        "range": (n - 1) * 0.01,
    }
    report = format_seed_report(study)
    for seed in range(n):
        assert str(seed) in report
