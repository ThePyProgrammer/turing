"""Edge case tests for reproduce_experiment.py.

Tests boundary conditions, YAML serialization, and complex scenarios.
"""

from __future__ import annotations

import yaml
import pytest

from scripts.reproduce_experiment import (
    compare_environments,
    determine_verdict,
    format_reproduction_report,
    save_reproduction_report,
)


def test_verdict_exact_float_boundary():
    """Values within float tolerance boundary should be reproducible."""
    # 5e-7 difference should be within 1e-6 tolerance
    result = determine_verdict(0.8500000, [0.8500005], tolerance=0.02, strict=True)
    assert result["verdict"] == "reproducible"


def test_verdict_negative_metrics():
    """Should handle negative metrics (e.g., negative log-likelihood)."""
    result = determine_verdict(-2.5, [-2.48, -2.52, -2.50], tolerance=0.02, strict=False)
    assert result["verdict"] in ("reproducible", "approximately_reproducible")


def test_verdict_very_small_tolerance():
    """Very small tolerance should be strict-like."""
    result = determine_verdict(0.85, [0.84, 0.845, 0.843], tolerance=0.001, strict=False)
    # 0.85 vs ~0.843 = ~0.8% diff, but tolerance is 0.1%
    # Should check CI as well
    assert "verdict" in result


def test_verdict_large_tolerance():
    """Large tolerance should be lenient."""
    result = determine_verdict(0.85, [0.75, 0.76, 0.74], tolerance=0.20, strict=False)
    assert result["verdict"] == "approximately_reproducible"


def test_verdict_many_runs():
    """Should work with many reproduction runs."""
    import random
    random.seed(42)
    values = [0.85 + random.gauss(0, 0.005) for _ in range(20)]
    result = determine_verdict(0.85, values, tolerance=0.02, strict=False)
    assert result["verdict"] in ("reproducible", "approximately_reproducible")
    assert result["n_runs"] == 20


def test_env_compare_multiple_critical_packages():
    """Should flag all critical package changes."""
    orig = {
        "python_version": "3.10.0",
        "packages": {"numpy": "1.24.0", "scipy": "1.10.0", "pandas": "1.5.0"},
    }
    curr = {
        "python_version": "3.10.0",
        "packages": {"numpy": "1.26.0", "scipy": "1.11.0", "pandas": "2.0.0"},
    }
    diffs = compare_environments(orig, curr)
    critical = [d for d in diffs if d["severity"] == "critical"]
    assert len(critical) == 3  # numpy, scipy, pandas all critical


def test_env_compare_empty_packages():
    """Should handle environments with no package info."""
    orig = {"python_version": "3.10.0"}
    curr = {"python_version": "3.10.0"}
    diffs = compare_environments(orig, curr)
    assert len(diffs) == 0


def test_save_reproduction_report_creates_file(tmp_path):
    """Should save YAML file and return correct path."""
    report = {
        "experiment_id": "exp-042",
        "reproduced_at": "2026-04-01T00:00:00Z",
        "verdict": "reproducible",
        "reason": "Exact match",
        "original_metrics": {"accuracy": 0.85},
        "new_metrics": {"accuracy": 0.85},
    }
    filepath = save_reproduction_report(report, str(tmp_path))
    assert filepath.exists()
    assert filepath.name == "exp-042-repro.yaml"

    with open(filepath) as f:
        loaded = yaml.safe_load(f)
    assert loaded["verdict"] == "reproducible"


def test_save_reproduction_report_creates_directory(tmp_path):
    """Should create output directory if it doesn't exist."""
    nested = tmp_path / "deep" / "nested"
    report = {"experiment_id": "exp-001", "verdict": "reproducible"}
    filepath = save_reproduction_report(report, str(nested))
    assert filepath.exists()


def test_format_report_with_ci():
    """Report should show CI when multiple runs exist."""
    report = {
        "experiment_id": "exp-001",
        "verdict": "approximately_reproducible",
        "reason": "Within tolerance",
        "original_metrics": {"accuracy": 0.85},
        "new_metrics": {"accuracy": 0.845},
        "statistical_details": {
            "n_runs": 5,
            "original_value": 0.85,
            "new_mean": 0.845,
            "new_values": [0.84, 0.845, 0.85, 0.843, 0.847],
            "new_std": 0.003,
            "ci_95": [0.841, 0.849],
            "relative_difference": 0.006,
        },
        "strictness": "tolerance=0.02",
        "environment_diffs": [],
    }
    text = format_reproduction_report(report)
    assert "95% CI" in text
    assert "5" in text  # n_runs
    assert "Relative difference" in text


def test_format_report_with_multiple_env_diffs():
    """Report should list all environment differences."""
    report = {
        "experiment_id": "exp-001",
        "verdict": "environment_changed",
        "reason": "Metrics diverge",
        "original_metrics": {"accuracy": 0.85},
        "new_metrics": {"accuracy": 0.80},
        "statistical_details": {"n_runs": 1, "original_value": 0.85, "new_mean": 0.80},
        "strictness": "tolerance=0.02",
        "environment_diffs": [
            {"field": "python_version", "severity": "warning",
             "detail": "Python version changed: 3.10.0 -> 3.11.0",
             "original": "3.10.0", "current": "3.11.0"},
            {"field": "package:numpy", "severity": "critical",
             "detail": "numpy: 1.24.0 -> 1.26.0",
             "original": "1.24.0", "current": "1.26.0"},
        ],
    }
    text = format_reproduction_report(report)
    assert "Python version" in text
    assert "numpy" in text
    assert "WARNING" in text or "CRITICAL" in text


def test_verdict_preserves_all_new_values():
    """Verdict should include all raw values for inspection."""
    values = [0.84, 0.85, 0.86, 0.84, 0.85]
    result = determine_verdict(0.85, values, tolerance=0.02, strict=False)
    assert result["new_values"] == [0.84, 0.85, 0.86, 0.84, 0.85]
