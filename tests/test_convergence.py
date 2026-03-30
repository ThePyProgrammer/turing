"""Tests for convergence detection (check_convergence.py).

Tier 1: These tests verify ADR-0006 — the mechanism that decides
when autonomous training stops. A false positive means premature stop
(wasted opportunity). A false negative means infinite loop (wasted compute).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.check_convergence import (
    check_convergence,
    compute_relative_improvement,
    load_convergence_config,
    load_kept_experiments,
)


# --- compute_relative_improvement ---


def test_improvement_higher_is_better():
    """Higher current = positive improvement when higher is better."""
    result = compute_relative_improvement(current=0.90, prior_best=0.85, lower_is_better=False)
    assert result > 0
    assert result == pytest.approx((0.90 - 0.85) / 0.85, abs=0.0001)


def test_improvement_lower_is_better():
    """Lower current = positive improvement when lower is better."""
    result = compute_relative_improvement(current=0.10, prior_best=0.15, lower_is_better=True)
    assert result > 0
    assert result == pytest.approx((0.15 - 0.10) / 0.15, abs=0.0001)


def test_regression_higher_is_better():
    """Lower current = negative improvement when higher is better."""
    result = compute_relative_improvement(current=0.80, prior_best=0.85, lower_is_better=False)
    assert result < 0


def test_no_change():
    """Same value = zero improvement."""
    result = compute_relative_improvement(current=0.85, prior_best=0.85, lower_is_better=False)
    assert result == 0.0


def test_prior_best_zero_nonzero_current():
    """Prior best of zero with nonzero current should return 1.0 (infinite improvement)."""
    result = compute_relative_improvement(current=0.5, prior_best=0.0, lower_is_better=False)
    assert result == 1.0


def test_prior_best_zero_zero_current():
    """Both zero should return 0.0 (no improvement)."""
    result = compute_relative_improvement(current=0.0, prior_best=0.0, lower_is_better=False)
    assert result == 0.0


# --- check_convergence ---


def test_not_enough_experiments():
    """Fewer experiments than patience should not converge."""
    experiments = [{"id": "exp-001", "value": 0.85}]
    converged, count, msg = check_convergence(experiments, patience=3, improvement_threshold=0.005, lower_is_better=False)
    assert converged is False
    assert "Only 1 experiments" in msg


def test_convergence_detected():
    """N consecutive non-improvements should trigger convergence."""
    experiments = [
        {"id": "exp-001", "value": 0.85},
        {"id": "exp-002", "value": 0.86},  # improvement
        {"id": "exp-003", "value": 0.860},  # non-improvement (same)
        {"id": "exp-004", "value": 0.861},  # non-improvement (< 0.5% relative)
        {"id": "exp-005", "value": 0.859},  # non-improvement (regression)
    ]
    converged, count, msg = check_convergence(experiments, patience=3, improvement_threshold=0.005, lower_is_better=False)
    assert converged is True
    assert "CONVERGED" in msg


def test_no_convergence_with_improvements():
    """Consistent improvements should not trigger convergence."""
    experiments = [
        {"id": "exp-001", "value": 0.80},
        {"id": "exp-002", "value": 0.85},  # +6.25% improvement
        {"id": "exp-003", "value": 0.90},  # +5.88% improvement
        {"id": "exp-004", "value": 0.95},  # +5.56% improvement
    ]
    converged, count, msg = check_convergence(experiments, patience=3, improvement_threshold=0.005, lower_is_better=False)
    assert converged is False
    assert "Not converged" in msg


def test_convergence_lower_is_better():
    """Convergence should work with lower-is-better metrics (MAE, MSE)."""
    experiments = [
        {"id": "exp-001", "value": 0.50},
        {"id": "exp-002", "value": 0.30},  # improvement (lower)
        {"id": "exp-003", "value": 0.30},  # non-improvement
        {"id": "exp-004", "value": 0.31},  # non-improvement (worse)
        {"id": "exp-005", "value": 0.30},  # non-improvement (not enough gain)
    ]
    converged, count, msg = check_convergence(experiments, patience=3, improvement_threshold=0.005, lower_is_better=True)
    assert converged is True


def test_convergence_patience_1():
    """Patience of 1 should converge after a single non-improvement."""
    experiments = [
        {"id": "exp-001", "value": 0.90},
        {"id": "exp-002", "value": 0.89},  # regression
    ]
    converged, count, msg = check_convergence(experiments, patience=1, improvement_threshold=0.005, lower_is_better=False)
    assert converged is True


def test_convergence_exact_threshold():
    """Improvement exactly at threshold should NOT count as non-improvement."""
    # 0.005 relative improvement = exactly at threshold
    # 0.85 * 1.005 = 0.85425
    experiments = [
        {"id": "exp-001", "value": 0.850},
        {"id": "exp-002", "value": 0.855},  # 0.588% > 0.5% — improvement
        {"id": "exp-003", "value": 0.860},  # 0.585% > 0.5% — improvement
        {"id": "exp-004", "value": 0.865},  # 0.581% > 0.5% — improvement
    ]
    converged, count, msg = check_convergence(experiments, patience=3, improvement_threshold=0.005, lower_is_better=False)
    assert converged is False


# --- load_convergence_config ---


def test_load_config_defaults_on_missing_file():
    """Missing config file should return defaults, not crash."""
    cfg = load_convergence_config("/nonexistent/path/config.yaml")
    assert cfg["patience"] == 3
    assert cfg["improvement_threshold"] == 0.005
    assert cfg["primary_metric"] == "accuracy"
    assert cfg["lower_is_better"] is False


def test_load_config_from_file(tmp_path: Path):
    """Config values should be read from YAML file."""
    config = {
        "convergence": {"patience": 5, "improvement_threshold": 0.01},
        "evaluation": {"primary_metric": "mae", "lower_is_better": True},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    cfg = load_convergence_config(str(config_path))
    assert cfg["patience"] == 5
    assert cfg["improvement_threshold"] == 0.01
    assert cfg["primary_metric"] == "mae"
    assert cfg["lower_is_better"] is True


# --- load_kept_experiments ---


def test_load_experiments_empty_log(tmp_log: Path):
    """Empty log should return empty list."""
    result = load_kept_experiments(str(tmp_log), "accuracy")
    assert result == []


def test_load_experiments_filters_discarded(tmp_log: Path):
    """Only 'kept' experiments should be loaded."""
    import json

    entries = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.85}},
        {"experiment_id": "exp-002", "status": "discarded", "metrics": {"accuracy": 0.80}},
        {"experiment_id": "exp-003", "status": "kept", "metrics": {"accuracy": 0.90}},
    ]
    with open(tmp_log, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = load_kept_experiments(str(tmp_log), "accuracy")
    assert len(result) == 2
    assert result[0]["value"] == 0.85
    assert result[1]["value"] == 0.90


def test_load_experiments_handles_corrupted_lines(tmp_log: Path):
    """Corrupted JSONL lines should be skipped, not crash."""
    import json

    with open(tmp_log, "w") as f:
        f.write(json.dumps({"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.85}}) + "\n")
        f.write("this is not json\n")
        f.write("{bad json\n")
        f.write(json.dumps({"experiment_id": "exp-002", "status": "kept", "metrics": {"accuracy": 0.90}}) + "\n")

    result = load_kept_experiments(str(tmp_log), "accuracy")
    assert len(result) == 2


def test_load_experiments_missing_metric(tmp_log: Path):
    """Experiments without the requested metric should be skipped."""
    import json

    entries = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.85}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"f1": 0.80}},  # no accuracy
    ]
    with open(tmp_log, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = load_kept_experiments(str(tmp_log), "accuracy")
    assert len(result) == 1


def test_load_experiments_nonexistent_file():
    """Nonexistent log file should return empty list, not crash."""
    result = load_kept_experiments("/nonexistent/log.jsonl", "accuracy")
    assert result == []
