"""Tests for changelog generation (generate_changelog.py). Phase 25.3."""
from __future__ import annotations
import pytest
from scripts.generate_changelog import (
    compute_trajectory, detect_version_boundaries, group_into_versions,
    format_changelog_report,
)

EXPS = [
    {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.834},
     "config": {"model_type": "logistic_regression"}, "timestamp": "2026-03-01T00:00:00"},
    {"experiment_id": "exp-020", "status": "kept", "metrics": {"accuracy": 0.864},
     "config": {"model_type": "xgboost"}, "timestamp": "2026-03-10T00:00:00"},
    {"experiment_id": "exp-089", "status": "kept", "metrics": {"accuracy": 0.883},
     "config": {"model_type": "lightgbm"}, "timestamp": "2026-03-25T00:00:00"},
]

def test_compute_trajectory():
    traj = compute_trajectory(EXPS, "accuracy", False)
    assert len(traj) > 0

def test_compute_trajectory_empty():
    assert compute_trajectory([], "accuracy", False) == []

def test_detect_boundaries():
    traj = compute_trajectory(EXPS, "accuracy", False)
    boundaries = detect_version_boundaries(traj)
    assert isinstance(boundaries, list)

def test_group_into_versions():
    traj = compute_trajectory(EXPS, "accuracy", False)
    boundaries = detect_version_boundaries(traj)
    versions = group_into_versions(traj, boundaries)
    assert len(versions) >= 1

def test_format_report():
    report = {"versions": [{"version": 1}], "total_improvement": 0.05, "audience": "technical"}
    text = format_changelog_report(report)
    assert isinstance(text, str)
