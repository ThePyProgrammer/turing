"""Tests for long-term trend analysis (trend_analysis.py). Phase 24.1."""
from __future__ import annotations
import pytest
from scripts.trend_analysis import (
    compute_improvement_trajectory, compute_velocity, compute_family_roi,
    detect_diminishing_returns, detect_phase_transitions, format_trend_report,
)

EXPS = [
    {"experiment_id": f"exp-{i:03d}", "status": "kept", "metrics": {"accuracy": 0.70 + i * 0.01},
     "config": {"model_type": "xgboost" if i < 5 else "lightgbm"},
     "timestamp": f"2026-03-{i+1:02d}T00:00:00"}
    for i in range(10)
]

def test_trajectory():
    traj = compute_improvement_trajectory(EXPS, "accuracy", False)
    assert len(traj) > 0
    assert traj[-1]["best_so_far"] >= traj[0]["best_so_far"]

def test_trajectory_empty():
    assert compute_improvement_trajectory([], "accuracy", False) == []

def test_velocity():
    traj = compute_improvement_trajectory(EXPS, "accuracy", False)
    vel = compute_velocity(traj, window=3)
    assert len(vel) > 0
    assert all("velocity" in v for v in vel)

def test_velocity_empty():
    assert compute_velocity([], window=3) == []

def test_family_roi():
    roi = compute_family_roi(EXPS, "accuracy", False)
    assert len(roi) > 0
    assert all("family" in r for r in roi)
    assert all("roi" in r for r in roi)

def test_family_roi_empty():
    assert compute_family_roi([], "accuracy", False) == []

def test_diminishing_returns():
    traj = compute_improvement_trajectory(EXPS, "accuracy", False)
    dr = detect_diminishing_returns(traj, False)
    assert "detected" in dr or "status" in dr

def test_diminishing_returns_empty():
    dr = detect_diminishing_returns([], False)
    assert dr.get("detected") is not None or dr.get("status") == "insufficient_data"

def test_phase_transitions():
    traj = compute_improvement_trajectory(EXPS, "accuracy", False)
    transitions = detect_phase_transitions(traj)
    assert isinstance(transitions, list)

def test_format_error():
    assert "ERROR" in format_trend_report({"error": "fail"})
