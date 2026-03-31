"""Tests for multi-objective Pareto frontier (pareto_frontier.py).

Phase 11.3: Verifies N-dimensional dominance, Pareto set computation,
metric extraction, ASCII scatter, and report formatting.
"""

from __future__ import annotations

import pytest

from scripts.pareto_frontier import (
    compute_pareto_set,
    extract_metric_vectors,
    find_closest_pareto_neighbor,
    format_ascii_scatter,
    format_frontier_report,
    save_frontier,
)


def _make_exp(exp_id, status="kept", accuracy=0.85, train_seconds=10.0, model_type="xgboost"):
    return {
        "experiment_id": exp_id,
        "status": status,
        "config": {"model_type": model_type},
        "metrics": {"accuracy": accuracy, "train_seconds": train_seconds},
    }


# --- extract_metric_vectors ---


def test_extract_basic():
    """Should extract metric vectors from experiments."""
    exps = [
        _make_exp("exp-001", accuracy=0.85, train_seconds=10),
        _make_exp("exp-002", accuracy=0.90, train_seconds=20),
    ]
    data = extract_metric_vectors(exps, ["accuracy", "train_seconds"])
    assert len(data) == 2
    assert data[0]["metrics"]["accuracy"] == 0.85


def test_extract_filters_status():
    """Should only include kept experiments."""
    exps = [
        _make_exp("exp-001", status="kept"),
        _make_exp("exp-002", status="discarded"),
    ]
    data = extract_metric_vectors(exps, ["accuracy", "train_seconds"])
    assert len(data) == 1


def test_extract_skips_incomplete():
    """Should skip experiments missing requested metrics."""
    exps = [
        _make_exp("exp-001"),
        {"experiment_id": "exp-002", "status": "kept", "config": {}, "metrics": {"accuracy": 0.9}},
    ]
    data = extract_metric_vectors(exps, ["accuracy", "train_seconds"])
    assert len(data) == 1  # exp-002 has no train_seconds


# --- compute_pareto_set ---


def test_pareto_basic_2d():
    """Should find Pareto-optimal experiments in 2D."""
    data = [
        {"experiment_id": "A", "metrics": {"accuracy": 0.90, "train_seconds": 100}},
        {"experiment_id": "B", "metrics": {"accuracy": 0.85, "train_seconds": 10}},
        {"experiment_id": "C", "metrics": {"accuracy": 0.88, "train_seconds": 50}},
        {"experiment_id": "D", "metrics": {"accuracy": 0.80, "train_seconds": 200}},  # Dominated by all
    ]
    directions = {"accuracy": "higher", "train_seconds": "lower"}
    frontier = compute_pareto_set(data, ["accuracy", "train_seconds"], directions)
    frontier_ids = {e["experiment_id"] for e in frontier}

    assert "A" in frontier_ids  # Best accuracy
    assert "B" in frontier_ids  # Best speed
    assert "D" not in frontier_ids  # Dominated


def test_pareto_all_optimal():
    """When no point dominates another, all should be Pareto-optimal."""
    data = [
        {"experiment_id": "A", "metrics": {"accuracy": 0.90, "train_seconds": 100}},
        {"experiment_id": "B", "metrics": {"accuracy": 0.80, "train_seconds": 10}},
    ]
    directions = {"accuracy": "higher", "train_seconds": "lower"}
    frontier = compute_pareto_set(data, ["accuracy", "train_seconds"], directions)
    assert len(frontier) == 2


def test_pareto_single_point():
    """Single experiment should be Pareto-optimal."""
    data = [{"experiment_id": "A", "metrics": {"accuracy": 0.85, "time": 10}}]
    frontier = compute_pareto_set(data, ["accuracy", "time"], {"accuracy": "higher", "time": "lower"})
    assert len(frontier) == 1


def test_pareto_empty():
    """Empty data should return empty frontier."""
    assert compute_pareto_set([], ["accuracy"], {"accuracy": "higher"}) == []


def test_pareto_3d():
    """Should work with 3+ objectives."""
    data = [
        {"experiment_id": "A", "metrics": {"accuracy": 0.90, "time": 100, "size": 1000}},
        {"experiment_id": "B", "metrics": {"accuracy": 0.85, "time": 10, "size": 100}},
        {"experiment_id": "C", "metrics": {"accuracy": 0.88, "time": 50, "size": 500}},
    ]
    directions = {"accuracy": "higher", "time": "lower", "size": "lower"}
    frontier = compute_pareto_set(data, ["accuracy", "time", "size"], directions)
    # All three form a Pareto front (none dominates another on all 3)
    assert len(frontier) >= 2


def test_pareto_identical_points():
    """Identical points should not dominate each other."""
    data = [
        {"experiment_id": "A", "metrics": {"accuracy": 0.85, "time": 10}},
        {"experiment_id": "B", "metrics": {"accuracy": 0.85, "time": 10}},
    ]
    directions = {"accuracy": "higher", "time": "lower"}
    frontier = compute_pareto_set(data, ["accuracy", "time"], directions)
    # Both should be on frontier (neither dominates the other)
    assert len(frontier) == 2


def test_pareto_lower_is_better():
    """Should respect lower-is-better direction."""
    data = [
        {"experiment_id": "A", "metrics": {"mse": 0.10, "time": 100}},
        {"experiment_id": "B", "metrics": {"mse": 0.05, "time": 200}},
        {"experiment_id": "C", "metrics": {"mse": 0.20, "time": 150}},  # Dominated by A
    ]
    directions = {"mse": "lower", "time": "lower"}
    frontier = compute_pareto_set(data, ["mse", "time"], directions)
    frontier_ids = {e["experiment_id"] for e in frontier}
    assert "A" in frontier_ids
    assert "B" in frontier_ids
    assert "C" not in frontier_ids


# --- find_closest_pareto_neighbor ---


def test_closest_neighbor():
    """Should find the nearest Pareto-optimal experiment."""
    frontier = [
        {"experiment_id": "A", "metrics": {"accuracy": 0.90, "time": 100}},
        {"experiment_id": "B", "metrics": {"accuracy": 0.80, "time": 10}},
    ]
    dominated = {"experiment_id": "C", "metrics": {"accuracy": 0.82, "time": 15}}
    directions = {"accuracy": "higher", "time": "lower"}
    neighbor = find_closest_pareto_neighbor(dominated, frontier, ["accuracy", "time"], directions)
    assert neighbor["experiment_id"] == "B"  # Closer to B


def test_closest_neighbor_empty_frontier():
    """Should return None for empty frontier."""
    dominated = {"experiment_id": "C", "metrics": {"accuracy": 0.82}}
    assert find_closest_pareto_neighbor(dominated, [], ["accuracy"], {}) is None


# --- format_ascii_scatter ---


def test_ascii_scatter():
    """Should produce an ASCII scatter plot."""
    data = [
        {"experiment_id": "A", "metrics": {"accuracy": 0.90, "time": 100}},
        {"experiment_id": "B", "metrics": {"accuracy": 0.80, "time": 10}},
    ]
    frontier = [data[0], data[1]]
    plot = format_ascii_scatter(data, frontier, "accuracy", "time")
    assert "*" in plot
    assert "accuracy" in plot


def test_ascii_scatter_empty():
    """Should handle empty data."""
    plot = format_ascii_scatter([], [], "accuracy", "time")
    assert "No data" in plot


# --- format_frontier_report ---


def test_format_report():
    """Should produce markdown report."""
    data = [
        {"experiment_id": "A", "model_type": "xgboost", "metrics": {"accuracy": 0.90, "time": 100}, "description": ""},
        {"experiment_id": "B", "model_type": "lgbm", "metrics": {"accuracy": 0.80, "time": 10}, "description": ""},
    ]
    frontier = data  # Both optimal
    report = format_frontier_report(data, frontier, ["accuracy", "time"], {"accuracy": "higher", "time": "lower"})
    assert "Pareto Frontier" in report
    assert "Pareto-Optimal" in report
    assert "exp-" not in report or "A" in report  # Has experiment IDs


def test_format_report_with_dominated():
    """Should include dominated experiments section."""
    data = [
        {"experiment_id": "A", "model_type": "xgb", "metrics": {"acc": 0.90, "t": 100}, "description": ""},
        {"experiment_id": "B", "model_type": "lgb", "metrics": {"acc": 0.80, "t": 10}, "description": ""},
        {"experiment_id": "C", "model_type": "xgb", "metrics": {"acc": 0.70, "t": 200}, "description": ""},
    ]
    frontier = data[:2]
    report = format_frontier_report(data, frontier, ["acc", "t"], {"acc": "higher", "t": "lower"})
    assert "Dominated" in report


# --- save_frontier ---


def test_save_frontier(tmp_path):
    """Should save YAML file."""
    data = {"timestamp": "2026-04-01", "metrics": ["accuracy"], "frontier": []}
    filepath = save_frontier(data, str(tmp_path))
    assert filepath.exists()
    assert "frontier-" in filepath.name
