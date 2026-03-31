"""Edge case tests for pareto_frontier.py."""

from __future__ import annotations

import pytest

from scripts.pareto_frontier import (
    compute_pareto_set,
    extract_metric_vectors,
    find_closest_pareto_neighbor,
    format_ascii_scatter,
)


def test_pareto_many_points():
    """Should handle many experiments efficiently."""
    data = [
        {"experiment_id": f"exp-{i:03d}", "metrics": {"acc": 0.5 + i * 0.01, "time": 100 - i}}
        for i in range(50)
    ]
    directions = {"acc": "higher", "time": "lower"}
    frontier = compute_pareto_set(data, ["acc", "time"], directions)
    # Only the last point (best acc, lowest time) should dominate
    assert len(frontier) >= 1


def test_pareto_all_same():
    """All identical points should all be on frontier."""
    data = [
        {"experiment_id": f"exp-{i}", "metrics": {"acc": 0.85, "time": 10}}
        for i in range(5)
    ]
    directions = {"acc": "higher", "time": "lower"}
    frontier = compute_pareto_set(data, ["acc", "time"], directions)
    assert len(frontier) == 5


def test_pareto_single_metric():
    """Should work with single metric (degenerates to max/min)."""
    data = [
        {"experiment_id": "A", "metrics": {"accuracy": 0.90}},
        {"experiment_id": "B", "metrics": {"accuracy": 0.85}},
        {"experiment_id": "C", "metrics": {"accuracy": 0.80}},
    ]
    directions = {"accuracy": "higher"}
    frontier = compute_pareto_set(data, ["accuracy"], directions)
    assert len(frontier) == 1
    assert frontier[0]["experiment_id"] == "A"


def test_extract_non_numeric_metric():
    """Should skip experiments where metric value is non-numeric."""
    exps = [
        {"experiment_id": "exp-001", "status": "kept", "config": {},
         "metrics": {"accuracy": "N/A", "train_seconds": 10}},
        {"experiment_id": "exp-002", "status": "kept", "config": {},
         "metrics": {"accuracy": 0.85, "train_seconds": 10}},
    ]
    data = extract_metric_vectors(exps, ["accuracy", "train_seconds"])
    assert len(data) == 1
    assert data[0]["experiment_id"] == "exp-002"


def test_closest_neighbor_equidistant():
    """Should return one of the equidistant neighbors."""
    frontier = [
        {"experiment_id": "A", "metrics": {"x": 0, "y": 0}},
        {"experiment_id": "B", "metrics": {"x": 10, "y": 10}},
    ]
    dominated = {"experiment_id": "C", "metrics": {"x": 5, "y": 5}}
    neighbor = find_closest_pareto_neighbor(dominated, frontier, ["x", "y"], {"x": "higher", "y": "higher"})
    assert neighbor is not None


def test_ascii_scatter_single_point():
    """Should handle single data point."""
    data = [{"experiment_id": "A", "metrics": {"x": 1.0, "y": 2.0}}]
    frontier = data
    plot = format_ascii_scatter(data, frontier, "x", "y", width=20, height=10)
    assert "*" in plot


def test_ascii_scatter_same_values():
    """Should handle all identical values."""
    data = [
        {"experiment_id": f"exp-{i}", "metrics": {"x": 0.5, "y": 0.5}}
        for i in range(3)
    ]
    frontier = data
    plot = format_ascii_scatter(data, frontier, "x", "y")
    assert "*" in plot


def test_extract_no_kept_experiments():
    """Should return empty for all-discarded experiments."""
    exps = [
        {"experiment_id": "exp-001", "status": "discarded", "config": {},
         "metrics": {"accuracy": 0.85, "train_seconds": 10}},
    ]
    data = extract_metric_vectors(exps, ["accuracy", "train_seconds"])
    assert len(data) == 0
