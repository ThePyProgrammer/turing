"""Tests for experiment_queue.py."""

from __future__ import annotations

import yaml
import pytest

from scripts.experiment_queue import (
    add_to_queue,
    clear_queue,
    format_batch_summary,
    format_queue_list,
    get_next_queue_id,
    load_queue,
    save_queue,
    sort_queue,
)


def test_add_to_queue(tmp_path):
    """Should add an item and persist."""
    qp = str(tmp_path / "queue.yaml")
    item = add_to_queue("try LightGBM", "high", queue_path=qp)
    assert item["id"] == "q-001"
    assert item["priority"] == "high"
    assert item["status"] == "queued"

    queue = load_queue(qp)
    assert len(queue) == 1


def test_add_multiple(tmp_path):
    """Should generate sequential IDs."""
    qp = str(tmp_path / "queue.yaml")
    add_to_queue("first", queue_path=qp)
    item2 = add_to_queue("second", queue_path=qp)
    assert item2["id"] == "q-002"


def test_add_with_dependency(tmp_path):
    """Should store dependency link."""
    qp = str(tmp_path / "queue.yaml")
    add_to_queue("first", queue_path=qp)
    item2 = add_to_queue("second", after="q-001", queue_path=qp)
    assert item2["depends_on"] == "q-001"


def test_get_next_id():
    """Should increment from highest existing ID."""
    queue = [{"id": "q-001"}, {"id": "q-005"}, {"id": "q-003"}]
    assert get_next_queue_id(queue) == "q-006"


def test_get_next_id_empty():
    """Should start at q-001 for empty queue."""
    assert get_next_queue_id([]) == "q-001"


def test_sort_by_priority():
    """Should sort by priority (critical first)."""
    queue = [
        {"id": "q-001", "priority": "low", "status": "queued", "depends_on": None},
        {"id": "q-002", "priority": "critical", "status": "queued", "depends_on": None},
        {"id": "q-003", "priority": "high", "status": "queued", "depends_on": None},
    ]
    sorted_q = sort_queue(queue)
    assert sorted_q[0]["id"] == "q-002"  # critical first


def test_sort_respects_dependencies():
    """Should resolve dependencies before priority."""
    queue = [
        {"id": "q-001", "priority": "low", "status": "queued", "depends_on": None},
        {"id": "q-002", "priority": "critical", "status": "queued", "depends_on": "q-001"},
    ]
    sorted_q = sort_queue(queue)
    ids = [q["id"] for q in sorted_q]
    assert ids.index("q-001") < ids.index("q-002")


def test_sort_skips_completed():
    """Should only sort queued items."""
    queue = [
        {"id": "q-001", "priority": "medium", "status": "completed", "depends_on": None},
        {"id": "q-002", "priority": "medium", "status": "queued", "depends_on": None},
    ]
    sorted_q = sort_queue(queue)
    assert len(sorted_q) == 1
    assert sorted_q[0]["id"] == "q-002"


def test_clear_queue(tmp_path):
    """Should remove all queued items."""
    qp = str(tmp_path / "queue.yaml")
    add_to_queue("a", queue_path=qp)
    add_to_queue("b", queue_path=qp)
    n = clear_queue(qp)
    assert n == 2
    assert load_queue(qp) == []


def test_load_missing_queue():
    """Should return empty for missing file."""
    assert load_queue("/nonexistent/queue.yaml") == []


def test_save_and_load(tmp_path):
    """Should roundtrip through YAML."""
    qp = str(tmp_path / "queue.yaml")
    items = [{"id": "q-001", "description": "test", "status": "queued"}]
    save_queue(items, qp)
    loaded = load_queue(qp)
    assert loaded[0]["id"] == "q-001"


def test_format_queue_list():
    """Should produce markdown table."""
    queue = [
        {"id": "q-001", "priority": "high", "status": "queued",
         "description": "try LightGBM", "depends_on": None},
        {"id": "q-002", "priority": "medium", "status": "completed",
         "description": "deeper trees", "depends_on": "q-001"},
    ]
    table = format_queue_list(queue)
    assert "q-001" in table
    assert "LightGBM" in table


def test_format_queue_empty():
    """Should show empty message."""
    assert "empty" in format_queue_list([]).lower()


def test_format_batch_summary():
    """Should produce readable summary."""
    summary = {
        "status": "completed",
        "total": 3,
        "completed": 2,
        "failed": 1,
        "skipped": 0,
        "results": [
            {"id": "q-001", "description": "A", "status": "completed", "error": None},
            {"id": "q-002", "description": "B", "status": "failed", "error": "oom"},
        ],
    }
    report = format_batch_summary(summary)
    assert "completed" in report.lower()
    assert "q-001" in report
