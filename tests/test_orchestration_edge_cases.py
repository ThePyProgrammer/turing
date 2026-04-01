"""Edge case tests for Phase 15 orchestration scripts."""

from __future__ import annotations

import yaml
import pytest

from scripts.experiment_queue import (
    add_to_queue,
    clear_queue,
    load_queue,
    sort_queue,
)
from scripts.smart_retry import (
    apply_config_changes,
    classify_failure,
)
from scripts.fork_experiment import (
    create_branch,
    determine_winner,
    format_fork_report,
)


# --- Queue edge cases ---


def test_queue_circular_deps():
    """Should handle circular dependencies without infinite loop."""
    queue = [
        {"id": "q-001", "priority": "medium", "status": "queued", "depends_on": "q-002"},
        {"id": "q-002", "priority": "medium", "status": "queued", "depends_on": "q-001"},
    ]
    sorted_q = sort_queue(queue)
    assert len(sorted_q) == 2  # Should not hang


def test_queue_deep_dependency_chain():
    """Should resolve long dependency chains."""
    queue = [
        {"id": f"q-{i:03d}", "priority": "medium", "status": "queued",
         "depends_on": f"q-{i-1:03d}" if i > 1 else None}
        for i in range(1, 11)
    ]
    sorted_q = sort_queue(queue)
    ids = [q["id"] for q in sorted_q]
    # q-001 must come before q-002, etc.
    for i in range(len(ids) - 1):
        assert int(ids[i].split("-")[1]) <= int(ids[i+1].split("-")[1])


def test_queue_missing_dependency():
    """Should handle dependency on non-existent item."""
    queue = [
        {"id": "q-001", "priority": "medium", "status": "queued", "depends_on": "q-999"},
    ]
    sorted_q = sort_queue(queue)
    assert len(sorted_q) == 1  # Should still include it


def test_clear_preserves_completed(tmp_path):
    """Clear should only remove queued items, not completed."""
    qp = str(tmp_path / "queue.yaml")
    items = [
        {"id": "q-001", "status": "completed", "description": "done"},
        {"id": "q-002", "status": "queued", "description": "pending"},
    ]
    with open(qp, "w") as f:
        yaml.dump(items, f)
    n = clear_queue(qp)
    assert n == 1
    remaining = load_queue(qp)
    assert len(remaining) == 1
    assert remaining[0]["status"] == "completed"


def test_add_with_hypothesis_link(tmp_path):
    """Should store hypothesis ID."""
    qp = str(tmp_path / "queue.yaml")
    item = add_to_queue("test", hypothesis_id="hyp-001", queue_path=qp)
    assert item["hypothesis_id"] == "hyp-001"


# --- Retry edge cases ---


def test_classify_multiple_patterns():
    """Should match first matching pattern."""
    # Contains both OOM and NaN patterns — OOM should match first (listed first)
    result = classify_failure("CUDA out of memory and also loss is NaN")
    assert result["failure_type"] == "oom"


def test_apply_float_divide():
    """Should handle float division."""
    config = {"model": {"hyperparams": {"learning_rate": 0.01}}}
    result = apply_config_changes(config, {"learning_rate": "//10"})
    assert result["model"]["hyperparams"]["learning_rate"] == pytest.approx(0.001, abs=0.0001)


def test_apply_empty_changes():
    """Should handle empty changes dict."""
    config = {"model": {"hyperparams": {"batch_size": 32}}}
    result = apply_config_changes(config, {})
    assert result["model"]["hyperparams"]["batch_size"] == 32


def test_apply_creates_hyperparams():
    """Should create hyperparams dict if missing."""
    config = {}
    result = apply_config_changes(config, {"gradient_clip": 1.0})
    assert result["model"]["hyperparams"]["gradient_clip"] == 1.0


# --- Fork edge cases ---


def test_winner_tied_metric():
    """Should return one winner when metrics are tied."""
    branches = [
        {"branch_id": "b-1", "status": "completed", "metrics": {"accuracy": 0.85}},
        {"branch_id": "b-2", "status": "completed", "metrics": {"accuracy": 0.85}},
    ]
    winner = determine_winner(branches, "accuracy", lower_is_better=False)
    assert winner is not None  # Should pick one


def test_winner_missing_metric():
    """Should skip branches without the metric."""
    branches = [
        {"branch_id": "b-1", "status": "completed", "metrics": {"f1": 0.80}},
        {"branch_id": "b-2", "status": "completed", "metrics": {"accuracy": 0.85}},
    ]
    winner = determine_winner(branches, "accuracy", lower_is_better=False)
    assert winner["branch_id"] == "b-2"


def test_format_many_branches():
    """Should format many branches."""
    branches = [
        {"branch_id": f"b-{i}", "description": f"Branch {i}", "status": "completed",
         "metrics": {"accuracy": 0.80 + i * 0.01}}
        for i in range(5)
    ]
    winner = branches[-1]
    report = format_fork_report("exp-001", branches, winner, "accuracy")
    assert "WINNER" in report
    for i in range(5):
        assert f"b-{i}" in report


def test_create_branch_preserves_parent():
    """Branch should reference correct parent."""
    parent = {"experiment_id": "exp-100", "config": {}, "metrics": {}}
    branch = create_branch(parent, "test", 0)
    assert branch["parent_id"] == "exp-100"
