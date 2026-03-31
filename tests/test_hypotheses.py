"""Tests for hypothesis queue management (manage_hypotheses.py).

Verifies the taste-leverage mechanism: hypothesis injection,
prioritization, status transitions, and queue ordering.
"""

from __future__ import annotations

from pathlib import Path

from scripts.manage_hypotheses import (
    add_hypothesis,
    count_by_status,
    get_next_hypothesis,
    get_next_id,
    list_hypotheses,
    load_queue,
    mark_hypothesis,
)


def test_empty_queue(tmp_path: Path):
    """Empty/nonexistent queue should return empty list."""
    queue = load_queue(str(tmp_path / "hypotheses.yaml"))
    assert queue == []


def test_add_hypothesis(tmp_path: Path):
    """Adding a hypothesis should persist to YAML."""
    qpath = str(tmp_path / "hypotheses.yaml")
    hid = add_hypothesis(qpath, "Try LightGBM with dart boosting")
    assert hid == "hyp-001"

    queue = load_queue(qpath)
    assert len(queue) == 1
    assert queue[0]["description"] == "Try LightGBM with dart boosting"
    assert queue[0]["source"] == "human"
    assert queue[0]["status"] == "queued"
    assert queue[0]["priority"] == "high"


def test_sequential_ids(tmp_path: Path):
    """IDs should increment sequentially."""
    qpath = str(tmp_path / "hypotheses.yaml")
    id1 = add_hypothesis(qpath, "First")
    id2 = add_hypothesis(qpath, "Second")
    id3 = add_hypothesis(qpath, "Third")
    assert id1 == "hyp-001"
    assert id2 == "hyp-002"
    assert id3 == "hyp-003"


def test_next_id_with_gaps(tmp_path: Path):
    """Next ID should be max+1 even with gaps."""
    qpath = str(tmp_path / "hypotheses.yaml")
    # Manually create queue with gap
    import yaml
    queue = [
        {"id": "hyp-001", "status": "tested"},
        {"id": "hyp-005", "status": "queued"},
    ]
    with open(qpath, "w") as f:
        yaml.dump(queue, f)
    assert get_next_id(queue) == "hyp-006"


def test_get_next_prioritizes_high(tmp_path: Path):
    """High priority should come before medium and low."""
    qpath = str(tmp_path / "hypotheses.yaml")
    add_hypothesis(qpath, "Low priority idea", priority="low", source="agent")
    add_hypothesis(qpath, "High priority idea", priority="high", source="human")
    add_hypothesis(qpath, "Medium priority idea", priority="medium", source="agent")

    nxt = get_next_hypothesis(qpath)
    assert nxt["description"] == "High priority idea"


def test_get_next_human_before_agent(tmp_path: Path):
    """At same priority, human-sourced should come before agent-sourced."""
    qpath = str(tmp_path / "hypotheses.yaml")
    add_hypothesis(qpath, "Agent idea", priority="high", source="agent")
    add_hypothesis(qpath, "Human idea", priority="high", source="human")

    nxt = get_next_hypothesis(qpath)
    assert nxt["source"] == "human"


def test_get_next_skips_non_queued(tmp_path: Path):
    """Only 'queued' hypotheses should be returned by next."""
    qpath = str(tmp_path / "hypotheses.yaml")
    add_hypothesis(qpath, "Already tested")
    mark_hypothesis(qpath, "hyp-001", "tested")
    add_hypothesis(qpath, "Still queued")

    nxt = get_next_hypothesis(qpath)
    assert nxt["id"] == "hyp-002"


def test_get_next_empty_queue(tmp_path: Path):
    """No queued hypotheses should return None."""
    qpath = str(tmp_path / "hypotheses.yaml")
    assert get_next_hypothesis(qpath) is None


def test_mark_hypothesis(tmp_path: Path):
    """Marking should update status and optional result link."""
    qpath = str(tmp_path / "hypotheses.yaml")
    add_hypothesis(qpath, "Test idea")

    result = mark_hypothesis(qpath, "hyp-001", "tested", result_experiment="exp-005")
    assert result is True

    queue = load_queue(qpath)
    assert queue[0]["status"] == "tested"
    assert queue[0]["result_experiment"] == "exp-005"


def test_mark_nonexistent(tmp_path: Path):
    """Marking nonexistent hypothesis should return False."""
    qpath = str(tmp_path / "hypotheses.yaml")
    result = mark_hypothesis(qpath, "hyp-999", "tested")
    assert result is False


def test_mark_invalid_status(tmp_path: Path):
    """Invalid status should return False."""
    qpath = str(tmp_path / "hypotheses.yaml")
    add_hypothesis(qpath, "Test")
    result = mark_hypothesis(qpath, "hyp-001", "invalid_status")
    assert result is False


def test_list_with_filter(tmp_path: Path):
    """List with status filter should only return matching."""
    qpath = str(tmp_path / "hypotheses.yaml")
    add_hypothesis(qpath, "Queued one")
    add_hypothesis(qpath, "Queued two")
    mark_hypothesis(qpath, "hyp-001", "tested")

    queued = list_hypotheses(qpath, status_filter="queued")
    assert len(queued) == 1
    assert queued[0]["id"] == "hyp-002"

    tested = list_hypotheses(qpath, status_filter="tested")
    assert len(tested) == 1
    assert tested[0]["id"] == "hyp-001"


def test_count_by_status(tmp_path: Path):
    """Count should break down by status."""
    qpath = str(tmp_path / "hypotheses.yaml")
    add_hypothesis(qpath, "One")
    add_hypothesis(qpath, "Two")
    add_hypothesis(qpath, "Three")
    mark_hypothesis(qpath, "hyp-001", "tested")
    mark_hypothesis(qpath, "hyp-002", "dead-end")

    counts = count_by_status(qpath)
    assert counts["tested"] == 1
    assert counts["dead-end"] == 1
    assert counts["queued"] == 1


def test_parent_experiment_recorded(tmp_path: Path):
    """Parent experiment should be stored when provided."""
    qpath = str(tmp_path / "hypotheses.yaml")
    add_hypothesis(qpath, "Based on exp-003 results", parent_experiment="exp-003")

    queue = load_queue(qpath)
    assert queue[0]["parent_experiment"] == "exp-003"
