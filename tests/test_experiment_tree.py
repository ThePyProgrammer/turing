"""Tests for experiment dependency tree (show_experiment_tree.py).

Verifies the reasoning chain visualization that makes the agent's
decision path visible to the human.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.show_experiment_tree import (
    build_tree,
    find_roots,
    load_experiments,
    show_tree,
)


def _make_exp(exp_id: str, parent: str | None = None, status: str = "kept", accuracy: float = 0.85) -> dict:
    entry = {
        "experiment_id": exp_id,
        "status": status,
        "config": {"model_type": "xgboost"},
        "metrics": {"accuracy": accuracy},
        "description": f"Test {exp_id}",
    }
    if parent:
        entry["parent_experiment"] = parent
    return entry


def _write_log(path: Path, experiments: list[dict]) -> None:
    with open(path, "w") as f:
        for e in experiments:
            f.write(json.dumps(e) + "\n")


def test_empty_log(tmp_path: Path):
    """Empty log should produce 'no experiments' message."""
    result = show_tree(str(tmp_path / "nonexistent.jsonl"))
    assert "No experiments" in result


def test_single_experiment(tmp_path: Path):
    """Single experiment should be a root with no children."""
    log = tmp_path / "log.jsonl"
    _write_log(log, [_make_exp("exp-001")])
    result = show_tree(str(log))
    assert "exp-001" in result


def test_linear_chain(tmp_path: Path):
    """Linear chain should show proper parent-child nesting."""
    log = tmp_path / "log.jsonl"
    _write_log(log, [
        _make_exp("exp-001"),
        _make_exp("exp-002", parent="exp-001"),
        _make_exp("exp-003", parent="exp-002"),
    ])
    result = show_tree(str(log))
    assert "exp-001" in result
    assert "exp-002" in result
    assert "exp-003" in result


def test_branching_tree(tmp_path: Path):
    """Multiple children should all appear under parent."""
    log = tmp_path / "log.jsonl"
    _write_log(log, [
        _make_exp("exp-001"),
        _make_exp("exp-002", parent="exp-001"),
        _make_exp("exp-003", parent="exp-001"),
    ])
    tree = build_tree([
        _make_exp("exp-001"),
        _make_exp("exp-002", parent="exp-001"),
        _make_exp("exp-003", parent="exp-001"),
    ])
    assert len(tree["exp-001"]) == 2


def test_multiple_roots(tmp_path: Path):
    """Experiments without parents should all be roots."""
    exps = [
        _make_exp("exp-001"),
        _make_exp("exp-002"),
        _make_exp("exp-003", parent="exp-001"),
    ]
    tree = build_tree(exps)
    roots = find_roots(exps, tree)
    assert "exp-001" in roots
    assert "exp-002" in roots
    assert "exp-003" not in roots


def test_status_markers(tmp_path: Path):
    """Kept experiments should show [+], discarded should show [-]."""
    log = tmp_path / "log.jsonl"
    _write_log(log, [
        _make_exp("exp-001", status="kept"),
        _make_exp("exp-002", status="discarded"),
    ])
    result = show_tree(str(log))
    assert "[+]" in result
    assert "[-]" in result


def test_summary_line(tmp_path: Path):
    """Output should include a summary with total and kept counts."""
    log = tmp_path / "log.jsonl"
    _write_log(log, [
        _make_exp("exp-001", status="kept"),
        _make_exp("exp-002", status="discarded"),
        _make_exp("exp-003", status="kept"),
    ])
    result = show_tree(str(log))
    assert "Total: 3" in result
    assert "2 kept" in result
