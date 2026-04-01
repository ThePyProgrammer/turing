"""Tests for fork_experiment.py."""

from __future__ import annotations

import pytest

from scripts.fork_experiment import (
    create_branch,
    determine_winner,
    find_experiment,
    format_fork_report,
    save_fork_report,
)


def _make_exp(exp_id, accuracy=0.85):
    return {
        "experiment_id": exp_id,
        "status": "kept",
        "config": {"model_type": "xgboost"},
        "metrics": {"accuracy": accuracy},
    }


# --- create_branch ---


def test_create_branch():
    """Should create branch from parent."""
    parent = _make_exp("exp-042")
    branch = create_branch(parent, "try LightGBM", 0)
    assert branch["branch_id"] == "fork-exp-042-1"
    assert branch["parent_id"] == "exp-042"
    assert branch["description"] == "try LightGBM"
    assert branch["status"] == "pending"


def test_create_multiple_branches():
    """Should create branches with incrementing index."""
    parent = _make_exp("exp-042")
    b1 = create_branch(parent, "A", 0)
    b2 = create_branch(parent, "B", 1)
    assert b1["branch_id"] == "fork-exp-042-1"
    assert b2["branch_id"] == "fork-exp-042-2"


# --- determine_winner ---


def test_winner_higher_is_better():
    """Should pick highest metric when higher is better."""
    branches = [
        {"branch_id": "b-1", "status": "completed", "metrics": {"accuracy": 0.85}},
        {"branch_id": "b-2", "status": "completed", "metrics": {"accuracy": 0.90}},
    ]
    winner = determine_winner(branches, "accuracy", lower_is_better=False)
    assert winner["branch_id"] == "b-2"


def test_winner_lower_is_better():
    """Should pick lowest metric when lower is better."""
    branches = [
        {"branch_id": "b-1", "status": "completed", "metrics": {"mse": 0.10}},
        {"branch_id": "b-2", "status": "completed", "metrics": {"mse": 0.05}},
    ]
    winner = determine_winner(branches, "mse", lower_is_better=True)
    assert winner["branch_id"] == "b-2"


def test_winner_skips_failed():
    """Should ignore failed branches."""
    branches = [
        {"branch_id": "b-1", "status": "failed", "metrics": {}},
        {"branch_id": "b-2", "status": "completed", "metrics": {"accuracy": 0.80}},
    ]
    winner = determine_winner(branches, "accuracy", lower_is_better=False)
    assert winner["branch_id"] == "b-2"


def test_winner_all_failed():
    """Should return None when all branches fail."""
    branches = [
        {"branch_id": "b-1", "status": "failed", "metrics": {}},
        {"branch_id": "b-2", "status": "failed", "metrics": {}},
    ]
    assert determine_winner(branches, "accuracy", False) is None


def test_winner_no_branches():
    """Should return None for empty list."""
    assert determine_winner([], "accuracy", False) is None


def test_winner_single_branch():
    """Should return the only completed branch."""
    branches = [
        {"branch_id": "b-1", "status": "completed", "metrics": {"accuracy": 0.85}},
    ]
    winner = determine_winner(branches, "accuracy", False)
    assert winner["branch_id"] == "b-1"


# --- find_experiment ---


def test_find_experiment():
    """Should find by ID."""
    exps = [_make_exp("exp-001"), _make_exp("exp-002")]
    assert find_experiment(exps, "exp-002")["experiment_id"] == "exp-002"


def test_find_missing():
    """Should return None for missing ID."""
    assert find_experiment([_make_exp("exp-001")], "exp-999") is None


# --- format_fork_report ---


def test_format_report_with_winner():
    """Should show winner in report."""
    branches = [
        {"branch_id": "b-1", "description": "A", "status": "completed", "metrics": {"accuracy": 0.85}},
        {"branch_id": "b-2", "description": "B", "status": "completed", "metrics": {"accuracy": 0.90}},
    ]
    winner = branches[1]
    report = format_fork_report("exp-042", branches, winner, "accuracy")
    assert "WINNER" in report
    assert "b-2" in report


def test_format_report_with_failed():
    """Should show failed branches."""
    branches = [
        {"branch_id": "b-1", "description": "A", "status": "failed", "error": "oom"},
    ]
    report = format_fork_report("exp-042", branches, None, "accuracy")
    assert "FAILED" in report


def test_format_report_empty():
    """Should handle no branches."""
    report = format_fork_report("exp-042", [], None, "accuracy")
    assert "No branches" in report


# --- save_fork_report ---


def test_save_report(tmp_path):
    """Should save YAML file."""
    report = {"parent_id": "exp-042", "branches": []}
    filepath = save_fork_report(report, str(tmp_path))
    assert filepath.exists()
    assert "exp-042" in filepath.name
