"""Edge case tests for compute budget manager (budget_manager.py).

Phase 18.2: Covers zero budget, exceeded budget, no experiments,
time-only budget, corrupt state, and boundary conditions.
"""

from __future__ import annotations

import json

import pytest

from scripts.budget_manager import (
    check_budget_allows,
    determine_phase,
    format_budget_report,
    get_budget_status,
    load_budget,
    reset_budget,
    save_budget,
    set_budget,
)


# --- determine_phase edge cases ---


def test_phase_negative():
    """Negative fraction should still return explore."""
    assert determine_phase(-0.1) == "explore"


def test_phase_over_100():
    """Over 100% should return exploit."""
    assert determine_phase(1.5) == "exploit"


def test_phase_exact_boundaries():
    """Exact boundary values should transition correctly."""
    assert determine_phase(0.50) == "mixed"  # Exactly at explore->mixed
    assert determine_phase(0.80) == "exploit"  # Exactly at mixed->exploit


# --- set_budget edge cases ---


def test_set_budget_zero_experiments(tmp_path):
    """Zero experiments should still save."""
    state = str(tmp_path / "state.yaml")
    result = set_budget(max_experiments=0, state_path=state)
    assert result["budget"]["max_experiments"] == 0


def test_set_budget_very_large(tmp_path):
    """Very large budget should work."""
    state = str(tmp_path / "state.yaml")
    result = set_budget(max_experiments=100000, max_hours=10000, state_path=state)
    assert result["budget"]["max_experiments"] == 100000


def test_set_budget_overwrite(tmp_path):
    """Setting budget twice should overwrite."""
    state = str(tmp_path / "state.yaml")
    set_budget(max_experiments=50, state_path=state)
    result = set_budget(max_experiments=100, state_path=state)
    loaded = load_budget(state)
    assert loaded["max_experiments"] == 100


# --- load_budget edge cases ---


def test_load_empty_file(tmp_path):
    """Empty state file should return None."""
    state = tmp_path / "state.yaml"
    state.write_text("")
    assert load_budget(str(state)) is None


def test_load_state_without_budget(tmp_path):
    """State file without budget key should return None."""
    import yaml
    state = tmp_path / "state.yaml"
    with open(state, "w") as f:
        yaml.dump({"convergence": "running"}, f)
    assert load_budget(str(state)) is None


# --- get_budget_status edge cases ---


def test_status_experiments_only(tmp_path):
    """Budget with only experiments (no hours) should work."""
    state = str(tmp_path / "state.yaml")
    log = str(tmp_path / "log.jsonl")
    set_budget(max_experiments=10, state_path=state)
    with open(log, "w") as f:
        pass
    result = get_budget_status(state, log)
    assert result["usage"]["hours_max"] is None
    assert result["usage"]["experiments_max"] == 10


def test_status_hours_only(tmp_path):
    """Budget with only hours (no experiments) should work."""
    state = str(tmp_path / "state.yaml")
    log = str(tmp_path / "log.jsonl")
    set_budget(max_hours=4.0, state_path=state)
    with open(log, "w") as f:
        pass
    result = get_budget_status(state, log)
    assert result["usage"]["experiments_max"] is None
    assert result["usage"]["hours_max"] == 4.0


def test_status_with_experiments(tmp_path):
    """Should count experiments from log."""
    state = str(tmp_path / "state.yaml")
    log = str(tmp_path / "log.jsonl")
    set_budget(max_experiments=10, state_path=state)

    # Write some experiments with recent timestamps
    budget = load_budget(state)
    with open(log, "w") as f:
        for i in range(3):
            entry = {
                "experiment_id": f"exp-{i:03d}",
                "timestamp": "2099-01-01T00:00:00",  # Far future, after budget set
                "metrics": {"accuracy": 0.8, "train_seconds": 60},
            }
            f.write(json.dumps(entry) + "\n")

    result = get_budget_status(state, log)
    assert result["usage"]["experiments_used"] == 3
    assert result["usage"]["experiments_remaining"] == 7


def test_status_inactive_budget(tmp_path):
    """Inactive budget should error."""
    state = str(tmp_path / "state.yaml")
    set_budget(max_experiments=10, state_path=state)
    reset_budget(state)
    result = get_budget_status(state)
    assert "error" in result


# --- check_budget_allows edge cases ---


def test_check_inactive_budget(tmp_path):
    """Inactive budget should allow."""
    state = str(tmp_path / "state.yaml")
    set_budget(max_experiments=10, state_path=state)
    reset_budget(state)
    result = check_budget_allows(state)
    assert result["allowed"] is True


# --- format_budget_report edge cases ---


def test_format_reset():
    """Reset action should render."""
    text = format_budget_report({"action": "reset", "message": "Budget deactivated."})
    assert "Reset" in text


def test_format_unknown_action():
    """Unknown action should handle gracefully."""
    text = format_budget_report({"action": "unknown_thing"})
    assert "Unknown" in text


def test_format_no_burn_rate():
    """No burn rate should not crash."""
    report = {
        "action": "status",
        "budget": {"max_experiments": 10},
        "usage": {
            "experiments_used": 0,
            "experiments_max": 10,
            "experiments_remaining": 10,
            "hours_used": 0,
            "hours_max": None,
            "hours_remaining": None,
            "budget_fraction": 0.0,
        },
        "phase": "explore",
        "recommended_mode": "explore",
        "allocation": {"explore": 0, "exploit": 0},
        "burn_rate": None,
        "projected_exhaustion_hours": None,
        "exhausted": False,
    }
    text = format_budget_report(report)
    assert "0/10" in text
