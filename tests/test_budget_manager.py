"""Tests for compute budget manager (budget_manager.py).

Phase 18.2: Verifies budget set/status/reset, allocation policy,
phase determination, mode switching, and report formatting.
"""

from __future__ import annotations

import pytest

from scripts.budget_manager import (
    EXPLORE_PHASE_END,
    MIXED_PHASE_END,
    check_budget_allows,
    determine_phase,
    format_budget_report,
    load_budget,
    phase_to_mode,
    reset_budget,
    save_budget,
    set_budget,
    get_budget_status,
)


# --- determine_phase ---


def test_phase_explore():
    """Low usage should be explore."""
    assert determine_phase(0.0) == "explore"
    assert determine_phase(0.25) == "explore"
    assert determine_phase(0.49) == "explore"


def test_phase_mixed():
    """Mid usage should be mixed."""
    assert determine_phase(0.50) == "mixed"
    assert determine_phase(0.65) == "mixed"
    assert determine_phase(0.79) == "mixed"


def test_phase_exploit():
    """High usage should be exploit."""
    assert determine_phase(0.80) == "exploit"
    assert determine_phase(0.95) == "exploit"
    assert determine_phase(1.0) == "exploit"


# --- phase_to_mode ---


def test_mode_explore():
    assert phase_to_mode("explore") == "explore"


def test_mode_mixed():
    assert phase_to_mode("mixed") == "explore"


def test_mode_exploit():
    assert phase_to_mode("exploit") == "exploit"


# --- set_budget ---


def test_set_budget_experiments(tmp_path):
    """Should save experiment budget."""
    state = str(tmp_path / "state.yaml")
    result = set_budget(max_experiments=50, state_path=state)
    assert result["action"] == "set"
    assert result["budget"]["max_experiments"] == 50
    assert result["budget"]["active"] is True


def test_set_budget_hours(tmp_path):
    """Should save hours budget."""
    state = str(tmp_path / "state.yaml")
    result = set_budget(max_hours=8.0, state_path=state)
    assert result["budget"]["max_hours"] == 8.0


def test_set_budget_both(tmp_path):
    """Should save both constraints."""
    state = str(tmp_path / "state.yaml")
    result = set_budget(max_experiments=50, max_hours=8.0, state_path=state)
    assert result["budget"]["max_experiments"] == 50
    assert result["budget"]["max_hours"] == 8.0


def test_set_budget_no_constraints():
    """No constraints should error."""
    result = set_budget()
    assert "error" in result


# --- load_budget / save_budget ---


def test_load_save_roundtrip(tmp_path):
    """Should roundtrip through save/load."""
    state = str(tmp_path / "state.yaml")
    budget = {"max_experiments": 30, "active": True, "set_at": "2026-01-01"}
    save_budget(budget, state)
    loaded = load_budget(state)
    assert loaded["max_experiments"] == 30


def test_load_missing_file(tmp_path):
    """Missing file should return None."""
    assert load_budget(str(tmp_path / "missing.yaml")) is None


def test_save_preserves_other_state(tmp_path):
    """Should not overwrite other state keys."""
    import yaml
    state = str(tmp_path / "state.yaml")
    with open(state, "w") as f:
        yaml.dump({"convergence": {"status": "running"}}, f)

    save_budget({"max_experiments": 10, "active": True}, state)

    with open(state) as f:
        data = yaml.safe_load(f)
    assert data["convergence"]["status"] == "running"
    assert data["budget"]["max_experiments"] == 10


# --- reset_budget ---


def test_reset_budget(tmp_path):
    """Should deactivate budget."""
    state = str(tmp_path / "state.yaml")
    set_budget(max_experiments=50, state_path=state)
    result = reset_budget(state)
    assert result["action"] == "reset"
    loaded = load_budget(state)
    assert loaded["active"] is False


def test_reset_no_budget(tmp_path):
    """Resetting with no budget should be safe."""
    result = reset_budget(str(tmp_path / "missing.yaml"))
    assert result["action"] == "reset"


# --- get_budget_status ---


def test_status_no_budget(tmp_path):
    """No active budget should error."""
    result = get_budget_status(str(tmp_path / "missing.yaml"))
    assert "error" in result


def test_status_with_budget(tmp_path):
    """Should return usage stats."""
    state = str(tmp_path / "state.yaml")
    set_budget(max_experiments=50, state_path=state)
    # No experiments logged yet
    log = str(tmp_path / "log.jsonl")
    with open(log, "w") as f:
        pass  # Empty log
    result = get_budget_status(state, log)
    assert result["usage"]["experiments_used"] == 0
    assert result["usage"]["experiments_remaining"] == 50
    assert result["phase"] == "explore"
    assert result["exhausted"] is False


# --- check_budget_allows ---


def test_check_no_budget(tmp_path):
    """No budget should allow."""
    result = check_budget_allows(str(tmp_path / "missing.yaml"))
    assert result["allowed"] is True


def test_check_budget_allows_normal(tmp_path):
    """Active budget with remaining should allow."""
    state = str(tmp_path / "state.yaml")
    log = str(tmp_path / "log.jsonl")
    set_budget(max_experiments=50, state_path=state)
    with open(log, "w") as f:
        pass
    result = check_budget_allows(state, log)
    assert result["allowed"] is True


# --- format_budget_report ---


def test_format_status():
    """Should produce readable status."""
    report = {
        "action": "status",
        "budget": {"max_experiments": 50, "max_hours": 8.0},
        "usage": {
            "experiments_used": 23,
            "experiments_max": 50,
            "experiments_remaining": 27,
            "hours_used": 3.2,
            "hours_max": 8.0,
            "hours_remaining": 4.8,
            "budget_fraction": 0.46,
        },
        "phase": "explore",
        "recommended_mode": "explore",
        "allocation": {"explore": 15, "exploit": 8},
        "burn_rate": 7.2,
        "projected_exhaustion_hours": 3.75,
        "exhausted": False,
    }
    text = format_budget_report(report)
    assert "Budget Status" in text
    assert "23/50" in text
    assert "EXPLORE" in text


def test_format_set():
    """Set action should show confirmation."""
    report = {"action": "set", "budget": {}, "message": "Budget set: 50 experiments"}
    text = format_budget_report(report)
    assert "Budget Set" in text


def test_format_error():
    """Error should show error message."""
    text = format_budget_report({"error": "No budget"})
    assert "ERROR" in text


def test_format_exhausted():
    """Exhausted budget should show warning."""
    report = {
        "action": "status",
        "budget": {"max_experiments": 10},
        "usage": {
            "experiments_used": 10,
            "experiments_max": 10,
            "experiments_remaining": 0,
            "hours_used": 2.0,
            "hours_max": None,
            "hours_remaining": None,
            "budget_fraction": 1.0,
        },
        "phase": "exploit",
        "recommended_mode": "exploit",
        "allocation": {"explore": 5, "exploit": 5},
        "burn_rate": None,
        "projected_exhaustion_hours": None,
        "exhausted": True,
        "warning": "Budget exhausted.",
    }
    text = format_budget_report(report)
    assert "WARNING" in text
