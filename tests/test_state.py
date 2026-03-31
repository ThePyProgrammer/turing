"""Tests for structured experiment state manager (update_state.py).

Phase 4.1: Verifies the machine-readable agent memory that replaces
free-text MEMORY.md with validated, queryable YAML.
"""

from __future__ import annotations

from pathlib import Path

from scripts.update_state import (
    add_direction,
    add_failure,
    add_observation,
    generate_memory,
    load_state,
    log_session,
    save_state,
    set_best,
    validate_state,
    EMPTY_STATE,
)


def test_load_nonexistent(tmp_path: Path):
    """Nonexistent file should return empty state."""
    state = load_state(str(tmp_path / "nope.yaml"))
    assert state == EMPTY_STATE


def test_save_and_load_roundtrip(tmp_path: Path):
    """Saved state should be loadable."""
    path = str(tmp_path / "state.yaml")
    state = dict(EMPTY_STATE)
    state["goal"] = "Predict churn"
    save_state(path, state)
    loaded = load_state(path)
    assert loaded["goal"] == "Predict churn"


def test_validate_empty_state():
    """Empty state should be valid."""
    errors = validate_state(dict(EMPTY_STATE))
    assert errors == []


def test_validate_missing_key():
    """Missing required key should be an error."""
    state = {"goal": "test"}
    errors = validate_state(state)
    assert any("Missing" in e for e in errors)


def test_validate_bad_best_result():
    """Non-dict best_result should be an error."""
    state = dict(EMPTY_STATE)
    state["best_result"] = "not a dict"
    errors = validate_state(state)
    assert any("best_result" in e for e in errors)


def test_set_best():
    state = dict(EMPTY_STATE)
    set_best(state, "exp-005", {"accuracy": 0.90})
    assert state["best_result"]["experiment_id"] == "exp-005"
    assert state["best_result"]["metrics"]["accuracy"] == 0.90
    assert "updated_at" in state["best_result"]


def test_add_observation():
    state = dict(EMPTY_STATE)
    state["observations"] = []
    add_observation(state, "LightGBM converges faster", "exp-003")
    assert len(state["observations"]) == 1
    assert state["observations"][0]["text"] == "LightGBM converges faster"
    assert state["observations"][0]["experiment_id"] == "exp-003"


def test_add_failure():
    state = dict(EMPTY_STATE)
    state["failed_approaches"] = []
    add_failure(state, "Random forest", "exp-002", "accuracy dropped 5%")
    assert len(state["failed_approaches"]) == 1
    assert state["failed_approaches"][0]["reason"] == "accuracy dropped 5%"


def test_add_direction():
    state = dict(EMPTY_STATE)
    state["promising_directions"] = []
    add_direction(state, "Try gradient boosting with dart", "high")
    assert len(state["promising_directions"]) == 1
    assert state["promising_directions"][0]["priority"] == "high"


def test_log_session():
    state = dict(EMPTY_STATE)
    state["session_history"] = []
    log_session(state, 5, 0.87)
    assert len(state["session_history"]) == 1
    assert state["session_history"][0]["experiments_run"] == 5
    assert state["session_history"][0]["best_metric"] == 0.87


def test_generate_memory_with_data():
    state = dict(EMPTY_STATE)
    state["goal"] = "Predict sentiment"
    state["primary_metric"] = "accuracy"
    state["metric_direction"] = "higher"
    set_best(state, "exp-010", {"accuracy": 0.92})
    state["observations"] = [{"text": "XGBoost works well", "timestamp": "2026-03-31"}]
    state["failed_approaches"] = [{"description": "MLP", "experiment_id": "exp-005", "reason": "too slow", "timestamp": "2026-03-31"}]
    state["promising_directions"] = [{"description": "Feature selection", "priority": "high", "timestamp": "2026-03-31"}]

    md = generate_memory(state)
    assert "# ML Researcher Memory" in md
    assert "Predict sentiment" in md
    assert "exp-010" in md
    assert "XGBoost works well" in md
    assert "MLP" in md
    assert "Feature selection" in md


def test_generate_memory_empty():
    md = generate_memory(dict(EMPTY_STATE))
    assert "# ML Researcher Memory" in md
    assert "No experiments completed" in md


def test_full_lifecycle(tmp_path: Path):
    """Test a complete state lifecycle: init, update, validate, generate."""
    path = str(tmp_path / "state.yaml")

    # Init
    save_state(path, dict(EMPTY_STATE))

    # Update
    state = load_state(path)
    state["goal"] = "Classify images"
    state["primary_metric"] = "f1_weighted"
    set_best(state, "exp-001", {"f1_weighted": 0.80})
    add_observation(state, "Baseline established")
    add_direction(state, "Try CNN features")
    log_session(state, 3, 0.80)
    save_state(path, state)

    # Validate
    loaded = load_state(path)
    errors = validate_state(loaded)
    assert errors == []

    # Generate
    md = generate_memory(loaded)
    assert "Classify images" in md
    assert "exp-001" in md
    assert "CNN features" in md
