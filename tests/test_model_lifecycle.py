"""Tests for model registry and lifecycle (model_lifecycle.py).

Phase 28.2: Verifies registration, promotion, demotion, gates, history, formatting.
"""

from __future__ import annotations

import pytest
import yaml

from scripts.model_lifecycle import (
    load_registry,
    save_registry,
    register_model,
    promote_model,
    demote_model,
    archive_model,
    list_models,
    get_model_at_stage,
    get_history,
    check_gates,
    format_registry_list,
    format_history,
    _find_model,
    STAGES,
    PROMOTION_GATES,
)


def _empty_registry():
    return {"models": [], "history": []}


def _registry_with_candidate():
    reg = _empty_registry()
    register_model(reg, "exp-095", "v4.1", metric=0.893)
    return reg


def _registry_with_staging():
    reg = _registry_with_candidate()
    promote_model(reg, "exp-095", "staging", {"regression": "PASS", "seed_study": "PASS"})
    return reg


# --- load/save registry ---

def test_load_missing(tmp_path):
    reg = load_registry(str(tmp_path / "missing.yaml"))
    assert reg["models"] == []
    assert reg["history"] == []

def test_save_and_load(tmp_path):
    path = str(tmp_path / "registry.yaml")
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1", metric=0.85)
    save_registry(reg, path)
    loaded = load_registry(path)
    assert len(loaded["models"]) == 1
    assert loaded["models"][0]["exp_id"] == "exp-001"


# --- register_model ---

def test_register_basic():
    reg = _empty_registry()
    result = register_model(reg, "exp-095", "v4.1", metric=0.893)
    assert "error" not in result
    assert len(result["models"]) == 1
    assert result["models"][0]["stage"] == "candidate"

def test_register_custom_stage():
    reg = _empty_registry()
    result = register_model(reg, "exp-001", "v1", stage="staging")
    assert result["models"][0]["stage"] == "staging"

def test_register_duplicate():
    reg = _registry_with_candidate()
    result = register_model(reg, "exp-095", "v4.2")
    assert "error" in result

def test_register_invalid_stage():
    reg = _empty_registry()
    result = register_model(reg, "exp-001", "v1", stage="invalid")
    assert "error" in result

def test_register_adds_history():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    assert len(reg["history"]) == 1
    assert reg["history"][0]["action"] == "register"


# --- promote_model ---

def test_promote_candidate_to_staging():
    reg = _registry_with_candidate()
    gates = {"regression": "PASS", "seed_study": "PASS"}
    result = promote_model(reg, "exp-095", "staging", gates)
    assert "error" not in result
    model = _find_model(result, "exp-095")
    assert model["stage"] == "staging"

def test_promote_staging_to_production():
    reg = _registry_with_staging()
    gates = {"audit": "PASS", "calibration": "PASS"}
    result = promote_model(reg, "exp-095", "production", gates)
    assert "error" not in result
    model = _find_model(result, "exp-095")
    assert model["stage"] == "production"

def test_promote_wrong_target():
    reg = _registry_with_candidate()
    result = promote_model(reg, "exp-095", "production")
    assert "error" in result

def test_promote_not_found():
    reg = _empty_registry()
    result = promote_model(reg, "exp-999", "staging")
    assert "error" in result

def test_promote_gates_required():
    reg = _registry_with_candidate()
    result = promote_model(reg, "exp-095", "staging")
    assert "error" in result
    assert "required_gates" in result

def test_promote_gates_failed():
    reg = _registry_with_candidate()
    gates = {"regression": "PASS", "seed_study": "FAIL"}
    result = promote_model(reg, "exp-095", "staging", gates)
    assert "error" in result
    assert "failed_gates" in result

def test_promote_force():
    reg = _registry_with_candidate()
    result = promote_model(reg, "exp-095", "staging", force=True)
    assert "error" not in result
    model = _find_model(result, "exp-095")
    assert model["stage"] == "staging"

def test_promote_from_production():
    reg = _registry_with_staging()
    promote_model(reg, "exp-095", "production", {"audit": "PASS", "calibration": "PASS"})
    result = promote_model(reg, "exp-095", "production")
    assert "error" in result

def test_promote_adds_history():
    reg = _registry_with_candidate()
    promote_model(reg, "exp-095", "staging", {"regression": "PASS", "seed_study": "PASS"})
    promote_events = [h for h in reg["history"] if h["action"] == "promote"]
    assert len(promote_events) == 1


# --- demote_model ---

def test_demote_staging_to_candidate():
    reg = _registry_with_staging()
    result = demote_model(reg, "exp-095", "candidate", reason="regression found")
    assert "error" not in result
    model = _find_model(result, "exp-095")
    assert model["stage"] == "candidate"

def test_demote_not_demotion():
    reg = _registry_with_candidate()
    result = demote_model(reg, "exp-095", "staging")
    assert "error" in result

def test_demote_not_found():
    result = demote_model(_empty_registry(), "exp-999", "candidate")
    assert "error" in result

def test_demote_adds_history():
    reg = _registry_with_staging()
    demote_model(reg, "exp-095", "candidate", "testing")
    demote_events = [h for h in reg["history"] if h["action"] == "demote"]
    assert len(demote_events) == 1
    assert demote_events[0]["reason"] == "testing"


# --- archive_model ---

def test_archive():
    reg = _registry_with_candidate()
    result = archive_model(reg, "exp-095", "superseded")
    assert "error" not in result
    model = _find_model(result, "exp-095")
    assert model["stage"] == "archived"

def test_archive_not_found():
    result = archive_model(_empty_registry(), "exp-999")
    assert "error" in result


# --- list_models ---

def test_list_all():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    register_model(reg, "exp-002", "v2")
    assert len(list_models(reg)) == 2

def test_list_filtered():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1", stage="candidate")
    register_model(reg, "exp-002", "v2", stage="staging")
    assert len(list_models(reg, stage="candidate")) == 1

def test_list_empty():
    assert list_models(_empty_registry()) == []


# --- get_model_at_stage ---

def test_get_production():
    reg = _registry_with_staging()
    promote_model(reg, "exp-095", "production", {"audit": "PASS", "calibration": "PASS"})
    model = get_model_at_stage(reg, "production")
    assert model["exp_id"] == "exp-095"

def test_get_empty_stage():
    assert get_model_at_stage(_empty_registry(), "production") is None


# --- get_history ---

def test_history_all():
    reg = _registry_with_staging()
    assert len(get_history(reg)) >= 2  # register + promote

def test_history_filtered():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    register_model(reg, "exp-002", "v2")
    assert len(get_history(reg, "exp-001")) == 1


# --- check_gates ---

def test_gates_no_artifacts():
    results = check_gates("exp-001", ["regression", "seed_study"])
    assert results["regression"] == "NOT_RUN"
    assert results["seed_study"] == "NOT_RUN"

def test_gates_with_artifact(tmp_path):
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "exp-001-seeds.yaml").write_text("verdict: PASS\n")
    results = check_gates("exp-001", ["seed_study"], seed_dir=str(seed_dir))
    assert results["seed_study"] == "PASS"


# --- format_registry_list ---

def test_format_empty():
    text = format_registry_list([])
    assert "No models" in text

def test_format_with_models():
    models = [
        {"stage": "production", "exp_id": "exp-078", "version": "v3",
         "metric": 0.872, "registered_at": "2026-03-20T00:00:00Z"},
    ]
    text = format_registry_list(models)
    assert "exp-078" in text
    assert "0.872" in text


# --- format_history ---

def test_format_history_empty():
    text = format_history([])
    assert "No history" in text

def test_format_history_entries():
    history = [
        {"action": "register", "exp_id": "exp-001", "stage": "candidate",
         "version": "v1", "timestamp": "2026-04-01T00:00:00Z"},
        {"action": "promote", "exp_id": "exp-001", "from_stage": "candidate",
         "to_stage": "staging", "forced": False, "timestamp": "2026-04-01T01:00:00Z"},
    ]
    text = format_history(history)
    assert "register" in text
    assert "promote" in text
    assert "candidate → staging" in text
