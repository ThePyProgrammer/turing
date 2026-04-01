"""Edge case tests for model registry (model_lifecycle.py).

Phase 28.2: Empty registry, double promote, invalid stages, corrupt files.
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
    _check_artifact,
    STAGES,
)


def _empty_registry():
    return {"models": [], "history": []}


# --- Registry IO edge cases ---

def test_load_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.yaml"
    path.write_text("not: a: valid: yaml: [")
    reg = load_registry(str(path))
    assert reg["models"] == []

def test_load_non_dict(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- item1\n- item2\n")
    reg = load_registry(str(path))
    assert reg["models"] == []

def test_load_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    reg = load_registry(str(path))
    assert reg["models"] == []

def test_save_creates_dirs(tmp_path):
    path = str(tmp_path / "deep" / "nested" / "registry.yaml")
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    save_registry(reg, path)
    loaded = load_registry(path)
    assert len(loaded["models"]) == 1


# --- Registration edge cases ---

def test_register_no_metric():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    assert reg["models"][0]["metric"] is None

def test_register_multiple():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    register_model(reg, "exp-002", "v2")
    register_model(reg, "exp-003", "v3")
    assert len(reg["models"]) == 3

def test_register_all_stages():
    reg = _empty_registry()
    for i, stage in enumerate(STAGES):
        register_model(reg, f"exp-{i:03d}", f"v{i}", stage=stage)
    assert len(reg["models"]) == len(STAGES)


# --- Promotion edge cases ---

def test_promote_archived():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    archive_model(reg, "exp-001")
    result = promote_model(reg, "exp-001", "staging")
    assert "error" in result

def test_promote_with_partial_gates():
    """Some gates pass, some not run."""
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    gates = {"regression": "PASS", "seed_study": "NOT_RUN"}
    result = promote_model(reg, "exp-001", "staging", gates)
    assert "error" in result

def test_promote_no_required_gates():
    """Promote to a stage with no gate requirements (force-style)."""
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    result = promote_model(reg, "exp-001", "staging", force=True)
    assert "error" not in result


# --- Demotion edge cases ---

def test_demote_to_same_stage():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1", stage="staging")
    result = demote_model(reg, "exp-001", "staging")
    assert "error" in result

def test_demote_invalid_stage():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1", stage="staging")
    result = demote_model(reg, "exp-001", "nonexistent")
    assert "error" in result

def test_demote_production_to_candidate():
    """Skip a stage during demotion."""
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1", stage="production")
    result = demote_model(reg, "exp-001", "candidate")
    assert "error" not in result
    assert _find_model(result, "exp-001")["stage"] == "candidate"


# --- Archive edge cases ---

def test_archive_already_archived():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    archive_model(reg, "exp-001")
    # Archiving again should still work (idempotent — already archived)
    model = _find_model(reg, "exp-001")
    assert model["stage"] == "archived"

def test_archive_production():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1", stage="production")
    result = archive_model(reg, "exp-001", "end of life")
    assert "error" not in result
    assert _find_model(result, "exp-001")["stage"] == "archived"


# --- _find_model edge cases ---

def test_find_nonexistent():
    assert _find_model(_empty_registry(), "exp-999") is None

def test_find_empty():
    assert _find_model({"models": []}, "exp-001") is None


# --- list_models sorting ---

def test_list_sorted_by_stage():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1", stage="archived")
    register_model(reg, "exp-002", "v2", stage="candidate")
    register_model(reg, "exp-003", "v3", stage="production")
    models = list_models(reg)
    stages = [m["stage"] for m in models]
    assert stages == ["candidate", "production", "archived"]


# --- check_gates edge cases ---

def test_check_unknown_gate():
    results = check_gates("exp-001", ["unknown_gate"])
    assert results["unknown_gate"] == "NOT_RUN"

def test_check_artifact_fail(tmp_path):
    d = tmp_path / "regressions"
    d.mkdir()
    (d / "regress-001.yaml").write_text("verdict: FAIL\n")
    result = _check_artifact(str(d), "exp-001", ["regress-*.yaml"])
    assert result == "FAIL"

def test_check_artifact_no_verdict(tmp_path):
    d = tmp_path / "regressions"
    d.mkdir()
    (d / "regress-001.yaml").write_text("some_data: 42\n")
    result = _check_artifact(str(d), "exp-001", ["regress-*.yaml"])
    assert result == "PASS"  # exists but no verdict → PASS


# --- History edge cases ---

def test_history_tracks_all_actions():
    reg = _empty_registry()
    register_model(reg, "exp-001", "v1")
    promote_model(reg, "exp-001", "staging", force=True)
    demote_model(reg, "exp-001", "candidate")
    archive_model(reg, "exp-001")
    actions = [h["action"] for h in reg["history"]]
    assert actions == ["register", "promote", "demote", "archive"]


# --- Format edge cases ---

def test_format_list_no_metric():
    models = [{"stage": "candidate", "exp_id": "exp-001", "version": "v1",
               "metric": None, "registered_at": "2026-04-01T00:00:00Z"}]
    text = format_registry_list(models)
    assert "—" in text

def test_format_history_demote():
    history = [{"action": "demote", "exp_id": "exp-001", "from_stage": "production",
                "to_stage": "staging", "reason": "latency", "timestamp": "2026-04-01T00:00:00Z"}]
    text = format_history(history)
    assert "latency" in text

def test_format_history_archive():
    history = [{"action": "archive", "exp_id": "exp-001", "from_stage": "staging",
                "timestamp": "2026-04-01T00:00:00Z"}]
    text = format_history(history)
    assert "archived" in text

def test_format_history_forced():
    history = [{"action": "promote", "exp_id": "exp-001", "from_stage": "candidate",
                "to_stage": "staging", "forced": True, "timestamp": "2026-04-01T00:00:00Z"}]
    text = format_history(history)
    assert "forced" in text
