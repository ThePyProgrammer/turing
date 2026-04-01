"""Tests for pipeline composition manager (pipeline_manager.py).

Phase 17.2: Verifies stage definition, hashing, swap logic, cache
management, composition actions, and report formatting.
"""

from __future__ import annotations

import pytest

from scripts.pipeline_manager import (
    DEFAULT_STAGES,
    check_cache,
    compose_pipeline,
    compute_stage_hash,
    define_stages,
    extract_stage_from_experiment,
    format_pipeline_report,
    get_cache_stats,
    swap_stage,
)


# --- define_stages ---


def test_define_stages_default():
    """Should create default 4-stage pipeline."""
    stages = define_stages({})
    assert len(stages) == 4
    names = [s["name"] for s in stages]
    assert names == DEFAULT_STAGES


def test_define_stages_custom():
    """Should respect custom stage list."""
    config = {"pipeline": {"stages": ["preprocess", "model"]}}
    stages = define_stages(config)
    assert len(stages) == 2
    assert stages[0]["name"] == "preprocess"
    assert stages[1]["name"] == "model"


def test_define_stages_with_model_config():
    """Should describe model stage from config."""
    config = {
        "model": {
            "type": "xgboost",
            "hyperparams": {"max_depth": 6, "n_estimators": 500},
        },
    }
    stages = define_stages(config)
    model_stage = next(s for s in stages if s["name"] == "model")
    assert "xgboost" in model_stage["description"]
    assert "max_depth=6" in model_stage["description"]


def test_define_stages_each_has_hash():
    """Every stage should have a hash."""
    stages = define_stages({})
    for s in stages:
        assert "hash" in s
        assert len(s["hash"]) == 8


# --- compute_stage_hash ---


def test_hash_deterministic():
    """Same config should produce same hash."""
    stage = {"config": {"a": 1, "b": 2}, "description": "test"}
    h1 = compute_stage_hash(stage)
    h2 = compute_stage_hash(stage)
    assert h1 == h2


def test_hash_different_config():
    """Different configs should produce different hashes."""
    s1 = {"config": {"a": 1}, "description": "test"}
    s2 = {"config": {"a": 2}, "description": "test"}
    assert compute_stage_hash(s1) != compute_stage_hash(s2)


def test_hash_key_order_independent():
    """Key order should not affect hash (sorted)."""
    s1 = {"config": {"a": 1, "b": 2}, "description": ""}
    s2 = {"config": {"b": 2, "a": 1}, "description": ""}
    assert compute_stage_hash(s1) == compute_stage_hash(s2)


# --- extract_stage_from_experiment ---


def test_extract_model_stage():
    """Should extract model config from experiment."""
    exp = {
        "experiment_id": "exp-042",
        "config": {
            "model_type": "xgboost",
            "hyperparams": {"max_depth": 8, "learning_rate": 0.05},
        },
    }
    stage = extract_stage_from_experiment(exp, "model")
    assert stage["name"] == "model"
    assert stage["config"]["type"] == "xgboost"
    assert stage["source_experiment"] == "exp-042"


def test_extract_preprocess_stage():
    """Should extract preprocessing config."""
    exp = {
        "experiment_id": "exp-001",
        "config": {"hyperparams": {"scaler": "standard", "handle_missing": "median"}},
    }
    stage = extract_stage_from_experiment(exp, "preprocess")
    assert stage["config"]["scaler"] == "standard"


def test_extract_unknown_stage():
    """Unknown stage should return empty config."""
    exp = {"experiment_id": "exp-001", "config": {"hyperparams": {}}}
    stage = extract_stage_from_experiment(exp, "custom_stage")
    assert stage["config"] == {}


# --- swap_stage ---


def test_swap_replaces_correct_stage():
    """Should replace only the named stage."""
    stages = [
        {"name": "preprocess", "description": "old", "hash": "aaa"},
        {"name": "model", "description": "old model", "hash": "bbb"},
    ]
    new_model = {"name": "model", "config": {"type": "lightgbm"}, "description": "new model"}
    result = swap_stage(stages, "model", new_model)
    assert result[0]["description"] == "old"  # Unchanged
    assert result[1]["description"] == "new model"  # Swapped


def test_swap_preserves_order():
    """Stage order should be preserved."""
    stages = [{"name": n, "hash": "x"} for n in DEFAULT_STAGES]
    new = {"name": "features", "config": {}, "description": "new"}
    result = swap_stage(stages, "features", new)
    assert [s["name"] for s in result] == DEFAULT_STAGES


def test_swap_missing_stage():
    """Swapping a non-existent stage should not change anything."""
    stages = [{"name": "model", "hash": "x", "description": "old"}]
    new = {"name": "nonexistent", "config": {}}
    result = swap_stage(stages, "nonexistent", new)
    assert len(result) == 1
    assert result[0]["description"] == "old"


# --- cache operations ---


def test_check_cache_nonexistent(tmp_path):
    """Non-existent cache should return False."""
    stage = {"name": "model", "hash": "abc12345", "config": {}}
    assert check_cache(stage, str(tmp_path)) is False


def test_check_cache_exists(tmp_path):
    """Existing cache dir should return True."""
    stage = {"name": "model", "hash": "abc12345", "config": {}}
    (tmp_path / "model-abc12345").mkdir()
    assert check_cache(stage, str(tmp_path)) is True


def test_cache_stats_empty(tmp_path):
    """Empty dir should return zero stats."""
    stats = get_cache_stats(str(tmp_path))
    assert stats["total_files"] == 0


def test_cache_stats_nonexistent(tmp_path):
    """Non-existent dir should return zero stats."""
    stats = get_cache_stats(str(tmp_path / "nonexistent"))
    assert stats["total_files"] == 0


def test_cache_stats_with_files(tmp_path):
    """Should count files and size."""
    d = tmp_path / "preprocess-abc123"
    d.mkdir()
    (d / "data.pkl").write_bytes(b"x" * 100)
    stats = get_cache_stats(str(tmp_path))
    assert stats["total_files"] == 1
    assert stats["total_size_bytes"] == 100


# --- compose_pipeline ---


def test_compose_show():
    """Show action should list stages."""
    result = compose_pipeline("show")
    assert result["action"] == "show"
    assert len(result["stages"]) == 4


def test_compose_swap_missing_stage():
    """Swap without stage name should error."""
    result = compose_pipeline("swap")
    assert "error" in result


def test_compose_swap_missing_from():
    """Swap without --from should error."""
    result = compose_pipeline("swap", stage_name="model")
    assert "error" in result


def test_compose_unknown_action():
    """Unknown action should error."""
    result = compose_pipeline("unknown_action")
    assert "error" in result


# --- format_pipeline_report ---


def test_format_show_report():
    """Show report should contain stage table."""
    report = {
        "action": "show",
        "stages": [
            {"name": "model", "description": "xgboost(max_depth=6)", "hash": "abc12345", "cached": False},
        ],
        "cache_stats": {"total_files": 0, "total_size_mb": 0},
    }
    text = format_pipeline_report(report)
    assert "Pipeline Stages" in text
    assert "model" in text
    assert "abc12345" in text


def test_format_swap_report():
    """Swap report should show old vs new."""
    report = {
        "action": "swap",
        "stage": "model",
        "source_experiment": "exp-031",
        "old_stage": {"description": "xgboost", "hash": "aaa"},
        "new_stage": {"config": {"type": "lightgbm"}},
        "updated_pipeline": [
            {"name": "model", "description": "lightgbm", "hash": "bbb"},
        ],
    }
    text = format_pipeline_report(report)
    assert "Swapped" in text
    assert "exp-031" in text


def test_format_error_report():
    """Error should show error message."""
    text = format_pipeline_report({"error": "Missing stage"})
    assert "ERROR" in text
