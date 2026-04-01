"""Edge case tests for pipeline composition manager (pipeline_manager.py).

Phase 17.2: Covers empty pipeline, single stage, missing experiment,
cache miss, deeply nested configs, and boundary conditions.
"""

from __future__ import annotations

import pytest

from scripts.pipeline_manager import (
    compose_pipeline,
    compute_stage_hash,
    define_stages,
    extract_stage_from_experiment,
    format_pipeline_report,
    get_cache_stats,
    swap_stage,
)


# --- define_stages edge cases ---


def test_define_stages_empty_config():
    """Empty config should use defaults."""
    stages = define_stages({})
    assert len(stages) == 4


def test_define_stages_single_stage():
    """Single-stage pipeline should work."""
    config = {"pipeline": {"stages": ["model"]}}
    stages = define_stages(config)
    assert len(stages) == 1
    assert stages[0]["name"] == "model"


def test_define_stages_empty_stage_list():
    """Empty stage list should produce empty pipeline."""
    config = {"pipeline": {"stages": []}}
    stages = define_stages(config)
    assert len(stages) == 0


def test_define_stages_custom_stage_names():
    """Custom stage names should work."""
    config = {"pipeline": {"stages": ["augment", "train", "evaluate"]}}
    stages = define_stages(config)
    assert len(stages) == 3
    assert stages[0]["name"] == "augment"


# --- compute_stage_hash edge cases ---


def test_hash_empty_config():
    """Empty config should produce valid hash."""
    h = compute_stage_hash({"config": {}, "description": ""})
    assert len(h) == 8


def test_hash_nested_config():
    """Nested config should hash consistently."""
    stage = {"config": {"a": {"b": {"c": 1}}}, "description": ""}
    h1 = compute_stage_hash(stage)
    h2 = compute_stage_hash(stage)
    assert h1 == h2


def test_hash_none_description():
    """Missing description should not crash."""
    h = compute_stage_hash({"config": {"a": 1}})
    assert len(h) == 8


# --- extract_stage_from_experiment edge cases ---


def test_extract_empty_experiment():
    """Experiment with no config should return empty stage."""
    exp = {"experiment_id": "exp-001"}
    stage = extract_stage_from_experiment(exp, "model")
    assert stage["name"] == "model"
    assert stage["config"]["type"] == "unknown"


def test_extract_feature_stage():
    """Should extract feature engineering config."""
    exp = {
        "experiment_id": "exp-001",
        "config": {"hyperparams": {"features": ["age", "income"], "polynomial": True}},
    }
    stage = extract_stage_from_experiment(exp, "features")
    assert stage["config"]["features"] == ["age", "income"]
    assert stage["config"]["polynomial"] is True


def test_extract_postprocess_stage():
    """Should extract postprocessing config."""
    exp = {
        "experiment_id": "exp-001",
        "config": {"hyperparams": {"calibration": "isotonic", "threshold": 0.5}},
    }
    stage = extract_stage_from_experiment(exp, "postprocess")
    assert stage["config"]["calibration"] == "isotonic"


# --- swap_stage edge cases ---


def test_swap_empty_pipeline():
    """Swapping in empty pipeline should return empty."""
    result = swap_stage([], "model", {"name": "model", "config": {}})
    assert len(result) == 0


def test_swap_updates_hash():
    """Swapped stage should get a new hash."""
    stages = [{"name": "model", "hash": "old_hash", "description": "old"}]
    new = {"name": "model", "config": {"type": "new_model"}, "description": "new"}
    result = swap_stage(stages, "model", new)
    assert result[0]["hash"] != "old_hash"


# --- get_cache_stats edge cases ---


def test_cache_stats_nested_dirs(tmp_path):
    """Should handle nested directory structures."""
    d1 = tmp_path / "model-abc"
    d1.mkdir()
    d2 = d1 / "sub"
    d2.mkdir()
    (d2 / "weights.bin").write_bytes(b"x" * 500)
    stats = get_cache_stats(str(tmp_path))
    assert stats["total_files"] == 1
    assert stats["total_size_bytes"] == 500


def test_cache_stats_mixed_files_and_dirs(tmp_path):
    """Should count both files and dirs correctly."""
    (tmp_path / "orphan.txt").write_bytes(b"x" * 10)
    d = tmp_path / "stage-abc"
    d.mkdir()
    (d / "data.pkl").write_bytes(b"y" * 20)
    stats = get_cache_stats(str(tmp_path))
    assert stats["total_files"] == 2
    assert stats["total_size_bytes"] == 30


# --- compose_pipeline edge cases ---


def test_compose_run_no_cache():
    """Run with no cache should run all stages."""
    result = compose_pipeline("run", cache_dir="/nonexistent/path")
    assert result["action"] == "run"
    assert len(result["cached_stages"]) == 0
    assert len(result["stages_to_run"]) == 4


def test_compose_cache_action():
    """Cache action should list all stages."""
    result = compose_pipeline("cache", cache_dir="/nonexistent/path")
    assert result["action"] == "cache"
    assert len(result["stages"]) == 4


# --- format_pipeline_report edge cases ---


def test_format_run_report():
    """Run report should show cached vs to-run."""
    report = {
        "action": "run",
        "total_stages": 4,
        "cached_stages": ["preprocess", "features"],
        "stages_to_run": ["model", "postprocess"],
        "message": "Would skip 2 cached stage(s) and run 2",
    }
    text = format_pipeline_report(report)
    assert "skip" in text.lower()
    assert "preprocess" in text
    assert "model" in text


def test_format_cache_report():
    """Cache report should show per-stage cache info."""
    report = {
        "action": "cache",
        "stages": [
            {"name": "model", "hash": "abc", "cache_path": "/tmp/model-abc", "already_cached": True},
        ],
        "cache_dir": "/tmp",
        "stats": {"total_files": 1, "total_size_mb": 0.5},
    }
    text = format_pipeline_report(report)
    assert "cached" in text.lower()
    assert "model" in text


def test_format_empty_show():
    """Show with no stages should still render."""
    report = {
        "action": "show",
        "stages": [],
        "cache_stats": {"total_files": 0, "total_size_mb": 0},
    }
    text = format_pipeline_report(report)
    assert "Pipeline Stages" in text
