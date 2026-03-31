"""Tests for checkpoint manager (checkpoint_manager.py).

Phase 12.2: Verifies checkpoint scanning, Pareto pruning, disk stats,
enrichment from experiment log, and report formatting.
"""

from __future__ import annotations

import json

import pytest
import yaml

from scripts.checkpoint_manager import (
    compute_disk_stats,
    compute_pareto_checkpoints,
    enrich_checkpoints_from_log,
    format_checkpoint_list,
    format_prune_report,
    format_stats_report,
    prune_dominated,
    scan_checkpoints,
)


def _make_ckpt(name, exp_id, accuracy=0.85, train_seconds=10.0, size_bytes=1024000):
    return {
        "path": f"/tmp/checkpoints/{name}",
        "name": name,
        "experiment_id": exp_id,
        "metrics": {"accuracy": accuracy, "train_seconds": train_seconds},
        "config": {"model_type": "xgboost"},
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1024**2, 1),
        "created": "2026-04-01T00:00:00Z",
    }


# --- scan_checkpoints ---


def test_scan_empty_dir(tmp_path):
    """Should return empty for nonexistent directory."""
    assert scan_checkpoints(str(tmp_path / "nonexistent")) == []


def test_scan_empty_checkpoint_dir(tmp_path):
    """Should return empty for empty directory."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    assert scan_checkpoints(str(ckpt_dir)) == []


def test_scan_directory_checkpoints(tmp_path):
    """Should find directory-based checkpoints with metadata."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt1 = ckpt_dir / "exp-001"
    ckpt1.mkdir(parents=True)
    # Write metadata
    meta = {"experiment_id": "exp-001", "metrics": {"accuracy": 0.85}}
    with open(ckpt1 / "metadata.yaml", "w") as f:
        yaml.dump(meta, f)
    # Write a model file
    (ckpt1 / "model.joblib").write_bytes(b"x" * 1000)

    results = scan_checkpoints(str(ckpt_dir))
    assert len(results) == 1
    assert results[0]["experiment_id"] == "exp-001"
    assert results[0]["size_bytes"] > 0


def test_scan_file_checkpoints(tmp_path):
    """Should find single-file checkpoints."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "exp-001.joblib").write_bytes(b"x" * 500)
    (ckpt_dir / "exp-002.pt").write_bytes(b"x" * 1000)

    results = scan_checkpoints(str(ckpt_dir))
    assert len(results) == 2


# --- enrich_checkpoints_from_log ---


def test_enrich_from_log():
    """Should add metrics from experiment log to checkpoints."""
    checkpoints = [{
        "path": "/tmp/ckpt",
        "name": "exp-001",
        "experiment_id": "exp-001",
        "metrics": {},
        "config": {},
        "size_bytes": 1000,
        "size_mb": 0.0,
        "created": None,
    }]
    experiments = [{
        "experiment_id": "exp-001",
        "metrics": {"accuracy": 0.85},
        "config": {"model_type": "xgboost"},
        "timestamp": "2026-04-01T00:00:00Z",
    }]
    enriched = enrich_checkpoints_from_log(checkpoints, experiments)
    assert enriched[0]["metrics"]["accuracy"] == 0.85
    assert enriched[0]["created"] == "2026-04-01T00:00:00Z"


def test_enrich_preserves_existing():
    """Should not overwrite existing checkpoint metrics."""
    checkpoints = [{
        "path": "/tmp/ckpt",
        "name": "exp-001",
        "experiment_id": "exp-001",
        "metrics": {"accuracy": 0.90},
        "config": {},
        "size_bytes": 1000,
        "size_mb": 0.0,
        "created": "2026-03-01",
    }]
    experiments = [{
        "experiment_id": "exp-001",
        "metrics": {"accuracy": 0.85},
    }]
    enriched = enrich_checkpoints_from_log(checkpoints, experiments)
    assert enriched[0]["metrics"]["accuracy"] == 0.90  # Preserved


# --- compute_pareto_checkpoints ---


def test_pareto_basic():
    """Should separate Pareto-optimal from dominated checkpoints."""
    checkpoints = [
        _make_ckpt("a", "exp-001", accuracy=0.90, train_seconds=100),
        _make_ckpt("b", "exp-002", accuracy=0.85, train_seconds=10),
        _make_ckpt("c", "exp-003", accuracy=0.80, train_seconds=200),  # Dominated by a
    ]
    directions = {"accuracy": "higher", "train_seconds": "lower"}
    pareto, dominated = compute_pareto_checkpoints(checkpoints, ["accuracy", "train_seconds"], directions)

    pareto_ids = {c["experiment_id"] for c in pareto}
    dominated_ids = {c["experiment_id"] for c in dominated}

    assert "exp-001" in pareto_ids
    assert "exp-002" in pareto_ids
    assert "exp-003" in dominated_ids


def test_pareto_all_optimal():
    """When none dominates another, all are Pareto-optimal."""
    checkpoints = [
        _make_ckpt("a", "exp-001", accuracy=0.90, train_seconds=100),
        _make_ckpt("b", "exp-002", accuracy=0.85, train_seconds=10),
    ]
    directions = {"accuracy": "higher", "train_seconds": "lower"}
    pareto, dominated = compute_pareto_checkpoints(checkpoints, ["accuracy", "train_seconds"], directions)
    assert len(pareto) == 2
    assert len(dominated) == 0


def test_pareto_empty():
    """Empty input should return empty."""
    pareto, dominated = compute_pareto_checkpoints([], ["accuracy"], {"accuracy": "higher"})
    assert pareto == []
    assert dominated == []


def test_pareto_missing_metrics():
    """Checkpoints without metrics should be kept (not pruned)."""
    checkpoints = [
        _make_ckpt("a", "exp-001", accuracy=0.90, train_seconds=100),
        {"path": "/tmp/ckpt/b", "name": "b", "experiment_id": "exp-002",
         "metrics": {}, "config": {}, "size_bytes": 1000, "size_mb": 0.0, "created": None},
    ]
    directions = {"accuracy": "higher", "train_seconds": "lower"}
    pareto, dominated = compute_pareto_checkpoints(checkpoints, ["accuracy", "train_seconds"], directions)
    # exp-002 has no metrics, should be in pareto (kept)
    pareto_ids = {c["experiment_id"] for c in pareto}
    assert "exp-002" in pareto_ids


# --- prune_dominated ---


def test_prune_removes_files(tmp_path):
    """Should delete dominated checkpoint files."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    # Create a file checkpoint
    ckpt_file = ckpt_dir / "exp-003.joblib"
    ckpt_file.write_bytes(b"x" * 1000)

    dominated = [{
        "path": str(ckpt_file),
        "name": "exp-003.joblib",
        "experiment_id": "exp-003",
        "size_bytes": 1000,
        "size_mb": 0.0,
    }]
    all_ckpts = dominated + [{"path": "/tmp/other", "created": "2026-04-01"}]

    result = prune_dominated(all_ckpts, dominated, keep_latest=False)
    assert result["pruned_count"] == 1
    assert not ckpt_file.exists()


def test_prune_dry_run(tmp_path):
    """Dry run should not delete files."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt_file = ckpt_dir / "exp-003.joblib"
    ckpt_file.write_bytes(b"x" * 1000)

    dominated = [{
        "path": str(ckpt_file),
        "name": "exp-003.joblib",
        "experiment_id": "exp-003",
        "size_bytes": 1000,
        "size_mb": 0.0,
    }]
    result = prune_dominated([], dominated, dry_run=True)
    assert result["dry_run"] is True
    assert ckpt_file.exists()  # Not deleted


def test_prune_keeps_latest():
    """Should protect the most recent checkpoint."""
    dominated = [
        {"path": "/tmp/a", "name": "a", "experiment_id": "exp-old",
         "size_bytes": 1000, "size_mb": 0.0, "created": "2026-01-01"},
        {"path": "/tmp/b", "name": "b", "experiment_id": "exp-new",
         "size_bytes": 1000, "size_mb": 0.0, "created": "2026-04-01"},
    ]
    all_ckpts = dominated
    result = prune_dominated(all_ckpts, dominated, keep_latest=True)
    # exp-new should be protected
    pruned_ids = {p["experiment_id"] for p in result["pruned"]}
    assert "exp-new" not in pruned_ids


# --- compute_disk_stats ---


def test_disk_stats_basic():
    """Should compute correct disk statistics."""
    checkpoints = [
        _make_ckpt("a", "exp-001", size_bytes=1024000),
        _make_ckpt("b", "exp-002", size_bytes=2048000),
    ]
    stats = compute_disk_stats(checkpoints)
    assert stats["total_count"] == 2
    assert stats["total_size_mb"] > 0
    assert stats["avg_size_mb"] > 0
    assert stats["largest"] is not None


def test_disk_stats_empty():
    """Should handle empty checkpoint list."""
    stats = compute_disk_stats([])
    assert stats["total_count"] == 0
    assert stats["total_size_mb"] == 0


# --- format functions ---


def test_format_checkpoint_list():
    """Should produce markdown table."""
    checkpoints = [_make_ckpt("exp-001", "exp-001")]
    table = format_checkpoint_list(checkpoints, {"/tmp/checkpoints/exp-001"}, "accuracy")
    assert "exp-001" in table
    assert "accuracy" in table


def test_format_checkpoint_list_empty():
    """Should handle no checkpoints."""
    result = format_checkpoint_list([], set(), "accuracy")
    assert "No checkpoints" in result


def test_format_prune_report():
    """Should produce readable prune report."""
    prune_result = {
        "pruned_count": 3,
        "mb_saved": 150.5,
        "pruned": [
            {"name": "exp-003", "experiment_id": "exp-003", "size_mb": 50.0},
        ],
        "dry_run": False,
    }
    stats_before = {"total_count": 10, "total_size_gb": 2.0}
    stats_after = {"total_count": 7, "total_size_gb": 1.85}
    report = format_prune_report(prune_result, stats_before, stats_after)
    assert "Pruning" in report
    assert "150.5" in report


def test_format_stats_report():
    """Should produce readable stats report."""
    stats = {
        "total_count": 5,
        "total_size_mb": 500,
        "total_size_gb": 0.49,
        "avg_size_mb": 100,
        "largest": "exp-042",
        "largest_mb": 200,
    }
    checkpoints = [_make_ckpt("exp-001", "exp-001")]
    report = format_stats_report(stats, checkpoints)
    assert "Checkpoint Storage" in report
    assert "5" in report
