"""Edge case tests for checkpoint_manager.py."""

from __future__ import annotations

import json

import yaml
import pytest

from scripts.checkpoint_manager import (
    compute_disk_stats,
    compute_pareto_checkpoints,
    enrich_checkpoints_from_log,
    prune_dominated,
    scan_checkpoints,
    save_checkpoint_report,
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


def test_scan_json_metadata(tmp_path):
    """Should read JSON metadata files."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt = ckpt_dir / "exp-001"
    ckpt.mkdir(parents=True)
    meta = {"experiment_id": "exp-001", "metrics": {"accuracy": 0.90}}
    with open(ckpt / "metadata.json", "w") as f:
        json.dump(meta, f)
    (ckpt / "model.pt").write_bytes(b"x" * 500)

    results = scan_checkpoints(str(ckpt_dir))
    assert len(results) == 1
    assert results[0]["metrics"]["accuracy"] == 0.90


def test_scan_mixed_checkpoints(tmp_path):
    """Should handle mix of directory and file checkpoints."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    # Directory checkpoint
    (ckpt_dir / "exp-001").mkdir()
    (ckpt_dir / "exp-001" / "model.pt").write_bytes(b"x" * 100)
    # File checkpoint
    (ckpt_dir / "exp-002.joblib").write_bytes(b"x" * 200)
    # Non-checkpoint file (should be ignored)
    (ckpt_dir / "README.md").write_text("ignore me")

    results = scan_checkpoints(str(ckpt_dir))
    assert len(results) == 2


def test_enrich_missing_experiment():
    """Should not crash when experiment not in log."""
    checkpoints = [{
        "path": "/tmp/ckpt",
        "name": "unknown",
        "experiment_id": "exp-999",
        "metrics": {},
        "config": {},
        "size_bytes": 1000,
        "size_mb": 0.0,
        "created": None,
    }]
    enriched = enrich_checkpoints_from_log(checkpoints, [])
    assert enriched[0]["metrics"] == {}


def test_pareto_single_checkpoint():
    """Single checkpoint should always be Pareto-optimal."""
    ckpts = [_make_ckpt("a", "exp-001")]
    pareto, dominated = compute_pareto_checkpoints(ckpts, ["accuracy"], {"accuracy": "higher"})
    assert len(pareto) == 1
    assert len(dominated) == 0


def test_pareto_lower_is_better():
    """Should respect lower-is-better for loss metrics."""
    ckpts = [
        _make_ckpt("a", "exp-001", accuracy=0.10),  # Lower MSE = better
        _make_ckpt("b", "exp-002", accuracy=0.05),
        _make_ckpt("c", "exp-003", accuracy=0.20),
    ]
    # All same train_seconds, so only accuracy matters
    pareto, dominated = compute_pareto_checkpoints(ckpts, ["accuracy"], {"accuracy": "lower"})
    pareto_ids = {c["experiment_id"] for c in pareto}
    assert "exp-002" in pareto_ids


def test_prune_nonexistent_file():
    """Should handle files that don't exist (already deleted)."""
    dominated = [{
        "path": "/tmp/nonexistent/checkpoint",
        "name": "ghost",
        "experiment_id": "exp-ghost",
        "size_bytes": 1000,
        "size_mb": 0.0,
    }]
    result = prune_dominated([], dominated, keep_latest=False)
    assert result["pruned_count"] == 0  # File didn't exist


def test_prune_directory_checkpoint(tmp_path):
    """Should remove directory checkpoints."""
    ckpt_dir = tmp_path / "checkpoints" / "exp-003"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "model.joblib").write_bytes(b"x" * 1000)
    (ckpt_dir / "metadata.yaml").write_text("test: true")

    dominated = [{
        "path": str(ckpt_dir),
        "name": "exp-003",
        "experiment_id": "exp-003",
        "size_bytes": 2000,
        "size_mb": 0.0,
    }]
    result = prune_dominated([], dominated, keep_latest=False)
    assert result["pruned_count"] == 1
    assert not ckpt_dir.exists()


def test_disk_stats_single_checkpoint():
    """Should compute stats for single checkpoint."""
    stats = compute_disk_stats([_make_ckpt("a", "exp-001", size_bytes=1048576)])
    assert stats["total_count"] == 1
    assert stats["total_size_mb"] == pytest.approx(1.0, abs=0.1)
    assert stats["avg_size_mb"] == pytest.approx(1.0, abs=0.1)


def test_save_report(tmp_path):
    """Should save checkpoint report YAML."""
    report = {"action": "prune", "pruned_count": 5}
    filepath = save_checkpoint_report(report, str(tmp_path))
    assert filepath.exists()
    with open(filepath) as f:
        loaded = yaml.safe_load(f)
    assert loaded["pruned_count"] == 5
