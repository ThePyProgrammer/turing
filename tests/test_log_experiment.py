"""Tests for experiment logging (log_experiment.py).

Tier 2: These tests verify ADR-0007 — append-only JSONL logging
where every experiment (kept AND discarded) is recorded and no
information is lost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.log_experiment import (
    get_best_experiment,
    get_next_experiment_id,
    log_experiment,
)


# --- get_next_experiment_id ---


def test_next_id_empty_log(tmp_log: Path):
    """Empty log should return exp-001."""
    result = get_next_experiment_id(str(tmp_log))
    assert result == "exp-001"


def test_next_id_nonexistent_file():
    """Nonexistent file should return exp-001."""
    result = get_next_experiment_id("/nonexistent/log.jsonl")
    assert result == "exp-001"


def test_next_id_sequential(tmp_log: Path):
    """Should return the next sequential ID after existing entries."""
    entries = [
        {"experiment_id": "exp-001"},
        {"experiment_id": "exp-002"},
        {"experiment_id": "exp-003"},
    ]
    with open(tmp_log, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = get_next_experiment_id(str(tmp_log))
    assert result == "exp-004"


def test_next_id_handles_gaps(tmp_log: Path):
    """Should handle non-sequential IDs (gaps in numbering)."""
    entries = [
        {"experiment_id": "exp-001"},
        {"experiment_id": "exp-005"},
    ]
    with open(tmp_log, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = get_next_experiment_id(str(tmp_log))
    assert result == "exp-006"


def test_next_id_handles_corrupted_entries(tmp_log: Path):
    """Corrupted entries should be skipped."""
    with open(tmp_log, "w") as f:
        f.write(json.dumps({"experiment_id": "exp-003"}) + "\n")
        f.write("not json\n")
        f.write(json.dumps({"experiment_id": "bad-format"}) + "\n")

    result = get_next_experiment_id(str(tmp_log))
    assert result == "exp-004"


# --- log_experiment ---


def test_log_creates_entry(tmp_path: Path):
    """Logging should create a valid JSONL entry."""
    log_path = str(tmp_path / "experiments" / "log.jsonl")
    log_experiment(
        log_path=log_path,
        experiment_id="exp-001",
        config={"model_type": "xgboost"},
        metrics={"accuracy": 0.85},
        model_path="models/model.joblib",
        description="Test experiment",
        status="kept",
    )

    path = Path(log_path)
    assert path.exists()
    with open(path) as f:
        entry = json.loads(f.readline())
    assert entry["experiment_id"] == "exp-001"
    assert entry["status"] == "kept"
    assert entry["metrics"]["accuracy"] == 0.85
    assert entry["description"] == "Test experiment"
    assert "timestamp" in entry


def test_log_append_only(tmp_path: Path):
    """Multiple logs should append, not overwrite."""
    log_path = str(tmp_path / "experiments" / "log.jsonl")

    for i in range(3):
        log_experiment(
            log_path=log_path,
            experiment_id=f"exp-{i+1:03d}",
            config={"model_type": "xgboost"},
            metrics={"accuracy": 0.80 + i * 0.05},
            model_path="models/model.joblib",
            description=f"Experiment {i+1}",
            status="kept",
        )

    with open(log_path) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 3

    entries = [json.loads(line) for line in lines]
    assert entries[0]["experiment_id"] == "exp-001"
    assert entries[1]["experiment_id"] == "exp-002"
    assert entries[2]["experiment_id"] == "exp-003"


def test_log_creates_parent_dirs(tmp_path: Path):
    """Should create parent directories if they don't exist."""
    log_path = str(tmp_path / "deep" / "nested" / "log.jsonl")
    log_experiment(
        log_path=log_path,
        experiment_id="exp-001",
        config={},
        metrics={"accuracy": 0.85},
        model_path="model.joblib",
        description="Test",
    )
    assert Path(log_path).exists()


def test_log_creates_tsv(tmp_path: Path):
    """Logging should also create a TSV summary file."""
    log_path = str(tmp_path / "experiments" / "log.jsonl")
    log_experiment(
        log_path=log_path,
        experiment_id="exp-001",
        config={"model_type": "xgboost"},
        metrics={"accuracy": 0.85},
        model_path="model.joblib",
        description="Test",
    )

    tsv_path = tmp_path / "experiments" / "results.tsv"
    assert tsv_path.exists()
    content = tsv_path.read_text()
    assert "experiment_id" in content  # header
    assert "exp-001" in content  # data row


# --- get_best_experiment ---


def test_best_experiment_empty_log(tmp_log: Path):
    """Empty log should return None."""
    result = get_best_experiment(str(tmp_log))
    assert result is None


def test_best_experiment_higher_is_better(tmp_path: Path):
    """Should return experiment with highest metric value."""
    log_path = str(tmp_path / "log.jsonl")
    entries = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.80}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"accuracy": 0.90}},
        {"experiment_id": "exp-003", "status": "kept", "metrics": {"accuracy": 0.85}},
    ]
    with open(log_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = get_best_experiment(log_path, primary_metric="accuracy", lower_is_better=False)
    assert result["experiment_id"] == "exp-002"


def test_best_experiment_lower_is_better(tmp_path: Path):
    """Should return experiment with lowest metric value for MAE/MSE."""
    log_path = str(tmp_path / "log.jsonl")
    entries = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"mae": 0.30}},
        {"experiment_id": "exp-002", "status": "kept", "metrics": {"mae": 0.10}},
        {"experiment_id": "exp-003", "status": "kept", "metrics": {"mae": 0.20}},
    ]
    with open(log_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = get_best_experiment(log_path, primary_metric="mae", lower_is_better=True)
    assert result["experiment_id"] == "exp-002"


def test_best_experiment_skips_discarded(tmp_path: Path):
    """Only 'kept' experiments should be candidates for best."""
    log_path = str(tmp_path / "log.jsonl")
    entries = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.80}},
        {"experiment_id": "exp-002", "status": "discarded", "metrics": {"accuracy": 0.99}},
    ]
    with open(log_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = get_best_experiment(log_path, primary_metric="accuracy", lower_is_better=False)
    assert result["experiment_id"] == "exp-001"  # exp-002 is discarded


def test_best_experiment_nonexistent_file():
    """Nonexistent file should return None."""
    result = get_best_experiment("/nonexistent/log.jsonl")
    assert result is None
