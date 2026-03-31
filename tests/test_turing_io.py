"""Tests for shared data loading module (turing_io.py).

Verifies the three core IO functions that are imported by 10+ scripts:
load_experiments, load_config, and load_hypotheses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.turing_io import (
    load_config,
    load_experiments,
    load_hypotheses,
    load_reproduction,
    load_seed_study,
)


# --- load_experiments ---


def test_load_experiments_from_jsonl(tmp_path: Path):
    """Reads well-formed JSONL and returns list of dicts."""
    log = tmp_path / "log.jsonl"
    entries = [
        {"experiment_id": "exp-001", "status": "kept", "metrics": {"accuracy": 0.85}},
        {"experiment_id": "exp-002", "status": "discarded", "metrics": {"accuracy": 0.80}},
    ]
    with open(log, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    result = load_experiments(str(log))
    assert len(result) == 2
    assert result[0]["experiment_id"] == "exp-001"
    assert result[1]["status"] == "discarded"


def test_load_experiments_missing_file():
    """Returns empty list when file does not exist."""
    assert load_experiments("/nonexistent/path/log.jsonl") == []


def test_load_experiments_empty_file(tmp_path: Path):
    """Returns empty list for an empty file."""
    log = tmp_path / "log.jsonl"
    log.touch()
    assert load_experiments(str(log)) == []


def test_load_experiments_skips_blank_lines(tmp_path: Path):
    """Blank lines in the JSONL are silently skipped."""
    log = tmp_path / "log.jsonl"
    log.write_text(
        '{"experiment_id": "exp-001"}\n'
        "\n"
        '{"experiment_id": "exp-002"}\n'
        "   \n"
    )
    result = load_experiments(str(log))
    assert len(result) == 2


def test_load_experiments_skips_malformed_json(tmp_path: Path):
    """Malformed JSON lines are silently skipped."""
    log = tmp_path / "log.jsonl"
    log.write_text(
        '{"experiment_id": "exp-001"}\n'
        "this is not json\n"
        '{"experiment_id": "exp-002"}\n'
    )
    result = load_experiments(str(log))
    assert len(result) == 2
    assert result[0]["experiment_id"] == "exp-001"
    assert result[1]["experiment_id"] == "exp-002"


def test_load_experiments_preserves_order(tmp_path: Path):
    """Entries are returned in file order."""
    log = tmp_path / "log.jsonl"
    ids = ["exp-003", "exp-001", "exp-002"]
    with open(log, "w") as f:
        for eid in ids:
            f.write(json.dumps({"experiment_id": eid}) + "\n")
    result = load_experiments(str(log))
    assert [r["experiment_id"] for r in result] == ids


# --- load_config ---


def test_load_config_from_yaml(tmp_path: Path):
    """Reads a valid YAML config file."""
    config_path = tmp_path / "config.yaml"
    config = {
        "evaluation": {"primary_metric": "accuracy", "lower_is_better": False},
        "model": {"type": "xgboost"},
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = load_config(str(config_path))
    assert result["evaluation"]["primary_metric"] == "accuracy"
    assert result["model"]["type"] == "xgboost"


def test_load_config_missing_file():
    """Returns empty dict when file does not exist."""
    assert load_config("/nonexistent/config.yaml") == {}


def test_load_config_empty_file(tmp_path: Path):
    """Returns empty dict for an empty YAML file."""
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    assert load_config(str(config_path)) == {}


def test_load_config_default_path(tmp_path: Path, monkeypatch):
    """Uses 'config.yaml' as default path argument."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config = {"key": "value"}
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = load_config()
    assert result["key"] == "value"


def test_load_config_default_path_missing(tmp_path: Path, monkeypatch):
    """Returns empty dict when default config.yaml is missing."""
    monkeypatch.chdir(tmp_path)
    assert load_config() == {}


# --- load_hypotheses ---


def test_load_hypotheses_from_yaml(tmp_path: Path):
    """Reads a valid hypotheses YAML file."""
    hyp_path = tmp_path / "hypotheses.yaml"
    hypotheses = [
        {"id": "hyp-001", "description": "try xgboost", "status": "queued"},
        {"id": "hyp-002", "description": "try lightgbm", "status": "tested"},
    ]
    with open(hyp_path, "w") as f:
        yaml.dump(hypotheses, f)

    result = load_hypotheses(str(hyp_path))
    assert len(result) == 2
    assert result[0]["id"] == "hyp-001"


def test_load_hypotheses_missing_file():
    """Returns empty list when file does not exist."""
    assert load_hypotheses("/nonexistent/hypotheses.yaml") == []


def test_load_hypotheses_empty_file(tmp_path: Path):
    """Returns empty list for an empty file."""
    hyp_path = tmp_path / "hypotheses.yaml"
    hyp_path.touch()
    assert load_hypotheses(str(hyp_path)) == []


def test_load_hypotheses_non_list_yaml(tmp_path: Path):
    """Returns empty list when YAML content is not a list."""
    hyp_path = tmp_path / "hypotheses.yaml"
    with open(hyp_path, "w") as f:
        yaml.dump({"not": "a list"}, f)
    assert load_hypotheses(str(hyp_path)) == []


def test_load_hypotheses_preserves_all_fields(tmp_path: Path):
    """All fields in hypothesis entries are preserved."""
    hyp_path = tmp_path / "hypotheses.yaml"
    hypotheses = [
        {
            "id": "hyp-001",
            "description": "test hypothesis",
            "source": "human",
            "status": "queued",
            "priority": "high",
            "created_at": "2026-03-15T09:00:00",
        },
    ]
    with open(hyp_path, "w") as f:
        yaml.dump(hypotheses, f)

    result = load_hypotheses(str(hyp_path))
    assert result[0]["source"] == "human"
    assert result[0]["priority"] == "high"
    assert result[0]["created_at"] == "2026-03-15T09:00:00"


# --- load_seed_study ---


def test_load_seed_study(tmp_path: Path):
    """Reads a seed study YAML file by experiment ID."""
    seed_dir = tmp_path / "seed_studies"
    seed_dir.mkdir()
    study = {"experiment_id": "exp-042", "mean": 0.8678, "std": 0.0071}
    with open(seed_dir / "exp-042-seeds.yaml", "w") as f:
        yaml.dump(study, f)

    result = load_seed_study("exp-042", str(seed_dir))
    assert result["experiment_id"] == "exp-042"
    assert result["mean"] == 0.8678


def test_load_seed_study_missing():
    """Returns None for missing seed study."""
    assert load_seed_study("exp-999", "/nonexistent") is None


def test_load_seed_study_missing_dir(tmp_path: Path):
    """Returns None when directory doesn't exist."""
    assert load_seed_study("exp-001", str(tmp_path / "nonexistent")) is None


# --- load_reproduction ---


def test_load_reproduction(tmp_path: Path):
    """Reads a reproduction report YAML file by experiment ID."""
    repro_dir = tmp_path / "reproductions"
    repro_dir.mkdir()
    report = {"experiment_id": "exp-042", "verdict": "reproducible"}
    with open(repro_dir / "exp-042-repro.yaml", "w") as f:
        yaml.dump(report, f)

    result = load_reproduction("exp-042", str(repro_dir))
    assert result["experiment_id"] == "exp-042"
    assert result["verdict"] == "reproducible"


def test_load_reproduction_missing():
    """Returns None for missing reproduction report."""
    assert load_reproduction("exp-999", "/nonexistent") is None


def test_load_reproduction_missing_dir(tmp_path: Path):
    """Returns None when directory doesn't exist."""
    assert load_reproduction("exp-001", str(tmp_path / "nonexistent")) is None
