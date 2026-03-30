"""Shared test fixtures for Turing plugin tests.

These test the plugin's template code, not a scaffolded user project.
The templates/ directory is added to sys.path so template modules
can be imported directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

# Add templates/ to path so we can import template modules
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
sys.path.insert(0, str(TEMPLATES_DIR))


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    """Write a minimal config.yaml and return its path."""
    config = {
        "data": {
            "source": str(tmp_path / "data.csv"),
            "splits_dir": str(tmp_path / "splits"),
            "target_column": "label",
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "random_state": 42,
        },
        "evaluation": {
            "primary_metric": "accuracy",
            "metrics": ["accuracy", "f1_weighted"],
            "lower_is_better": False,
        },
        "convergence": {
            "patience": 3,
            "improvement_threshold": 0.005,
        },
        "model": {
            "type": "xgboost",
            "hyperparams": {"n_estimators": 10, "max_depth": 3, "verbosity": 0},
        },
        "output": {
            "models_dir": str(tmp_path / "models"),
            "best_model_dir": str(tmp_path / "models" / "best"),
            "archive_dir": str(tmp_path / "models" / "archive"),
            "experiment_log": str(tmp_path / "experiments" / "log.jsonl"),
            "results_tsv": str(tmp_path / "experiments" / "results.tsv"),
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    """Create an empty experiment log and return its path."""
    log_dir = tmp_path / "experiments"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "log.jsonl"
    log_path.touch()
    return log_path


def make_experiment_entry(
    experiment_id: str,
    status: str = "kept",
    accuracy: float = 0.85,
    **extra_metrics: float,
) -> dict:
    """Helper to build a well-formed experiment log entry."""
    metrics = {"accuracy": accuracy, **extra_metrics}
    return {
        "experiment_id": experiment_id,
        "timestamp": "2026-03-31T00:00:00+00:00",
        "git_commit": "abc1234",
        "status": status,
        "config": {"model_type": "xgboost"},
        "metrics": metrics,
        "model_path": "models/model.joblib",
        "description": f"Test experiment {experiment_id}",
    }


def write_experiments(log_path: Path, entries: list[dict]) -> None:
    """Write experiment entries to a JSONL log file."""
    with open(log_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
