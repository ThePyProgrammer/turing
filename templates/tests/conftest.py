"""Shared test fixtures for the {{PROJECT_NAME}} ML pipeline tests.

These fixtures provide deterministic sample data for testing the
pipeline components. Customize with records matching your actual
data schema — the autoresearch agent uses these when running tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_training_data() -> list[dict]:
    """Sample training records for testing.

    Replace these with records matching your data schema.
    Include a representative mix of all label/target values.
    """
    # TODO: Replace with your actual data schema
    return [
        {"feature_1": 1.0, "feature_2": "category_a", "label": "positive"},
        {"feature_1": 2.0, "feature_2": "category_b", "label": "positive"},
        {"feature_1": 3.0, "feature_2": "category_a", "label": "negative"},
        {"feature_1": 4.0, "feature_2": "category_b", "label": "negative"},
        {"feature_1": 5.0, "feature_2": "category_a", "label": "positive"},
        {"feature_1": 6.0, "feature_2": "category_b", "label": "negative"},
        {"feature_1": 7.0, "feature_2": "category_a", "label": "positive"},
        {"feature_1": 8.0, "feature_2": "category_b", "label": "negative"},
        {"feature_1": 9.0, "feature_2": "category_a", "label": "positive"},
        {"feature_1": 10.0, "feature_2": "category_b", "label": "negative"},
    ]


@pytest.fixture
def sample_jsonl_file(tmp_path: Path, sample_training_data: list[dict]) -> Path:
    """Write sample training data to a temporary JSONL file."""
    path = tmp_path / "test_data.jsonl"
    with open(path, "w") as f:
        for record in sample_training_data:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Write a minimal config.yaml for testing."""
    import yaml

    config = {
        "data": {
            "source": str(tmp_path / "test_data.jsonl"),
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
            "hyperparams": {
                "n_estimators": 10,
                "max_depth": 3,
                "verbosity": 0,
            },
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
