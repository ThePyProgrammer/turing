"""Tests for model card generator (generate_model_card.py).

Verifies that standardized model cards are produced correctly from
experiment logs, model contracts, and project config.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.generate_model_card import (
    generate_card,
    load_best_experiment,
    load_model_contract,
)


def _make_exp(
    exp_id: str,
    status: str = "kept",
    accuracy: float = 0.85,
    f1_weighted: float = 0.83,
    model_type: str = "xgboost",
    **extra_metrics: float,
) -> dict:
    metrics = {"accuracy": accuracy, "f1_weighted": f1_weighted, **extra_metrics}
    return {
        "experiment_id": exp_id,
        "timestamp": "2026-03-30T12:00:00+00:00",
        "status": status,
        "config": {"model_type": model_type},
        "metrics": metrics,
        "model_path": "models/model.joblib",
        "description": f"Test {exp_id}",
    }


def _write_log(path: Path, experiments: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in experiments:
            f.write(json.dumps(e) + "\n")


def _write_config(path: Path, config: dict | None = None) -> None:
    if config is None:
        config = {
            "project_name": "test-project",
            "task_description": "Predict test outcomes",
            "data": {
                "source": "data/test.csv",
                "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            },
            "evaluation": {
                "primary_metric": "accuracy",
                "metrics": ["accuracy", "f1_weighted"],
                "lower_is_better": False,
            },
            "convergence": {"patience": 3, "improvement_threshold": 0.005},
            "model": {"type": "xgboost"},
        }
    with open(path, "w") as f:
        yaml.dump(config, f)


def _write_contract(path: Path) -> None:
    path.write_text(
        "# Model Artifact Contract\n\n"
        "Version: 2\n\n"
        "## Bundle Format\n\n"
        "The trained model is saved as a joblib bundle at `models/best/model.joblib`.\n",
        encoding="utf-8",
    )


# --- load_best_experiment ---


def test_load_best_experiment(tmp_path: Path):
    """Finds the correct best experiment among kept entries."""
    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, [
        _make_exp("exp-001", "kept", accuracy=0.80),
        _make_exp("exp-002", "kept", accuracy=0.92),
        _make_exp("exp-003", "discarded", accuracy=0.99),
        _make_exp("exp-004", "kept", accuracy=0.88),
    ])

    best = load_best_experiment(str(log_path), "accuracy", lower_is_better=False)
    assert best is not None
    assert best["experiment_id"] == "exp-002"
    assert best["metrics"]["accuracy"] == 0.92


def test_load_best_experiment_empty(tmp_path: Path):
    """Returns None when no experiments exist."""
    log_path = tmp_path / "log.jsonl"
    assert load_best_experiment(str(log_path), "accuracy", False) is None


# --- generate_card full integration ---


def test_generate_card_with_data(tmp_path: Path):
    """Full card generation with sample experiment data."""
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, [
        _make_exp("exp-001", "kept", accuracy=0.80, f1_weighted=0.78),
        _make_exp("exp-002", "discarded", accuracy=0.75, f1_weighted=0.72),
        _make_exp("exp-003", "kept", accuracy=0.90, f1_weighted=0.88),
    ])

    contract_path = tmp_path / "model_contract.md"
    _write_contract(contract_path)

    output_path = tmp_path / "MODEL_CARD.md"
    card = generate_card(
        str(config_path), str(log_path), str(contract_path), str(output_path),
    )

    assert output_path.exists()
    assert "# Model Card: test-project" in card
    assert "## Model Details" in card
    assert "## Training Data" in card
    assert "## Performance" in card
    assert "## Training History" in card
    assert "exp-003" in card  # best experiment
    assert "xgboost" in card


def test_generate_card_empty_log(tmp_path: Path):
    """Handles no experiments gracefully — produces a skeleton card."""
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    card = generate_card(
        str(config_path),
        str(tmp_path / "nonexistent.jsonl"),
        str(tmp_path / "nonexistent_contract.md"),
    )

    assert "# Model Card:" in card
    assert "No experiments completed yet" in card
    assert "N/A" in card
    assert "## Limitations" in card


def test_generate_card_includes_metrics(tmp_path: Path):
    """All configured metrics appear in the performance table."""
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, [
        _make_exp("exp-001", "kept", accuracy=0.85, f1_weighted=0.83),
    ])

    card = generate_card(str(config_path), str(log_path), str(tmp_path / "c.md"))

    assert "accuracy" in card
    assert "f1_weighted" in card
    assert "0.8500" in card
    assert "0.8300" in card


def test_generate_card_includes_limitations(tmp_path: Path):
    """Limitations section is present with data source reference."""
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, [_make_exp("exp-001", "kept")])

    card = generate_card(str(config_path), str(log_path), str(tmp_path / "c.md"))

    assert "## Limitations" in card
    assert "data/test.csv" in card
    assert "may not generalize" in card
    assert "validation set" in card


def test_generate_card_includes_contract(tmp_path: Path):
    """Artifact contract section includes version and bundle format."""
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, [_make_exp("exp-001", "kept")])

    contract_path = tmp_path / "model_contract.md"
    _write_contract(contract_path)

    card = generate_card(str(config_path), str(log_path), str(contract_path))

    assert "## Artifact Contract" in card
    assert "**Contract version:** 2" in card
    assert "joblib" in card.lower()
    assert "model_contract.md" in card
