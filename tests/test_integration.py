"""Integration tests for the experiment loop critical path.

Tests the contracts between modules that talk to each other through
string-formatted output (the --- delimited blocks). Single-module
unit tests pass while the system is broken if this contract drifts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluate import evaluate_model, format_metrics
from scripts.parse_metrics import parse_metrics_block
from scripts.log_experiment import log_experiment, get_best_experiment
from scripts.check_convergence import check_convergence, load_kept_experiments


def test_metrics_format_parse_roundtrip():
    """format_metrics() output fed to parse_metrics_block() recovers same values."""
    original = {"accuracy": 0.8732, "f1_weighted": 0.8519}
    formatted = format_metrics(original)
    parsed_metrics, parsed_metadata = parse_metrics_block(formatted)

    assert parsed_metrics["accuracy"] == original["accuracy"]
    assert parsed_metrics["f1_weighted"] == original["f1_weighted"]


def test_metrics_format_parse_with_metadata():
    """Metadata keys should round-trip through format and parse."""
    metrics = {"accuracy": 0.85, "model_type": "xgboost", "train_seconds": 2.5}
    formatted = format_metrics(metrics)
    parsed_metrics, parsed_metadata = parse_metrics_block(formatted)

    assert parsed_metrics["accuracy"] == 0.85
    assert parsed_metadata["model_type"] == "xgboost"
    assert "accuracy" not in parsed_metadata
    assert "model_type" not in parsed_metrics


def test_evaluate_format_parse_chain():
    """evaluate_model -> format_metrics -> parse_metrics_block end-to-end."""
    preds = np.array([0, 1, 0, 1, 0])
    truth = np.array([0, 1, 1, 1, 0])
    config = {"evaluation": {"metrics": ["accuracy", "f1_weighted"]}}

    metrics = evaluate_model(preds, truth, config)
    metrics["model_type"] = "test"
    metrics["train_seconds"] = 1.0

    formatted = format_metrics(metrics)
    parsed_metrics, parsed_metadata = parse_metrics_block(formatted)

    assert parsed_metrics["accuracy"] == metrics["accuracy"]
    assert parsed_metrics["f1_weighted"] == metrics["f1_weighted"]


def test_log_then_convergence_check(tmp_path: Path):
    """log_experiment -> load_kept_experiments -> check_convergence chain."""
    log_path = str(tmp_path / "log.jsonl")

    # Log 5 experiments with plateau
    for i in range(5):
        log_experiment(
            log_path=log_path,
            experiment_id=f"exp-{i+1:03d}",
            config={"model_type": "xgboost"},
            metrics={"accuracy": 0.85},  # same metric = no improvement
            model_path="model.joblib",
            description=f"Experiment {i+1}",
            status="kept",
        )

    experiments = load_kept_experiments(log_path, "accuracy")
    assert len(experiments) == 5

    converged, count, msg = check_convergence(
        experiments, patience=3, improvement_threshold=0.005, lower_is_better=False,
    )
    assert converged is True
    assert "CONVERGED" in msg


def test_log_then_best_experiment(tmp_path: Path):
    """log_experiment -> get_best_experiment chain."""
    log_path = str(tmp_path / "log.jsonl")

    log_experiment(log_path, "exp-001", {"model_type": "xgboost"}, {"accuracy": 0.80}, "m.joblib", "baseline")
    log_experiment(log_path, "exp-002", {"model_type": "xgboost"}, {"accuracy": 0.90}, "m.joblib", "improved")
    log_experiment(log_path, "exp-003", {"model_type": "xgboost"}, {"accuracy": 0.85}, "m.joblib", "regression", status="discarded")

    best = get_best_experiment(log_path, "accuracy", lower_is_better=False)
    assert best["experiment_id"] == "exp-002"
    assert best["metrics"]["accuracy"] == 0.90
