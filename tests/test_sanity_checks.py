"""Tests for pre-training sanity checks (sanity_checks.py).

Phase 20.1: Verifies each check type, verdict logic, and report formatting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.sanity_checks import (
    check_config_consistency,
    check_data_pipeline,
    check_gradient_flow,
    check_initial_loss,
    check_output_validation,
    check_single_batch_overfit,
    format_sanity_report,
    run_sanity_checks,
)


# --- check_initial_loss ---

def test_initial_loss_pass():
    result = check_initial_loss(2.31, num_classes=10, task_type="classification")
    assert result["status"] == "pass"

def test_initial_loss_fail_high():
    result = check_initial_loss(10.0, num_classes=10, task_type="classification")
    assert result["status"] == "fail"

def test_initial_loss_nan():
    result = check_initial_loss(float("nan"))
    assert result["status"] == "fail"

def test_initial_loss_regression():
    result = check_initial_loss(0.5, task_type="regression")
    assert result["status"] == "pass"

def test_initial_loss_negative_regression():
    result = check_initial_loss(-0.5, task_type="regression")
    assert result["status"] == "warn"

# --- check_single_batch_overfit ---

def test_overfit_pass():
    history = [2.3, 1.5, 0.8, 0.3, 0.1, 0.01]
    result = check_single_batch_overfit(history)
    assert result["status"] == "pass"

def test_overfit_fail():
    history = [2.3, 2.2, 2.1, 2.0, 1.95, 1.9]
    result = check_single_batch_overfit(history)
    assert result["status"] == "fail"

def test_overfit_nan():
    history = [2.3, 1.5, float("nan"), 0.8]
    result = check_single_batch_overfit(history)
    assert result["status"] == "fail"

def test_overfit_empty():
    result = check_single_batch_overfit([])
    assert result["status"] == "skip"

def test_overfit_slow():
    history = [2.3, 1.8, 1.3, 1.0, 0.8, 0.7]
    result = check_single_batch_overfit(history)
    assert result["status"] == "warn"

# --- check_gradient_flow ---

def test_gradient_pass():
    stats = [{"name": "layer1", "mean": 0.01, "max": 0.1, "min": -0.1}]
    result = check_gradient_flow(stats)
    assert result["status"] == "pass"

def test_gradient_dead():
    stats = [
        {"name": "layer1", "mean": 0.01, "max": 0.1},
        {"name": "layer2", "mean": 0.0, "max": 0.0},
    ]
    result = check_gradient_flow(stats)
    assert result["status"] == "warn"
    assert "layer2" in result["dead_layers"]

def test_gradient_exploding():
    stats = [
        {"name": "layer1", "mean": 0.01, "max": 0.1},
        {"name": "layer2", "mean": 0.01, "max": 100.0},
    ]
    result = check_gradient_flow(stats)
    assert result["status"] == "warn"

def test_gradient_empty():
    result = check_gradient_flow([])
    assert result["status"] == "skip"

# --- check_output_validation ---

def test_output_pass():
    result = check_output_validation([0.1, 0.5, 0.9, 0.3])
    assert result["status"] == "pass"

def test_output_nan():
    result = check_output_validation([0.1, float("nan"), 0.3])
    assert result["status"] == "fail"

def test_output_constant():
    result = check_output_validation([0.5, 0.5, 0.5, 0.5])
    assert result["status"] == "fail"

def test_output_extreme():
    result = check_output_validation([0.1, 200.0, -150.0], task_type="classification")
    assert result["status"] == "warn"

def test_output_empty():
    result = check_output_validation([])
    assert result["status"] == "skip"

# --- check_data_pipeline ---

def test_pipeline_pass():
    result = check_data_pipeline({"X": [32, 128], "y": [32]})
    assert result["status"] == "pass"

def test_pipeline_fail_nan():
    result = check_data_pipeline(has_nan=True)
    assert result["status"] == "fail"

def test_pipeline_fail_load():
    result = check_data_pipeline(loads_ok=False)
    assert result["status"] == "fail"

# --- check_config_consistency ---

def test_config_pass():
    config = {"model": {"hyperparams": {"learning_rate": 0.01, "batch_size": 32}}}
    result = check_config_consistency(config)
    assert result["status"] == "pass"

def test_config_high_lr():
    config = {"model": {"hyperparams": {"learning_rate": 5.0}}}
    result = check_config_consistency(config)
    assert result["status"] == "warn"

def test_config_tiny_lr():
    config = {"model": {"hyperparams": {"learning_rate": 1e-12}}}
    result = check_config_consistency(config)
    assert result["status"] == "warn"

def test_config_empty():
    result = check_config_consistency({})
    assert result["status"] == "pass"

# --- run_sanity_checks ---

def test_run_full():
    report = run_sanity_checks(
        initial_loss=2.3,
        num_classes=10,
        loss_history=[2.3, 1.0, 0.1, 0.01],
        gradient_stats=[{"name": "l1", "mean": 0.01, "max": 0.1}],
        outputs=[0.1, 0.5, 0.8],
        batch_shapes={"X": [32, 10], "y": [32]},
    )
    assert report["verdict"] == "pass"
    assert report["score"]["fail"] == 0

def test_run_quick():
    report = run_sanity_checks(quick=True, initial_loss=2.3, num_classes=10)
    assert report["quick_mode"] is True

# --- format_sanity_report ---

def test_format_pass():
    report = {
        "checked_at": "2026-01-01T00:00:00",
        "quick_mode": False,
        "verdict": "pass",
        "checks": [{"check": "data_pipeline", "status": "pass", "reason": "OK"}],
        "score": {"pass": 1, "fail": 0, "warn": 0, "skip": 0, "total": 1},
    }
    text = format_sanity_report(report)
    assert "PASS" in text

def test_format_fail():
    report = {
        "checked_at": "2026-01-01T00:00:00",
        "quick_mode": False,
        "verdict": "fail",
        "checks": [{"check": "overfit", "status": "fail", "reason": "Stuck"}],
        "score": {"pass": 0, "fail": 1, "warn": 0, "skip": 0, "total": 1},
    }
    text = format_sanity_report(report)
    assert "FAIL" in text
