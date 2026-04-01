"""Tests for smart_retry.py."""

from __future__ import annotations

import yaml
import pytest

from scripts.smart_retry import (
    BUILTIN_FAILURE_MODES,
    apply_config_changes,
    classify_failure,
    format_retry_report,
    load_failure_modes,
    save_retry_report,
)


# --- classify_failure ---


def test_classify_oom():
    """Should detect OOM errors."""
    result = classify_failure("RuntimeError: CUDA out of memory. Tried to allocate 2GB")
    assert result["failure_type"] == "oom"
    assert result["severity"] == "recoverable"


def test_classify_nan():
    """Should detect NaN loss."""
    result = classify_failure("Epoch 5: loss is NaN, stopping training")
    assert result["failure_type"] == "nan_loss"


def test_classify_timeout():
    """Should detect timeout."""
    result = classify_failure("TimeoutError: training exceeded time limit")
    assert result["failure_type"] == "timeout"


def test_classify_import():
    """Should detect import errors."""
    result = classify_failure("ModuleNotFoundError: No module named 'lightgbm'")
    assert result["failure_type"] == "import_error"
    assert result["severity"] == "requires_intervention"


def test_classify_convergence():
    """Should detect convergence failure."""
    result = classify_failure("Early stopping: loss did not decrease for 10 epochs")
    assert result["failure_type"] == "convergence_failure"


def test_classify_data():
    """Should detect data errors."""
    result = classify_failure("FileNotFoundError: data/train.csv not found")
    assert result["failure_type"] == "data_error"


def test_classify_unknown():
    """Should return unknown for unrecognized errors."""
    result = classify_failure("Something completely unexpected happened")
    assert result["failure_type"] == "unknown"
    assert result["severity"] == "requires_intervention"


def test_classify_case_insensitive():
    """Should match patterns case-insensitively."""
    result = classify_failure("memoryerror: unable to allocate array")
    assert result["failure_type"] == "oom"


def test_classify_with_custom_modes():
    """Should use custom failure modes when provided."""
    custom = {
        "custom_error": {
            "patterns": ["CustomCrash"],
            "fix": "Custom fix",
            "config_changes": {},
            "severity": "recoverable",
        }
    }
    result = classify_failure("CustomCrash occurred", failure_modes=custom)
    assert result["failure_type"] == "custom_error"


# --- apply_config_changes ---


def test_apply_divide():
    """Should divide parameter by specified amount."""
    config = {"model": {"hyperparams": {"batch_size": 32}}}
    result = apply_config_changes(config, {"batch_size": "//2"})
    assert result["model"]["hyperparams"]["batch_size"] == 16


def test_apply_multiply():
    """Should multiply parameter."""
    config = {"model": {"hyperparams": {"max_epochs": 50}}}
    result = apply_config_changes(config, {"max_epochs": "*2"})
    assert result["model"]["hyperparams"]["max_epochs"] == 100


def test_apply_literal():
    """Should set literal values."""
    config = {"model": {"hyperparams": {}}}
    result = apply_config_changes(config, {"gradient_clip": 1.0})
    assert result["model"]["hyperparams"]["gradient_clip"] == 1.0


def test_apply_missing_param():
    """Should skip missing parameters for divide/multiply."""
    config = {"model": {"hyperparams": {}}}
    result = apply_config_changes(config, {"nonexistent": "//2"})
    assert "nonexistent" not in result["model"]["hyperparams"]


def test_apply_divide_min_1():
    """Should not divide integer below 1."""
    config = {"model": {"hyperparams": {"batch_size": 1}}}
    result = apply_config_changes(config, {"batch_size": "//2"})
    assert result["model"]["hyperparams"]["batch_size"] >= 1


# --- load_failure_modes ---


def test_load_builtin():
    """Should return built-in modes when no config file."""
    modes = load_failure_modes("/nonexistent/path.yaml")
    assert "oom" in modes
    assert "nan_loss" in modes


def test_load_custom(tmp_path):
    """Should load custom modes from YAML."""
    custom = {"my_error": {"patterns": ["MyError"], "fix": "Fix it", "severity": "recoverable"}}
    path = tmp_path / "modes.yaml"
    with open(path, "w") as f:
        yaml.dump(custom, f)
    modes = load_failure_modes(str(path))
    assert "my_error" in modes


def test_builtin_completeness():
    """All built-in modes should have required fields."""
    for name, mode in BUILTIN_FAILURE_MODES.items():
        assert "patterns" in mode, f"{name} missing patterns"
        assert "fix" in mode, f"{name} missing fix"
        assert "severity" in mode, f"{name} missing severity"
        assert len(mode["patterns"]) > 0, f"{name} has no patterns"


# --- save_retry_report ---


def test_save_report(tmp_path):
    """Should save YAML file."""
    report = {"experiment_id": "exp-042", "final_status": "retry_succeeded"}
    filepath = save_retry_report(report, str(tmp_path))
    assert filepath.exists()
    assert "exp-042" in filepath.name


# --- format_retry_report ---


def test_format_recovered():
    """Should show RECOVERED for successful retry."""
    report = {
        "experiment_id": "exp-042",
        "classification": {"failure_type": "oom", "matched_pattern": "CUDA out of memory", "fix": "Halve batch"},
        "final_status": "retry_succeeded",
        "attempts": 1,
        "max_attempts": 3,
        "history": [{"status": "retry_succeeded", "message": "Fixed"}],
    }
    text = format_retry_report(report)
    assert "RECOVERED" in text
    assert "oom" in text


def test_format_failed():
    """Should show FAILED for exhausted retries."""
    report = {
        "experiment_id": "exp-001",
        "classification": {"failure_type": "nan_loss", "matched_pattern": "NaN", "fix": "Clip"},
        "final_status": "retry_failed",
        "attempts": 3,
        "max_attempts": 3,
        "history": [],
    }
    text = format_retry_report(report)
    assert "FAILED" in text


def test_format_manual():
    """Should show MANUAL FIX for non-recoverable."""
    report = {
        "experiment_id": "exp-001",
        "classification": {"failure_type": "import_error", "matched_pattern": "No module", "fix": "Install"},
        "final_status": "cannot_auto_fix",
        "attempts": 0,
        "max_attempts": 3,
    }
    text = format_retry_report(report)
    assert "MANUAL" in text


def test_format_error():
    """Should handle error dict."""
    report = {"error": "Not found"}
    text = format_retry_report(report)
    assert "ERROR" in text
