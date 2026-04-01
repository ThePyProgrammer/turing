"""Tests for warm-start from prior model (warm_start.py).

Phase 17.3: Verifies model type detection, warm-start strategy planning,
checkpoint discovery, layer freezing, LR adjustment, and report formatting.
"""

from __future__ import annotations

import pytest

from scripts.warm_start import (
    DEFAULT_LR_FACTOR,
    detect_model_type,
    find_checkpoint,
    format_warm_start_report,
    plan_warm_start,
    _detect_checkpoint_format,
)
from pathlib import Path


# --- detect_model_type ---


def test_detect_xgboost():
    """Should detect XGBoost as tree model."""
    exp = {"config": {"model_type": "xgboost"}}
    assert detect_model_type(exp) == "tree"


def test_detect_lightgbm():
    """Should detect LightGBM as tree model."""
    exp = {"config": {"model_type": "lightgbm"}}
    assert detect_model_type(exp) == "tree"


def test_detect_neural():
    """Should detect neural network models."""
    exp = {"config": {"model_type": "mlp"}}
    assert detect_model_type(exp) == "neural"


def test_detect_pytorch():
    """Should detect PyTorch as neural."""
    exp = {"config": {"model_type": "pytorch"}}
    assert detect_model_type(exp) == "neural"


def test_detect_sklearn():
    """Should detect sklearn warm-startable models."""
    exp = {"config": {"model_type": "random_forest"}}
    assert detect_model_type(exp) == "sklearn"


def test_detect_by_hyperparams():
    """Should detect tree model from hyperparams."""
    exp = {"config": {"model_type": "custom", "hyperparams": {"n_estimators": 100, "max_depth": 6}}}
    assert detect_model_type(exp) == "tree"


def test_detect_neural_by_hyperparams():
    """Should detect neural model from hyperparams."""
    exp = {"config": {"model_type": "custom", "hyperparams": {"hidden_size": 256}}}
    assert detect_model_type(exp) == "neural"


def test_detect_unknown():
    """Unknown model type should return unknown."""
    exp = {"config": {"model_type": "custom_model"}}
    assert detect_model_type(exp) == "unknown"


# --- plan_warm_start ---


def test_plan_xgboost():
    """XGBoost should plan continue_boosting."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "xgboost", "hyperparams": {"n_estimators": 500}}}
    plan = plan_warm_start(exp)
    assert plan["strategy"] == "continue_boosting"
    assert plan["config_changes"]["n_estimators"] == 600


def test_plan_lightgbm():
    """LightGBM should plan continue_boosting with init_model."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "lightgbm", "hyperparams": {"n_estimators": 300}}}
    plan = plan_warm_start(exp)
    assert plan["strategy"] == "continue_boosting"
    assert "init_model" in plan["config_changes"]


def test_plan_neural_basic():
    """Neural network should plan load_weights."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "pytorch", "hyperparams": {"learning_rate": 0.001}}}
    plan = plan_warm_start(exp)
    assert plan["strategy"] == "load_weights"
    assert plan["config_changes"]["learning_rate"] == 0.0001  # 0.1x default


def test_plan_neural_with_freeze():
    """Freezing layers should be included in plan."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "pytorch", "hyperparams": {"learning_rate": 0.001}}}
    plan = plan_warm_start(exp, freeze_layers=["encoder"])
    assert plan["config_changes"]["freeze_layers"] == ["encoder"]
    assert len(plan["warnings"]) == 0  # No warning when layers are frozen


def test_plan_neural_no_freeze_warning():
    """No frozen layers should produce a warning."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "pytorch", "hyperparams": {"lr": 0.001}}}
    plan = plan_warm_start(exp)
    assert any("frozen" in w.lower() for w in plan["warnings"])


def test_plan_neural_gradual_unfreezing():
    """Gradual unfreezing should be in config changes."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "pytorch", "hyperparams": {"lr": 0.001}}}
    plan = plan_warm_start(exp, freeze_layers=["encoder"], unfreeze_after=10)
    assert plan["config_changes"]["unfreeze_after_epochs"] == 10


def test_plan_sklearn():
    """sklearn should plan warm_start_param."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "random_forest", "hyperparams": {"n_estimators": 100}}}
    plan = plan_warm_start(exp)
    assert plan["strategy"] == "warm_start_param"
    assert plan["config_changes"]["warm_start"] is True


def test_plan_unknown_model():
    """Unknown model should produce unsupported strategy."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "custom_thing"}}
    plan = plan_warm_start(exp)
    assert plan["strategy"] == "unsupported"
    assert len(plan["warnings"]) > 0


def test_plan_custom_lr_factor():
    """Custom LR factor should apply."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "pytorch", "hyperparams": {"learning_rate": 0.01}}}
    plan = plan_warm_start(exp, lr_factor=0.01)
    assert plan["config_changes"]["learning_rate"] == 0.0001  # 0.01 * 0.01


# --- find_checkpoint ---


def test_find_checkpoint_directory(tmp_path):
    """Should find checkpoint in experiment directory."""
    exp_dir = tmp_path / "exp-042"
    exp_dir.mkdir()
    (exp_dir / "model.pt").write_bytes(b"x" * 1000)
    result = find_checkpoint("exp-042", str(tmp_path))
    assert result is not None
    assert result["format"] == "pytorch"
    assert result["size_bytes"] == 1000


def test_find_checkpoint_single_file(tmp_path):
    """Should find single-file checkpoints."""
    (tmp_path / "exp-042.joblib").write_bytes(b"x" * 500)
    result = find_checkpoint("exp-042", str(tmp_path))
    assert result is not None
    assert result["format"] == "joblib"


def test_find_checkpoint_not_found(tmp_path):
    """Missing checkpoint should return None."""
    result = find_checkpoint("exp-999", str(tmp_path))
    assert result is None


# --- _detect_checkpoint_format ---


def test_detect_format_pytorch():
    assert _detect_checkpoint_format([Path("model.pt")]) == "pytorch"


def test_detect_format_keras():
    assert _detect_checkpoint_format([Path("model.h5")]) == "keras"


def test_detect_format_unknown():
    assert _detect_checkpoint_format([Path("model.xyz")]) == "unknown"


# --- format_warm_start_report ---


def test_format_report_basic():
    """Should produce readable markdown."""
    report = {
        "source_experiment": "exp-042",
        "generated_at": "2026-01-01T00:00:00",
        "source_metrics": {"accuracy": 0.85},
        "checkpoint": {"path": "/ckpt/exp-042", "format": "pytorch", "size_mb": 10.5},
        "plan": {
            "model_type": "pytorch",
            "model_category": "neural",
            "strategy": "load_weights",
            "config_changes": {"learning_rate": 0.0001, "freeze_layers": ["encoder"]},
            "instructions": ["Load weights", "Freeze encoder", "Reduce LR"],
            "warnings": [],
        },
    }
    text = format_warm_start_report(report)
    assert "exp-042" in text
    assert "load_weights" in text
    assert "encoder" in text


def test_format_report_error():
    """Error should show error message."""
    text = format_warm_start_report({"error": "Experiment not found"})
    assert "ERROR" in text


def test_format_report_no_checkpoint():
    """Missing checkpoint should show warning."""
    report = {
        "source_experiment": "exp-001",
        "generated_at": "2026-01-01T00:00:00",
        "source_metrics": {},
        "checkpoint": None,
        "warning": "No checkpoint found",
        "plan": {
            "model_type": "xgboost",
            "model_category": "tree",
            "strategy": "continue_boosting",
            "config_changes": {},
            "instructions": ["Continue boosting"],
            "warnings": [],
        },
    }
    text = format_warm_start_report(report)
    assert "WARNING" in text
