"""Edge case tests for warm-start from prior model (warm_start.py).

Phase 17.3: Covers missing checkpoint, unsupported model, no layers
to freeze, zero epochs, empty configs, and boundary conditions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.warm_start import (
    detect_model_type,
    find_checkpoint,
    format_warm_start_report,
    plan_warm_start,
    _detect_checkpoint_format,
    _plan_neural_warm_start,
    _plan_tree_warm_start,
    _plan_sklearn_warm_start,
)


# --- detect_model_type edge cases ---


def test_detect_empty_config():
    """Empty config should return unknown."""
    assert detect_model_type({}) == "unknown"


def test_detect_case_insensitive():
    """Detection should be case-insensitive."""
    exp = {"config": {"model_type": "XGBoost"}}
    assert detect_model_type(exp) == "tree"


def test_detect_partial_match():
    """Should match partial model names."""
    exp = {"config": {"model_type": "xgboost_classifier"}}
    assert detect_model_type(exp) == "tree"


def test_detect_catboost():
    """CatBoost should be detected as tree."""
    exp = {"config": {"model_type": "catboost"}}
    assert detect_model_type(exp) == "tree"


def test_detect_transformer():
    """Transformer should be detected as neural."""
    exp = {"config": {"model_type": "transformer"}}
    assert detect_model_type(exp) == "neural"


# --- plan_warm_start edge cases ---


def test_plan_empty_hyperparams():
    """Empty hyperparams should not crash."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "xgboost", "hyperparams": {}}}
    plan = plan_warm_start(exp)
    assert plan["strategy"] == "continue_boosting"
    assert plan["config_changes"]["n_estimators"] == 200  # 100 (default) + 100


def test_plan_no_config():
    """No config should produce unknown strategy."""
    exp = {"experiment_id": "exp-001"}
    plan = plan_warm_start(exp)
    assert plan["strategy"] == "unsupported"


def test_plan_multiple_freeze_layers():
    """Multiple frozen layers should all be listed."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "pytorch", "hyperparams": {"lr": 0.001}}}
    plan = plan_warm_start(exp, freeze_layers=["encoder", "decoder", "attention"])
    assert len(plan["config_changes"]["freeze_layers"]) == 3


def test_plan_zero_lr_factor():
    """Zero LR factor should set LR to zero."""
    exp = {"experiment_id": "exp-001", "config": {"model_type": "pytorch", "hyperparams": {"learning_rate": 0.01}}}
    plan = plan_warm_start(exp, lr_factor=0.0)
    assert plan["config_changes"]["learning_rate"] == 0.0


# --- _plan_tree_warm_start edge cases ---


def test_tree_plan_generic_gradient_boosting():
    """Generic gradient boosting should use warm_start param."""
    plan = _plan_tree_warm_start(
        {"model_type": "gradient_boosting"},
        {"n_estimators": 200},
    )
    assert plan["strategy"] == "warm_start_param"
    assert plan["config_changes"]["warm_start"] is True


# --- _plan_neural_warm_start edge cases ---


def test_neural_plan_no_lr_in_config():
    """Missing LR should default to 0.001."""
    plan = _plan_neural_warm_start(
        {"model_type": "pytorch"},
        {},  # No learning_rate
        freeze_layers=None,
        unfreeze_after=None,
        lr_factor=0.1,
    )
    assert plan["config_changes"]["learning_rate"] == 0.0001  # 0.001 * 0.1


# --- _plan_sklearn_warm_start edge cases ---


def test_sklearn_plan_no_estimators():
    """Missing n_estimators should use default."""
    plan = _plan_sklearn_warm_start({"model_type": "random_forest"}, {})
    assert plan["config_changes"]["n_estimators"] == 150  # 100 + 50


# --- find_checkpoint edge cases ---


def test_find_checkpoint_empty_dir(tmp_path):
    """Empty experiment directory should return None."""
    (tmp_path / "exp-042").mkdir()
    result = find_checkpoint("exp-042", str(tmp_path))
    assert result is None


def test_find_checkpoint_multiple_formats(tmp_path):
    """Should find first matching format."""
    (tmp_path / "exp-042.joblib").write_bytes(b"x")
    (tmp_path / "exp-042.pkl").write_bytes(b"y")
    result = find_checkpoint("exp-042", str(tmp_path))
    assert result is not None


def test_find_checkpoint_xgboost(tmp_path):
    """Should find .xgb checkpoint."""
    (tmp_path / "exp-042.xgb").write_bytes(b"x" * 200)
    result = find_checkpoint("exp-042", str(tmp_path))
    assert result is not None
    assert result["format"] == "xgb"


def test_find_checkpoint_nonexistent_dir():
    """Non-existent checkpoint dir should return None."""
    result = find_checkpoint("exp-042", "/nonexistent/path")
    assert result is None


# --- _detect_checkpoint_format edge cases ---


def test_detect_format_joblib():
    assert _detect_checkpoint_format([Path("m.joblib")]) == "joblib"


def test_detect_format_pickle():
    assert _detect_checkpoint_format([Path("m.pkl")]) == "pickle"


def test_detect_format_lightgbm():
    assert _detect_checkpoint_format([Path("m.lgb")]) == "lightgbm"


def test_detect_format_xgboost():
    assert _detect_checkpoint_format([Path("m.xgb")]) == "xgboost"


def test_detect_format_mixed():
    """Mixed formats should prefer pytorch."""
    files = [Path("weights.pt"), Path("config.json")]
    assert _detect_checkpoint_format(files) == "pytorch"


# --- format_warm_start_report edge cases ---


def test_format_report_with_warnings():
    """Report with warnings should display them."""
    report = {
        "source_experiment": "exp-001",
        "generated_at": "2026-01-01T00:00:00",
        "source_metrics": {},
        "checkpoint": None,
        "plan": {
            "model_type": "custom",
            "model_category": "unknown",
            "strategy": "unsupported",
            "config_changes": {},
            "instructions": [],
            "warnings": ["Model type not supported for warm-starting"],
        },
    }
    text = format_warm_start_report(report)
    assert "Warnings" in text
    assert "not supported" in text


def test_format_report_empty_plan():
    """Empty plan should still render."""
    report = {
        "source_experiment": "exp-001",
        "generated_at": "2026-01-01T00:00:00",
        "source_metrics": {},
        "checkpoint": None,
        "plan": {
            "model_type": "?",
            "model_category": "unknown",
            "strategy": "unsupported",
            "config_changes": {},
            "instructions": [],
            "warnings": [],
        },
    }
    text = format_warm_start_report(report)
    assert "exp-001" in text
