"""Tests for model compression via distillation (model_distiller.py).

Phase 18.3: Verifies student architecture selection, compression config,
distillation planning, verdict logic, and report formatting.
"""

from __future__ import annotations

import pytest

from scripts.model_distiller import (
    compute_distillation_verdict,
    format_distillation_report,
    plan_distillation,
    select_student_architecture,
    _build_distillation_config,
    _is_neural_model,
    _is_sklearn_model,
    _is_tree_model,
)


# --- model type detection ---


def test_is_tree_xgboost():
    assert _is_tree_model("xgboost") is True


def test_is_tree_lightgbm():
    assert _is_tree_model("lightgbm") is True


def test_is_neural_mlp():
    assert _is_neural_model("mlp") is True


def test_is_neural_pytorch():
    assert _is_neural_model("pytorch") is True


def test_is_sklearn_rf():
    assert _is_sklearn_model("random_forest") is True


def test_not_tree():
    assert _is_tree_model("logistic_regression") is False


# --- select_student_architecture ---


def test_student_tree():
    """Tree model should reduce estimators and depth."""
    config = {"model_type": "xgboost", "hyperparams": {"n_estimators": 400, "max_depth": 8}}
    student = select_student_architecture(config, 4)
    assert student["hyperparams"]["n_estimators"] == 100  # 400/4
    assert student["hyperparams"]["max_depth"] == 4  # 8/sqrt(4)
    assert student["compression_strategy"] == "reduce_trees"


def test_student_neural():
    """Neural model should reduce layers and width."""
    config = {"model_type": "pytorch", "hyperparams": {"hidden_size": 256, "n_layers": 4}}
    student = select_student_architecture(config, 4)
    assert student["hyperparams"]["hidden_size"] == 128  # 256/sqrt(4)
    assert student["hyperparams"]["n_layers"] == 2  # 4/sqrt(4)
    assert student["compression_strategy"] == "reduce_architecture"


def test_student_sklearn():
    """sklearn model should pick simpler family."""
    config = {"model_type": "random_forest", "hyperparams": {"max_depth": 20}}
    student = select_student_architecture(config, 4)
    assert student["model_type"] == "decision_tree"
    assert student["compression_strategy"] == "simpler_family"


def test_student_generic():
    """Unknown model type should reduce numeric params."""
    config = {"model_type": "custom_model", "hyperparams": {"param_a": 100, "param_b": "string"}}
    student = select_student_architecture(config, 4)
    assert student["hyperparams"]["param_a"] == 25  # 100/4
    assert student["hyperparams"]["param_b"] == "string"  # Non-numeric preserved


# --- _build_distillation_config ---


def test_config_soft_labels():
    """Soft labels should include temperature and alpha."""
    cfg = _build_distillation_config("soft_labels")
    assert "temperature" in cfg
    assert "alpha" in cfg


def test_config_feature_matching():
    """Feature matching should specify layers."""
    cfg = _build_distillation_config("feature_matching")
    assert "match_layers" in cfg


def test_config_dataset_distillation():
    """Dataset distillation should specify sample count."""
    cfg = _build_distillation_config("dataset_distillation")
    assert "synthetic_samples" in cfg


# --- plan_distillation ---


def test_plan_basic():
    """Should produce a complete distillation plan."""
    teacher = {
        "experiment_id": "exp-042",
        "config": {"model_type": "xgboost", "hyperparams": {"n_estimators": 500, "max_depth": 8}},
        "metrics": {"accuracy": 0.884, "model_size_bytes": 48200000},
    }
    plan = plan_distillation(teacher, compression=4)
    assert plan["teacher_id"] == "exp-042"
    assert plan["compression"] == 4
    assert plan["student"]["hyperparams"]["n_estimators"] == 125


def test_plan_with_target_latency():
    """Target latency should adjust compression."""
    teacher = {
        "experiment_id": "exp-042",
        "config": {"model_type": "xgboost", "hyperparams": {"n_estimators": 500}},
        "metrics": {"accuracy": 0.884, "latency_ms": 20.0},
    }
    plan = plan_distillation(teacher, compression=2, target_latency=5.0)
    # Teacher: 20ms, target: 5ms → need 4x speedup → compression ≈ 16
    assert plan["compression"] > 2


# --- compute_distillation_verdict ---


def test_verdict_excellent():
    """< 1% loss should be excellent."""
    verdict = compute_distillation_verdict(
        {"accuracy": 0.884}, {"accuracy": 0.880}, "accuracy", 4,
    )
    assert verdict["verdict"] == "excellent"


def test_verdict_acceptable():
    """1-3% loss should be acceptable."""
    verdict = compute_distillation_verdict(
        {"accuracy": 0.884}, {"accuracy": 0.864}, "accuracy", 4,
    )
    assert verdict["verdict"] == "acceptable"


def test_verdict_marginal():
    """3-5% loss should be marginal."""
    verdict = compute_distillation_verdict(
        {"accuracy": 0.884}, {"accuracy": 0.845}, "accuracy", 4,
    )
    assert verdict["verdict"] == "marginal"


def test_verdict_too_much_loss():
    """> 5% loss should be too much."""
    verdict = compute_distillation_verdict(
        {"accuracy": 0.884}, {"accuracy": 0.800}, "accuracy", 4,
    )
    assert verdict["verdict"] == "too_much_loss"


def test_verdict_zero_teacher():
    """Zero teacher value should handle gracefully."""
    verdict = compute_distillation_verdict(
        {"accuracy": 0.0}, {"accuracy": 0.0}, "accuracy", 4,
    )
    assert verdict["verdict"] == "no_baseline"


# --- format_distillation_report ---


def test_format_report_basic():
    """Should produce readable markdown."""
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "plan": {
            "teacher_id": "exp-042",
            "teacher_metrics": {"accuracy": 0.884},
            "teacher_config": {"model_type": "xgboost"},
            "compression": 4,
            "method": "soft_labels",
            "student": {
                "model_type": "xgboost",
                "compression_strategy": "reduce_trees",
                "hyperparams": {"n_estimators": 125},
            },
            "estimates": {"size_reduction": "75%", "student_latency_ms": 3.1},
            "distillation_config": {"temperature": 3.0, "alpha": 0.7, "description": "Soft label training"},
        },
    }
    text = format_distillation_report(report)
    assert "exp-042" in text
    assert "4x" in text
    assert "reduce_trees" in text


def test_format_report_error():
    """Error should show error message."""
    text = format_distillation_report({"error": "Teacher not found"})
    assert "ERROR" in text


def test_format_report_with_verdict():
    """Should show verdict when present."""
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "plan": {
            "teacher_id": "exp-042",
            "teacher_metrics": {},
            "teacher_config": {},
            "compression": 4,
            "method": "soft_labels",
            "student": {"model_type": "xgboost", "compression_strategy": "reduce_trees", "hyperparams": {}},
            "estimates": {},
            "distillation_config": {"description": "Soft labels"},
        },
        "verdict": {
            "verdict": "excellent",
            "delta": -0.004,
            "relative_loss": 0.005,
            "compression": 4,
            "reason": "0.5% accuracy loss for 4x compression. Excellent.",
        },
    }
    text = format_distillation_report(report)
    assert "EXCELLENT" in text
