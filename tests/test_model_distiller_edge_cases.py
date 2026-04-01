"""Edge case tests for model compression via distillation (model_distiller.py).

Phase 18.3: Covers unknown model type, 1x compression, missing teacher,
no size data, extreme compression, and boundary conditions.
"""

from __future__ import annotations

import pytest

from scripts.model_distiller import (
    compute_distillation_verdict,
    format_distillation_report,
    plan_distillation,
    select_student_architecture,
    _build_distillation_config,
    _select_tree_student,
    _select_neural_student,
    _select_sklearn_student,
)


# --- select_student_architecture edge cases ---


def test_student_1x_compression():
    """1x compression should keep original params."""
    config = {"model_type": "xgboost", "hyperparams": {"n_estimators": 100, "max_depth": 6}}
    student = select_student_architecture(config, 1)
    assert student["hyperparams"]["n_estimators"] == 100
    assert student["hyperparams"]["max_depth"] == 6


def test_student_extreme_compression():
    """Very high compression should not go below 1."""
    config = {"model_type": "xgboost", "hyperparams": {"n_estimators": 10, "max_depth": 2}}
    student = select_student_architecture(config, 100)
    assert student["hyperparams"]["n_estimators"] >= 1
    assert student["hyperparams"]["max_depth"] >= 1


def test_student_empty_hyperparams():
    """Empty hyperparams should use defaults."""
    config = {"model_type": "xgboost", "hyperparams": {}}
    student = select_student_architecture(config, 4)
    assert "n_estimators" in student["hyperparams"]


def test_student_unknown_model():
    """Unknown model should use generic reduction."""
    config = {"model_type": "exotic_model", "hyperparams": {"layers": 16, "width": 512}}
    student = select_student_architecture(config, 4)
    assert student["compression_strategy"] == "generic_reduction"
    assert student["hyperparams"]["layers"] == 4  # 16/4
    assert student["hyperparams"]["width"] == 128  # 512/4


# --- _select_tree_student edge cases ---


def test_tree_student_no_estimators():
    """Missing n_estimators should use default 100."""
    result = _select_tree_student({}, 4)
    assert result["hyperparams"]["n_estimators"] == 25  # 100/4


def test_tree_student_preserves_lr():
    """Learning rate should be preserved."""
    result = _select_tree_student({"learning_rate": 0.05, "n_estimators": 200}, 2)
    assert result["hyperparams"]["learning_rate"] == 0.05


# --- _select_neural_student edge cases ---


def test_neural_student_no_hidden():
    """Missing hidden_size should use default 256."""
    result = _select_neural_student({}, 4)
    assert result["hyperparams"]["hidden_size"] == 128  # 256/sqrt(4)


def test_neural_student_minimum_size():
    """Very small model should not go below minimum."""
    result = _select_neural_student({"hidden_size": 16, "n_layers": 1}, 100)
    assert result["hyperparams"]["hidden_size"] >= 8
    assert result["hyperparams"]["n_layers"] >= 1


# --- _select_sklearn_student edge cases ---


def test_sklearn_student_svm():
    """SVM should map to logistic regression."""
    result = _select_sklearn_student("svm", {}, 4)
    assert result["model_type"] == "logistic_regression"


def test_sklearn_student_knn():
    """KNN should map to logistic regression."""
    result = _select_sklearn_student("knn", {}, 4)
    assert result["model_type"] == "logistic_regression"


def test_sklearn_student_unknown():
    """Unknown sklearn model should keep same type."""
    result = _select_sklearn_student("custom_sklearn", {}, 4)
    assert result["model_type"] == "custom_sklearn"


# --- plan_distillation edge cases ---


def test_plan_no_metrics():
    """Teacher with no metrics should still plan."""
    teacher = {
        "experiment_id": "exp-001",
        "config": {"model_type": "xgboost", "hyperparams": {"n_estimators": 100}},
        "metrics": {},
    }
    plan = plan_distillation(teacher)
    assert plan["teacher_id"] == "exp-001"
    assert plan["estimates"]["student_size_bytes"] is None


def test_plan_with_size():
    """Should estimate student size from teacher."""
    teacher = {
        "experiment_id": "exp-001",
        "config": {"model_type": "xgboost", "hyperparams": {}},
        "metrics": {"model_size_bytes": 40000000},
    }
    plan = plan_distillation(teacher, compression=4)
    assert plan["estimates"]["student_size_bytes"] == 10000000


# --- compute_distillation_verdict edge cases ---


def test_verdict_identical_metrics():
    """Identical metrics should be excellent."""
    verdict = compute_distillation_verdict(
        {"accuracy": 0.85}, {"accuracy": 0.85}, "accuracy", 4,
    )
    assert verdict["verdict"] == "excellent"
    assert verdict["delta"] == 0.0


def test_verdict_student_better():
    """Student better than teacher (tiny improvement) should be excellent."""
    verdict = compute_distillation_verdict(
        {"accuracy": 0.850}, {"accuracy": 0.854}, "accuracy", 4,
    )
    assert verdict["verdict"] == "excellent"


def test_verdict_missing_metric():
    """Missing metric (0) should handle gracefully."""
    verdict = compute_distillation_verdict(
        {"accuracy": 0.85}, {}, "accuracy", 4,
    )
    # student metric defaults to 0, large loss
    assert verdict["verdict"] == "too_much_loss"


# --- _build_distillation_config edge cases ---


def test_config_unknown_method():
    """Unknown method should return basic config."""
    cfg = _build_distillation_config("unknown_method")
    assert "description" in cfg


# --- format_distillation_report edge cases ---


def test_format_report_no_estimates():
    """Report with no estimates should still render."""
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "plan": {
            "teacher_id": "exp-001",
            "teacher_metrics": {},
            "teacher_config": {},
            "compression": 4,
            "method": "soft_labels",
            "student": {"model_type": "?", "compression_strategy": "?", "hyperparams": {}},
            "estimates": {"size_reduction": None, "student_latency_ms": None, "student_size_bytes": None},
            "distillation_config": {"description": "Soft labels"},
        },
    }
    text = format_distillation_report(report)
    assert "exp-001" in text
