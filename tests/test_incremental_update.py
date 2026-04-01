"""Tests for incremental model update (incremental_update.py).

Phase 28.1: Verifies model detection, update planning, forgetting check, reporting.
"""

from __future__ import annotations

import pytest

from scripts.incremental_update import (
    detect_model_type,
    plan_tree_update,
    plan_neural_update,
    plan_sklearn_update,
    plan_update,
    check_forgetting,
    build_update_report,
    incremental_update,
    format_update_report,
    _count_data_samples,
    DEFAULT_FORGETTING_TOLERANCE,
)


# --- detect_model_type ---

def test_detect_xgboost():
    exp = {"config": {"model_type": "xgboost"}}
    assert detect_model_type(exp) == "tree"

def test_detect_lightgbm():
    exp = {"config": {"model_type": "lightgbm"}}
    assert detect_model_type(exp) == "tree"

def test_detect_neural():
    exp = {"config": {"model_type": "mlp"}}
    assert detect_model_type(exp) == "neural"

def test_detect_sklearn_partial():
    exp = {"config": {"model_type": "sgd"}}
    assert detect_model_type(exp) == "sklearn_partial"

def test_detect_sklearn_warm():
    exp = {"config": {"model_type": "random_forest"}}
    assert detect_model_type(exp) == "sklearn_warm"

def test_detect_from_hyperparams():
    exp = {"config": {"model_type": "", "hyperparams": {"n_estimators": 100, "max_depth": 6}}}
    assert detect_model_type(exp) == "tree"

def test_detect_neural_from_hyperparams():
    exp = {"config": {"model_type": "", "hyperparams": {"hidden_size": 128}}}
    assert detect_model_type(exp) == "neural"

def test_detect_unknown():
    exp = {"config": {"model_type": "custom_model"}}
    assert detect_model_type(exp) == "unknown"


# --- plan_tree_update ---

def test_tree_plan():
    exp = {"experiment_id": "exp-042", "config": {"hyperparams": {"n_estimators": 100}}}
    plan = plan_tree_update(exp, new_data_size=5000)
    assert plan["strategy"] == "continued_boosting"
    assert plan["current_rounds"] == 100
    assert plan["additional_rounds"] == 50
    assert plan["total_rounds"] == 150

def test_tree_plan_custom_rounds():
    exp = {"experiment_id": "exp-042", "config": {"hyperparams": {"n_estimators": 200}}}
    plan = plan_tree_update(exp, new_data_size=1000, new_rounds=100)
    assert plan["total_rounds"] == 300


# --- plan_neural_update ---

def test_neural_plan():
    exp = {"experiment_id": "exp-042", "config": {"hyperparams": {"learning_rate": 0.001}}}
    plan = plan_neural_update(exp, new_data_size=5000, replay_ratio=0.1)
    assert plan["strategy"] == "fine_tune_with_replay"
    assert plan["fine_tune_lr"] == 0.0001
    assert plan["replay_size"] == 500
    assert plan["total_training_size"] == 5500

def test_neural_plan_custom_replay():
    exp = {"experiment_id": "exp-042", "config": {"hyperparams": {"lr": 0.01}}}
    plan = plan_neural_update(exp, new_data_size=10000, replay_ratio=0.2)
    assert plan["replay_size"] == 2000


# --- plan_sklearn_update ---

def test_sklearn_partial():
    exp = {"experiment_id": "exp-042"}
    plan = plan_sklearn_update(exp, 5000, "sklearn_partial")
    assert plan["strategy"] == "partial_fit"

def test_sklearn_warm():
    exp = {"experiment_id": "exp-042"}
    plan = plan_sklearn_update(exp, 5000, "sklearn_warm")
    assert plan["strategy"] == "warm_start_retrain"


# --- plan_update (dispatcher) ---

def test_plan_tree():
    exp = {"config": {"model_type": "xgboost", "hyperparams": {"n_estimators": 100}}}
    plan = plan_update(exp, 5000)
    assert plan["strategy"] == "continued_boosting"

def test_plan_neural():
    exp = {"config": {"model_type": "pytorch", "hyperparams": {"learning_rate": 0.001}}}
    plan = plan_update(exp, 5000)
    assert plan["strategy"] == "fine_tune_with_replay"

def test_plan_unknown():
    exp = {"config": {"model_type": "custom"}}
    plan = plan_update(exp, 5000)
    assert "error" in plan


# --- check_forgetting ---

def test_forgetting_pass():
    result = check_forgetting(
        {"accuracy": 0.891}, {"accuracy": 0.889},
        "accuracy", tolerance=0.005,
    )
    assert result["verdict"] == "PASS"
    assert result["within_tolerance"]

def test_forgetting_warning():
    result = check_forgetting(
        {"accuracy": 0.891}, {"accuracy": 0.883},
        "accuracy", tolerance=0.005,
    )
    assert result["verdict"] == "WARNING"

def test_forgetting_fail():
    result = check_forgetting(
        {"accuracy": 0.891}, {"accuracy": 0.850},
        "accuracy", tolerance=0.005,
    )
    assert result["verdict"] == "FAIL"

def test_forgetting_improved():
    result = check_forgetting(
        {"accuracy": 0.891}, {"accuracy": 0.895},
        "accuracy", tolerance=0.005,
    )
    assert result["verdict"] == "PASS"
    assert result["degradation"] < 0

def test_forgetting_lower_is_better():
    result = check_forgetting(
        {"loss": 0.10}, {"loss": 0.11},
        "loss", tolerance=0.02, lower_is_better=True,
    )
    assert result["verdict"] == "PASS"

def test_forgetting_missing_metric():
    result = check_forgetting(
        {"accuracy": 0.89}, {},
        "accuracy", tolerance=0.005,
    )
    assert result["verdict"] == "UNKNOWN"


# --- build_update_report ---

def test_report_basic():
    exp = {"experiment_id": "exp-089"}
    plan = {"strategy": "continued_boosting", "model_type": "tree"}
    report = build_update_report(exp, plan)
    assert report["experiment_id"] == "exp-089"
    assert report["family"] == "update"
    assert report["verdict"] == "PENDING"

def test_report_with_metrics():
    exp = {"experiment_id": "exp-089"}
    plan = {"strategy": "continued_boosting", "model_type": "tree"}
    report = build_update_report(
        exp, plan,
        old_data_metrics_before={"accuracy": 0.891},
        old_data_metrics_after={"accuracy": 0.889},
        new_data_metrics={"accuracy": 0.873},
        combined_metrics={"accuracy": 0.885},
        primary_metric="accuracy",
    )
    assert len(report["metric_table"]) == 3
    assert report["forgetting_check"]["verdict"] == "PASS"

def test_report_with_speedup():
    exp = {"experiment_id": "exp-089"}
    plan = {"strategy": "continued_boosting", "model_type": "tree"}
    report = build_update_report(
        exp, plan,
        update_time_seconds=45, full_retrain_time_seconds=720,
    )
    assert report["speedup"] == 16.0


# --- incremental_update ---

def test_update_not_found(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text("")
    result = incremental_update("exp-999", config_path=str(config), log_path=str(log))
    assert "error" in result

def test_update_no_data(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-042", "config": {"model_type": "xgboost"}, "metrics": {"accuracy": 0.85}}\n')
    result = incremental_update("exp-042", config_path=str(config), log_path=str(log))
    assert "error" in result

def test_update_success(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-042", "config": {"model_type": "xgboost", "hyperparams": {"n_estimators": 100}}, "metrics": {"accuracy": 0.85}}\n')
    data = tmp_path / "new.csv"
    data.write_text("col1,col2,label\n1,2,0\n3,4,1\n5,6,0\n")

    result = incremental_update("exp-042", new_data_path=str(data), config_path=str(config), log_path=str(log))
    assert "error" not in result
    assert result["plan"]["strategy"] == "continued_boosting"


# --- _count_data_samples ---

def test_count_csv(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("header\n1\n2\n3\n")
    assert _count_data_samples(str(f)) == 3

def test_count_jsonl(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"a":1}\n{"a":2}\n')
    assert _count_data_samples(str(f)) == 2

def test_count_missing():
    assert _count_data_samples("/nonexistent/path") == 0


# --- format_update_report ---

def test_format_error():
    text = format_update_report({"error": "not found"})
    assert "not found" in text

def test_format_plan():
    report = {
        "experiment_id": "exp-089",
        "plan": {"strategy": "continued_boosting", "model_type": "tree", "instructions": ["Step 1"]},
        "metric_table": [],
        "forgetting_check": None,
        "verdict": "PENDING",
        "generated_at": "2026-04-01",
    }
    text = format_update_report(report)
    assert "continued_boosting" in text
    assert "Step 1" in text

def test_format_with_metrics():
    report = {
        "experiment_id": "exp-089",
        "plan": {"strategy": "fine_tune_with_replay", "model_type": "neural", "instructions": []},
        "metric_table": [
            {"dataset": "Old data", "before": 0.891, "after": 0.889, "delta": -0.002},
        ],
        "forgetting_check": {"verdict": "PASS", "reason": "Within tolerance"},
        "verdict": "PASS",
        "speedup": 16.0,
        "update_time_seconds": 45,
        "full_retrain_time_seconds": 720,
        "generated_at": "2026-04-01",
    }
    text = format_update_report(report)
    assert "0.891" in text
    assert "PASS" in text
    assert "16.0x" in text
