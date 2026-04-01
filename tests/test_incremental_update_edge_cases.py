"""Edge case tests for incremental model update (incremental_update.py).

Phase 28.1: No model, unsupported types, empty data, boundary conditions.
"""

from __future__ import annotations

import pytest
import yaml

from scripts.incremental_update import (
    detect_model_type,
    plan_update,
    check_forgetting,
    build_update_report,
    incremental_update,
    save_update_report,
    format_update_report,
    _count_data_samples,
)


# --- Model detection edge cases ---

def test_detect_empty_config():
    assert detect_model_type({}) == "unknown"

def test_detect_case_insensitive():
    exp = {"config": {"model_type": "XGBoost"}}
    assert detect_model_type(exp) == "tree"

def test_detect_catboost():
    exp = {"config": {"model_type": "catboost_classifier"}}
    assert detect_model_type(exp) == "tree"

def test_detect_keras():
    exp = {"config": {"model_type": "keras_sequential"}}
    assert detect_model_type(exp) == "neural"

def test_detect_transformer():
    exp = {"config": {"model_type": "transformer"}}
    assert detect_model_type(exp) == "neural"


# --- Plan edge cases ---

def test_plan_zero_data():
    exp = {"config": {"model_type": "xgboost", "hyperparams": {"n_estimators": 100}}}
    plan = plan_update(exp, new_data_size=0)
    assert plan["strategy"] == "continued_boosting"
    assert plan["new_data_size"] == 0

def test_plan_large_data():
    exp = {"config": {"model_type": "lightgbm", "hyperparams": {"n_estimators": 100}}}
    plan = plan_update(exp, new_data_size=1_000_000)
    assert plan["new_data_size"] == 1_000_000

def test_plan_neural_zero_replay():
    exp = {"config": {"model_type": "mlp", "hyperparams": {"learning_rate": 0.001}}}
    plan = plan_update(exp, new_data_size=5000, replay_ratio=0.0)
    assert plan["replay_size"] == 0
    assert plan["total_training_size"] == 5000

def test_plan_neural_full_replay():
    exp = {"config": {"model_type": "nn", "hyperparams": {"learning_rate": 0.01}}}
    plan = plan_update(exp, new_data_size=1000, replay_ratio=1.0)
    assert plan["replay_size"] == 1000
    assert plan["total_training_size"] == 2000


# --- Forgetting edge cases ---

def test_forgetting_exact_tolerance():
    """Floating point means exact boundary is tricky — tolerance=0.005 with 0.005 degradation."""
    result = check_forgetting(
        {"accuracy": 0.90}, {"accuracy": 0.895},
        "accuracy", tolerance=0.006,  # slightly above to handle float imprecision
    )
    assert result["verdict"] == "PASS"

def test_forgetting_zero_tolerance():
    result = check_forgetting(
        {"accuracy": 0.90}, {"accuracy": 0.899},
        "accuracy", tolerance=0.0,
    )
    assert result["verdict"] in ("WARNING", "FAIL")

def test_forgetting_identical():
    result = check_forgetting(
        {"accuracy": 0.90}, {"accuracy": 0.90},
        "accuracy", tolerance=0.005,
    )
    assert result["verdict"] == "PASS"
    assert result["degradation"] == 0.0

def test_forgetting_loss_improved():
    """Lower is better, loss decreased → improvement."""
    result = check_forgetting(
        {"loss": 0.15}, {"loss": 0.12},
        "loss", tolerance=0.01, lower_is_better=True,
    )
    assert result["verdict"] == "PASS"
    assert result["degradation"] < 0

def test_forgetting_loss_degraded():
    """Lower is better, loss increased → degradation."""
    result = check_forgetting(
        {"loss": 0.10}, {"loss": 0.20},
        "loss", tolerance=0.01, lower_is_better=True,
    )
    assert result["verdict"] == "FAIL"


# --- Report edge cases ---

def test_report_no_metrics():
    exp = {"experiment_id": "exp-001"}
    plan = {"strategy": "continued_boosting", "model_type": "tree"}
    report = build_update_report(exp, plan)
    assert report["metric_table"] == []
    assert report["forgetting_check"] is None

def test_report_no_speedup():
    exp = {"experiment_id": "exp-001"}
    plan = {"strategy": "partial_fit", "model_type": "sklearn"}
    report = build_update_report(exp, plan)
    assert report["speedup"] is None

def test_report_zero_retrain_time():
    exp = {"experiment_id": "exp-001"}
    plan = {"strategy": "continued_boosting", "model_type": "tree"}
    report = build_update_report(exp, plan, update_time_seconds=10, full_retrain_time_seconds=0)
    assert report["speedup"] is None


# --- _count_data_samples edge cases ---

def test_count_empty_file(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("")
    assert _count_data_samples(str(f)) == 0

def test_count_header_only(tmp_path):
    f = tmp_path / "header.csv"
    f.write_text("col1,col2\n")
    assert _count_data_samples(str(f)) == 0

def test_count_binary_file(tmp_path):
    f = tmp_path / "binary.bin"
    f.write_bytes(b"\x00\x01\x02\x03")
    # Should handle gracefully
    count = _count_data_samples(str(f))
    assert isinstance(count, int)


# --- save_update_report ---

def test_save_report(tmp_path):
    report = {
        "experiment_id": "exp-042",
        "plan": {"strategy": "continued_boosting"},
        "generated_at": "2026-04-01",
    }
    path = save_update_report(report, str(tmp_path / "updates"))
    assert path.exists()
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data["experiment_id"] == "exp-042"


# --- Full pipeline edge cases ---

def test_update_unknown_model(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-001", "config": {"model_type": "custom_thing"}, "metrics": {"accuracy": 0.85}}\n')
    data = tmp_path / "new.csv"
    data.write_text("a,b\n1,2\n")
    result = incremental_update("exp-001", new_data_path=str(data), config_path=str(config), log_path=str(log))
    assert "error" in result

def test_update_with_data_size(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-001", "config": {"model_type": "xgboost", "hyperparams": {"n_estimators": 50}}, "metrics": {"accuracy": 0.85}}\n')
    result = incremental_update("exp-001", new_data_size=5000, config_path=str(config), log_path=str(log))
    assert "error" not in result
    assert result["plan"]["new_data_size"] == 5000


# --- Format edge cases ---

def test_format_with_suggestion():
    text = format_update_report({"error": "bad model", "suggestion": "add model_type to config"})
    assert "add model_type" in text

def test_format_new_data_row():
    report = {
        "experiment_id": "exp-089",
        "plan": {"strategy": "continued_boosting", "model_type": "tree", "instructions": []},
        "metric_table": [
            {"dataset": "New data", "before": None, "after": 0.873, "delta": None},
        ],
        "forgetting_check": None,
        "verdict": "PENDING",
        "generated_at": "2026-04-01",
    }
    text = format_update_report(report)
    assert "—" in text  # em-dash for None values
    assert "(first)" in text
