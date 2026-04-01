"""Tests for architecture modification (architecture_surgery.py). Phase 23.4."""

from __future__ import annotations
import pytest
from scripts.architecture_surgery import format_surgery_report, plan_operation

HP_NN = {"hidden_size": 256, "n_layers": 4, "learning_rate": 0.001}
HP_TREE = {"n_estimators": 500, "max_depth": 6}

def test_widen_neural():
    plan = plan_operation("widen", {}, HP_NN, "pytorch", ["2"])
    assert plan["new_config"]["hidden_size"] == 512

def test_widen_tree():
    plan = plan_operation("widen", {}, HP_TREE, "xgboost", ["2"])
    assert plan["new_config"]["n_estimators"] == 1000

def test_narrow_neural():
    plan = plan_operation("narrow", {}, HP_NN, "mlp", ["0.5"])
    assert plan["new_config"]["hidden_size"] == 128

def test_narrow_tree():
    plan = plan_operation("narrow", {}, HP_TREE, "lightgbm", ["0.5"])
    assert plan["new_config"]["n_estimators"] == 250

def test_add_layer():
    plan = plan_operation("add-layer", {}, HP_NN, "pytorch")
    assert plan["new_config"]["n_layers"] == 5

def test_remove_layer():
    plan = plan_operation("remove-layer", {}, HP_NN, "pytorch")
    assert plan["new_config"]["n_layers"] == 3

def test_remove_layer_minimum():
    plan = plan_operation("remove-layer", {}, {"n_layers": 1}, "mlp")
    assert "Cannot remove" in plan["instructions"][0]

def test_deepen_tree():
    plan = plan_operation("deepen", {}, HP_TREE, "xgboost")
    assert plan["new_config"]["max_depth"] == 8

def test_deepen_neural():
    plan = plan_operation("deepen", {}, HP_NN, "pytorch")
    assert plan["new_config"]["n_layers"] == 6

def test_swap_activation():
    plan = plan_operation("swap-activation", {}, HP_NN, "pytorch", ["relu", "gelu"])
    assert plan["new_config"]["activation"] == "gelu"

def test_swap_activation_default():
    plan = plan_operation("swap-activation", {}, HP_NN, "mlp")
    assert plan["new_config"]["activation"] == "gelu"

def test_add_skip():
    plan = plan_operation("add-skip", {}, HP_NN, "pytorch")
    assert plan["new_config"]["skip_connections"] is True

def test_add_norm():
    plan = plan_operation("add-norm", {}, HP_NN, "pytorch", ["layer_norm"])
    assert plan["new_config"]["normalization"] == "layer_norm"

def test_swap_objective():
    plan = plan_operation("swap-objective", {}, HP_TREE, "xgboost", ["logloss", "focal"])
    assert plan["new_config"]["objective"] == "focal"

def test_unknown_operation():
    plan = plan_operation("unknown_op", {}, {}, "pytorch")
    assert "Unknown" in plan["instructions"][0]

def test_format_basic():
    report = {
        "generated_at": "2026-01-01T00:00:00", "experiment_id": "exp-042",
        "plan": {"operation": "widen", "model_type": "pytorch",
                "original_config": {"hidden_size": 256}, "new_config": {"hidden_size": 512},
                "instructions": ["Multiply hidden: 256 → 512"], "param_change": "+300%"},
        "warm_start_from": "exp-042",
    }
    text = format_surgery_report(report)
    assert "Surgery" in text
    assert "256 → 512" in text

def test_format_error():
    assert "ERROR" in format_surgery_report({"error": "Not found"})
