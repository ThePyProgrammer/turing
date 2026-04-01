"""Tests for internal model diagnostics (model_xray.py).

Phase 21.1: Verifies neural/tree/sklearn diagnostics, issue detection, report formatting.
"""

from __future__ import annotations

import pytest

from scripts.model_xray import (
    diagnose_neural_layers,
    diagnose_sklearn_model,
    diagnose_tree_model,
    format_xray_report,
    xray_model,
)


# --- diagnose_neural_layers ---

def test_neural_healthy():
    stats = [{"name": "l1", "grad_mean": 0.01, "grad_max": 0.1, "dead_pct": 0, "weight_std": 0.3}]
    result = diagnose_neural_layers(stats)
    assert result["model_type"] == "neural"
    assert len(result["issues"]) == 0

def test_neural_dead_gradient():
    stats = [{"name": "l1", "grad_mean": 0.01, "grad_max": 0.1},
             {"name": "l2", "grad_mean": 0.0, "grad_max": 0.0}]
    result = diagnose_neural_layers(stats)
    issues = [i for i in result["issues"] if i["issue"] == "dead_gradient"]
    assert len(issues) == 1

def test_neural_dead_neurons():
    stats = [{"name": "l1", "grad_mean": 0.01, "grad_max": 0.1, "dead_pct": 25}]
    result = diagnose_neural_layers(stats)
    issues = [i for i in result["issues"] if i["issue"] == "dead_neurons"]
    assert len(issues) == 1

def test_neural_exploding_gradient():
    stats = [{"name": "l1", "grad_mean": 0.01, "grad_max": 0.1},
             {"name": "l2", "grad_mean": 0.01, "grad_max": 50.0}]
    result = diagnose_neural_layers(stats)
    issues = [i for i in result["issues"] if i["issue"] == "exploding_gradient"]
    assert len(issues) == 1

def test_neural_sparse_weights():
    stats = [{"name": "l1", "grad_mean": 0.01, "grad_max": 0.1, "near_zero_pct": 60}]
    result = diagnose_neural_layers(stats)
    issues = [i for i in result["issues"] if i["issue"] == "sparse_weights"]
    assert len(issues) == 1

def test_neural_empty():
    result = diagnose_neural_layers([])
    assert result["issues"] == []

def test_neural_vanishing():
    stats = [{"name": "l1", "grad_mean": 1.0, "grad_max": 2.0},
             {"name": "l2", "grad_mean": 0.001, "grad_max": 0.002}]
    result = diagnose_neural_layers(stats)
    issues = [i for i in result["issues"] if i["issue"] == "vanishing_gradient"]
    assert len(issues) == 1

# --- diagnose_tree_model ---

def test_tree_healthy():
    stats = {"n_trees": 100, "avg_depth": 5, "max_depth_allowed": 8,
             "feature_split_freq": {"f1": 30, "f2": 25, "f3": 20}, "leaf_purity": 0.85}
    result = diagnose_tree_model(stats)
    assert result["model_type"] == "tree"
    assert len(result["issues"]) == 0

def test_tree_underutilized_depth():
    stats = {"n_trees": 100, "avg_depth": 2, "max_depth_allowed": 12,
             "feature_split_freq": {}, "leaf_purity": 0.7}
    result = diagnose_tree_model(stats)
    issues = [i for i in result["issues"] if i["issue"] == "underutilized_depth"]
    assert len(issues) == 1

def test_tree_feature_dominance():
    stats = {"n_trees": 50, "avg_depth": 6, "max_depth_allowed": 8,
             "feature_split_freq": {"f1": 80, "f2": 10, "f3": 10}, "leaf_purity": 0.8}
    result = diagnose_tree_model(stats)
    issues = [i for i in result["issues"] if i["issue"] == "feature_dominance"]
    assert len(issues) == 1

def test_tree_overfitting():
    stats = {"n_trees": 100, "avg_depth": 8, "max_depth_allowed": 8,
             "feature_split_freq": {}, "leaf_purity": 0.998}
    result = diagnose_tree_model(stats)
    issues = [i for i in result["issues"] if i["issue"] == "overfitting_risk"]
    assert len(issues) == 1

# --- diagnose_sklearn_model ---

def test_sklearn_healthy():
    stats = {"model_type": "logistic", "coefficients": [0.5, -0.3, 0.2]}
    result = diagnose_sklearn_model(stats)
    assert len(result["issues"]) == 0

def test_sklearn_large_coef():
    stats = {"model_type": "logistic", "coefficients": [150.0, -0.1]}
    result = diagnose_sklearn_model(stats)
    issues = [i for i in result["issues"] if i["issue"] == "large_coefficients"]
    assert len(issues) == 1

def test_sklearn_importance_concentrated():
    stats = {"model_type": "rf", "feature_importances": [0.8, 0.15, 0.03, 0.01, 0.01]}
    result = diagnose_sklearn_model(stats)
    issues = [i for i in result["issues"] if i["issue"] == "importance_concentrated"]
    assert len(issues) == 1

def test_sklearn_empty():
    result = diagnose_sklearn_model({"model_type": "custom"})
    assert len(result["issues"]) == 0

# --- xray_model ---

def test_xray_neural():
    report = xray_model(exp_id="exp-042", layer_stats=[{"name": "l1", "grad_mean": 0.01, "grad_max": 0.1}])
    assert report["diagnosis"]["model_type"] == "neural"

def test_xray_tree():
    report = xray_model(exp_id="exp-001", tree_stats={"n_trees": 50, "avg_depth": 4, "max_depth_allowed": 6, "feature_split_freq": {}, "leaf_purity": 0.8})
    assert report["diagnosis"]["model_type"] == "tree"

def test_xray_no_stats():
    report = xray_model(exp_id="exp-001")
    assert report["diagnosis"]["model_type"] == "unknown"

# --- format_xray_report ---

def test_format_neural():
    report = {"generated_at": "2026-01-01T00:00:00", "experiment_id": "exp-042",
              "diagnosis": {"model_type": "neural",
                           "layers": [{"name": "l1", "grad_mean": 0.01, "grad_max": 0.1, "dead_pct": 0, "weight_std": 0.3}],
                           "issues": []}, "n_issues": 0}
    text = format_xray_report(report)
    assert "X-Ray" in text
    assert "exp-042" in text

def test_format_issues():
    report = {"generated_at": "2026-01-01T00:00:00", "experiment_id": "exp-001",
              "diagnosis": {"model_type": "neural", "layers": [],
                           "issues": [{"layer": "l2", "issue": "dead_neurons", "severity": "high", "message": "23% dead"}]},
              "n_issues": 1}
    text = format_xray_report(report)
    assert "23% dead" in text

def test_format_error():
    assert "ERROR" in format_xray_report({"error": "fail"})
