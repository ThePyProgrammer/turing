"""Tests for ablation study (ablation_study.py).

Phase 11.2: Verifies component detection, delta computation, table
formatting, dead-weight flagging, and LaTeX output.
"""

from __future__ import annotations

import pytest

from scripts.ablation_study import (
    compute_ablation_table,
    detect_ablatable_components,
    format_ablation_table,
    format_latex_table,
    parse_component_list,
    save_ablation,
)


# --- detect_ablatable_components ---


def test_detect_regularization_params():
    """Should detect regularization hyperparameters."""
    config = {
        "model": {
            "hyperparams": {
                "max_depth": 6,
                "reg_alpha": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.9,
            }
        }
    }
    components = detect_ablatable_components(config)
    names = [c["name"] for c in components]
    assert "max_depth" in names
    assert "reg_alpha" in names
    assert "subsample" in names
    assert "colsample_bytree" in names


def test_detect_learning_rate():
    """Should detect low learning rate as ablatable."""
    config = {"model": {"hyperparams": {"learning_rate": 0.01}}}
    components = detect_ablatable_components(config)
    names = [c["name"] for c in components]
    assert "learning_rate" in names


def test_detect_skips_default_values():
    """Should skip params already at their removal value."""
    config = {"model": {"hyperparams": {"reg_alpha": 0, "subsample": 1.0}}}
    components = detect_ablatable_components(config)
    names = [c["name"] for c in components]
    assert "reg_alpha" not in names
    assert "subsample" not in names


def test_detect_complexity_params():
    """Should detect complexity parameters."""
    config = {"model": {"hyperparams": {"n_estimators": 500}}}
    components = detect_ablatable_components(config)
    names = [c["name"] for c in components]
    assert "n_estimators" in names


def test_detect_empty_config():
    """Should return empty list for empty config."""
    assert detect_ablatable_components({}) == []
    assert detect_ablatable_components({"model": {}}) == []
    assert detect_ablatable_components({"model": {"hyperparams": {}}}) == []


# --- parse_component_list ---


def test_parse_components():
    """Should parse comma-separated component names."""
    assert parse_component_list("dropout,feature_X,regularization") == [
        "dropout", "feature_X", "regularization"
    ]


def test_parse_components_with_spaces():
    """Should handle extra whitespace."""
    assert parse_component_list(" a , b , c ") == ["a", "b", "c"]


def test_parse_components_empty():
    """Should handle empty string."""
    assert parse_component_list("") == []


# --- compute_ablation_table ---


def test_ablation_table_basic():
    """Should compute correct deltas."""
    full_metric = 0.872
    results = [
        {
            "component": {"name": "dropout", "description": "dropout rate"},
            "metric_value": 0.863,
        },
        {
            "component": {"name": "feature_X", "description": "feature X"},
            "metric_value": 0.851,
        },
    ]
    rows = compute_ablation_table(full_metric, results, "accuracy", lower_is_better=False)

    assert len(rows) == 2
    # Sorted by absolute delta
    assert abs(rows[0]["delta"]) >= abs(rows[1]["delta"])
    # feature_X has larger impact
    assert rows[0]["configuration"] == "− feature_X"


def test_ablation_table_dead_weight():
    """Should flag dead-weight components (removing improves metric)."""
    full_metric = 0.872
    results = [
        {
            "component": {"name": "useless_feature", "description": "useless"},
            "metric_value": 0.880,  # Better without it!
        },
    ]
    rows = compute_ablation_table(full_metric, results, "accuracy", lower_is_better=False)
    assert rows[0]["is_dead_weight"] is True


def test_ablation_table_dead_weight_lower_is_better():
    """Dead weight for lower-is-better metrics."""
    full_metric = 0.10
    results = [
        {
            "component": {"name": "bad_feature", "description": "bad"},
            "metric_value": 0.08,  # Lower is better, so 0.08 < 0.10 = improvement
        },
    ]
    rows = compute_ablation_table(full_metric, results, "mse", lower_is_better=True)
    assert rows[0]["is_dead_weight"] is True


def test_ablation_table_failed_run():
    """Should handle failed ablation runs."""
    results = [
        {
            "component": {"name": "broken", "description": "broken"},
            "metric_value": None,
            "status": "failed",
        },
    ]
    rows = compute_ablation_table(0.85, results, "accuracy", lower_is_better=False)
    assert rows[0]["status"] == "failed"


# --- format_ablation_table ---


def test_format_table():
    """Should produce markdown table."""
    rows = [
        {
            "configuration": "− dropout",
            "component": {"name": "dropout", "description": "dropout rate"},
            "metric_value": 0.863,
            "delta": -0.009,
            "delta_pct": -1.03,
            "is_dead_weight": False,
            "status": "completed",
        },
    ]
    table = format_ablation_table(0.872, rows, "accuracy", False)
    assert "Ablation Study" in table
    assert "Full model" in table
    assert "0.8720" in table
    assert "dropout" in table


def test_format_table_dead_weight_section():
    """Should include dead-weight section when applicable."""
    rows = [
        {
            "configuration": "− useless",
            "component": {"name": "useless", "description": "useless feature"},
            "metric_value": 0.880,
            "delta": 0.008,
            "delta_pct": 0.92,
            "is_dead_weight": True,
            "status": "completed",
        },
    ]
    table = format_ablation_table(0.872, rows, "accuracy", False)
    assert "Dead-Weight" in table


# --- format_latex_table ---


def test_format_latex():
    """Should produce valid LaTeX."""
    rows = [
        {
            "configuration": "− dropout",
            "component": {"name": "dropout", "description": "dropout"},
            "metric_value": 0.863,
            "delta": -0.009,
            "delta_pct": -1.03,
            "is_dead_weight": False,
            "status": "completed",
        },
    ]
    latex = format_latex_table(0.872, rows, "accuracy")
    assert r"\begin{table}" in latex
    assert r"\end{table}" in latex
    assert "0.8720" in latex
    assert "dropout" in latex


def test_format_latex_escapes_underscores():
    """Should escape underscores in LaTeX."""
    rows = [
        {
            "configuration": "− feature_X",
            "component": {"name": "feature_X", "description": "test"},
            "metric_value": 0.85,
            "delta": -0.02,
            "status": "completed",
        },
    ]
    latex = format_latex_table(0.87, rows, "accuracy")
    assert r"feature\_X" in latex


# --- save_ablation ---


def test_save_ablation(tmp_path):
    """Should save YAML and return correct path."""
    study = {"experiment_id": "exp-042", "results": []}
    filepath = save_ablation(study, str(tmp_path))
    assert filepath.exists()
    assert filepath.name == "exp-042-ablation.yaml"
