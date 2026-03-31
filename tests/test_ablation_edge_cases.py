"""Edge case tests for ablation_study.py."""

from __future__ import annotations

import yaml
import pytest

from scripts.ablation_study import (
    compute_ablation_table,
    detect_ablatable_components,
    format_ablation_table,
    format_latex_table,
    save_ablation,
)


def test_detect_all_regularization_types():
    """Should detect all regularization parameter types."""
    config = {
        "model": {
            "hyperparams": {
                "gamma": 0.1,
                "min_child_weight": 5,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "dropout_rate": 0.3,
                "weight_decay": 0.01,
            }
        }
    }
    components = detect_ablatable_components(config)
    names = {c["name"] for c in components}
    assert "gamma" in names
    assert "min_child_weight" in names
    assert "dropout_rate" in names
    assert "weight_decay" in names


def test_detect_high_learning_rate_skipped():
    """Should not ablate already-high learning rate."""
    config = {"model": {"hyperparams": {"learning_rate": 0.8}}}
    components = detect_ablatable_components(config)
    names = [c["name"] for c in components]
    assert "learning_rate" not in names


def test_ablation_table_zero_full_metric():
    """Should handle zero full metric without division error."""
    results = [{
        "component": {"name": "test", "description": "test"},
        "metric_value": 0.1,
    }]
    rows = compute_ablation_table(0.0, results, "metric", lower_is_better=False)
    assert rows[0]["delta_pct"] == 0


def test_ablation_table_many_components():
    """Should handle many ablation results."""
    results = [
        {"component": {"name": f"comp_{i}", "description": f"component {i}"}, "metric_value": 0.85 - i * 0.01}
        for i in range(20)
    ]
    rows = compute_ablation_table(0.87, results, "accuracy", lower_is_better=False)
    assert len(rows) == 20
    # Should be sorted by absolute delta
    for i in range(len(rows) - 1):
        if rows[i]["delta"] is not None and rows[i + 1]["delta"] is not None:
            assert abs(rows[i]["delta"]) >= abs(rows[i + 1]["delta"])


def test_format_table_failed_runs():
    """Should handle all-failed runs in table."""
    rows = [
        {
            "configuration": "− broken",
            "component": {"name": "broken", "description": "broken"},
            "metric_value": None,
            "delta": None,
            "delta_pct": None,
            "is_dead_weight": False,
            "status": "failed",
        },
    ]
    table = format_ablation_table(0.85, rows, "accuracy", False)
    assert "FAILED" in table


def test_latex_skips_failed():
    """LaTeX output should skip failed runs."""
    rows = [
        {
            "configuration": "− good",
            "component": {"name": "good", "description": "good"},
            "metric_value": 0.85,
            "delta": -0.02,
            "status": "completed",
        },
        {
            "configuration": "− bad",
            "component": {"name": "bad", "description": "bad"},
            "metric_value": None,
            "delta": None,
            "status": "failed",
        },
    ]
    latex = format_latex_table(0.87, rows, "accuracy")
    assert "good" in latex
    assert "bad" not in latex


def test_save_creates_nested_dir(tmp_path):
    """Should create nested output directory."""
    nested = tmp_path / "deep" / "nested"
    study = {"experiment_id": "exp-001", "results": []}
    filepath = save_ablation(study, str(nested))
    assert filepath.exists()


def test_save_roundtrip(tmp_path):
    """Saved YAML should round-trip correctly."""
    study = {
        "experiment_id": "exp-042",
        "metric": "accuracy",
        "full_model_metric": 0.872,
        "dead_weight": ["useless_feature"],
    }
    filepath = save_ablation(study, str(tmp_path))
    with open(filepath) as f:
        loaded = yaml.safe_load(f)
    assert loaded["experiment_id"] == "exp-042"
    assert loaded["dead_weight"] == ["useless_feature"]
