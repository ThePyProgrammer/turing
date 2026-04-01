"""Edge case tests for draft_paper_sections.py."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from scripts.draft_paper_sections import (
    draft_ablation_table,
    draft_hyperparameter_table,
    draft_results_table,
    draft_setup_section,
    get_top_experiments,
    save_paper_sections,
)


def _make_exp(exp_id, model_type="xgboost", accuracy=0.85, **hp):
    return {
        "experiment_id": exp_id,
        "status": "kept",
        "config": {"model_type": model_type, "hyperparams": hp or {"max_depth": 6}},
        "metrics": {"accuracy": accuracy},
    }


def test_top_experiments_empty():
    """Should return empty for no experiments."""
    assert get_top_experiments([], "accuracy", False) == []


def test_top_experiments_no_metric():
    """Should skip experiments without the metric."""
    exps = [{"experiment_id": "exp-001", "status": "kept", "config": {}, "metrics": {}}]
    assert get_top_experiments(exps, "accuracy", False) == []


def test_setup_no_splits():
    """Should handle missing split ratios."""
    config = {"evaluation": {"primary_metric": "mse", "metrics": ["mse"], "lower_is_better": True}}
    section = draft_setup_section(config, [], {}, "markdown")
    assert "lower" in section


def test_results_single_model():
    """Should work with single model type."""
    exps = [_make_exp("exp-001", "xgboost", 0.85)]
    table = draft_results_table(exps, ["accuracy"], "accuracy", False, {}, "markdown")
    assert "xgboost" in table


def test_results_missing_metric():
    """Should show --- for missing metric values."""
    exps = [{"experiment_id": "exp-001", "status": "kept",
             "config": {"model_type": "xgb"}, "metrics": {}}]
    table = draft_results_table(exps, ["accuracy"], "accuracy", False, {}, "markdown")
    assert "---" in table


def test_ablation_skips_failed():
    """Should skip failed ablation runs."""
    studies = {
        "exp-001": {
            "experiment_id": "exp-001",
            "metric": "accuracy",
            "full_model_metric": 0.87,
            "results": [
                {"configuration": "- good", "metric_value": 0.86, "delta": -0.01, "status": "completed"},
                {"configuration": "- bad", "metric_value": None, "delta": None, "status": "failed"},
            ],
        }
    }
    table = draft_ablation_table(studies, "latex")
    assert "good" in table
    assert "bad" not in table  # Failed should be skipped


def test_hyperparams_multiple_models():
    """Should list hyperparams for each model type."""
    exps = [
        _make_exp("exp-001", "xgboost", max_depth=6),
        _make_exp("exp-002", "lightgbm", num_leaves=31),
    ]
    table = draft_hyperparameter_table(exps, "markdown")
    assert "xgboost" in table
    assert "lightgbm" in table


def test_hyperparams_empty():
    """Should handle experiments with no hyperparameters."""
    exps = [{"experiment_id": "exp-001", "status": "kept",
             "config": {"model_type": "xgb", "hyperparams": {}}, "metrics": {"accuracy": 0.85}}]
    table = draft_hyperparameter_table(exps, "markdown")
    assert "Hyperparameters" in table


def test_save_creates_nested_dir(tmp_path):
    """Should create nested output directory."""
    sections = {"setup": "## Setup\ntest"}
    saved = save_paper_sections(sections, str(tmp_path / "deep" / "nested"))
    assert len(saved) == 1
    assert saved[0].exists()


def test_latex_escapes_underscores():
    """LaTeX output should escape underscores in model names."""
    exps = [_make_exp("exp-001", "random_forest", accuracy=0.85)]
    table = draft_results_table(exps, ["accuracy"], "accuracy", False, {}, "latex")
    assert r"random\_forest" in table
