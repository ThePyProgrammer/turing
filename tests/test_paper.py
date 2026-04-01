"""Tests for draft_paper_sections.py.

Phase 14.2: Verifies setup prose, results table formatting, ablation
table, hyperparameter table, LaTeX and markdown output, number accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from scripts.draft_paper_sections import (
    draft_ablation_table,
    draft_hyperparameter_table,
    draft_paper,
    draft_results_table,
    draft_setup_section,
    get_top_experiments,
    group_by_model_type,
    save_paper_sections,
)


def _make_exp(exp_id, model_type="xgboost", accuracy=0.85, f1=0.83, **hyperparams):
    return {
        "experiment_id": exp_id,
        "status": "kept",
        "config": {"model_type": model_type, "hyperparams": hyperparams or {"max_depth": 6, "n_estimators": 100}},
        "metrics": {"accuracy": accuracy, "f1_weighted": f1},
        "description": f"Test {exp_id}",
    }


# --- get_top_experiments ---


def test_top_experiments_sorted():
    """Should return top-K by metric, sorted best first."""
    exps = [
        _make_exp("exp-001", accuracy=0.80),
        _make_exp("exp-002", accuracy=0.90),
        _make_exp("exp-003", accuracy=0.85),
    ]
    top = get_top_experiments(exps, "accuracy", lower_is_better=False, top_k=2)
    assert len(top) == 2
    assert top[0]["experiment_id"] == "exp-002"


def test_top_experiments_lower_is_better():
    """Should respect lower-is-better."""
    exps = [_make_exp("exp-001", accuracy=0.10), _make_exp("exp-002", accuracy=0.05)]
    top = get_top_experiments(exps, "accuracy", lower_is_better=True)
    assert top[0]["experiment_id"] == "exp-002"


def test_top_experiments_skips_discarded():
    """Should skip discarded experiments."""
    exps = [
        _make_exp("exp-001", accuracy=0.80),
        {**_make_exp("exp-002", accuracy=0.99), "status": "discarded"},
    ]
    top = get_top_experiments(exps, "accuracy", lower_is_better=False)
    assert len(top) == 1


# --- group_by_model_type ---


def test_group_by_model():
    """Should group experiments by model type."""
    exps = [
        _make_exp("exp-001", model_type="xgboost"),
        _make_exp("exp-002", model_type="lightgbm"),
        _make_exp("exp-003", model_type="xgboost"),
    ]
    groups = group_by_model_type(exps)
    assert len(groups["xgboost"]) == 2
    assert len(groups["lightgbm"]) == 1


# --- draft_setup_section ---


def test_setup_latex():
    """Should produce LaTeX setup section."""
    config = {
        "task_description": "sentiment analysis",
        "data": {"source": "imdb.csv", "split_ratios": {"train": 0.8, "test": 0.2}, "random_state": 42},
        "evaluation": {"primary_metric": "accuracy", "metrics": ["accuracy", "f1"], "lower_is_better": False},
    }
    section = draft_setup_section(config, [], {}, "latex")
    assert r"\subsection{Experimental Setup}" in section
    assert "accuracy" in section
    assert "imdb.csv" in section


def test_setup_markdown():
    """Should produce markdown setup section."""
    config = {
        "evaluation": {"primary_metric": "accuracy", "metrics": ["accuracy"], "lower_is_better": False},
    }
    section = draft_setup_section(config, [], {}, "markdown")
    assert "## Experimental Setup" in section


def test_setup_with_seed_study():
    """Should mention seed study methodology."""
    config = {"evaluation": {"primary_metric": "accuracy", "metrics": ["accuracy"], "lower_is_better": False}}
    seed_studies = {"exp-001": {"seeds_run": [42, 123, 456, 789, 1024]}}
    section = draft_setup_section(config, [], seed_studies, "latex")
    assert "5 random seeds" in section


# --- draft_results_table ---


def test_results_latex():
    """Should produce LaTeX results table with bold best."""
    exps = [
        _make_exp("exp-001", "xgboost", accuracy=0.872, f1=0.869),
        _make_exp("exp-002", "lightgbm", accuracy=0.865, f1=0.871),
    ]
    table = draft_results_table(exps, ["accuracy", "f1_weighted"], "accuracy", False, {}, "latex")
    assert r"\begin{table}" in table
    assert r"\textbf" in table  # Best should be bold
    assert "xgboost" in table.lower().replace(r"\_", "_")
    assert "0.872" in table


def test_results_markdown():
    """Should produce markdown results table."""
    exps = [_make_exp("exp-001", "xgboost", accuracy=0.85)]
    table = draft_results_table(exps, ["accuracy"], "accuracy", False, {}, "markdown")
    assert "| xgboost |" in table
    assert "**" in table  # Bold best


def test_results_with_seed_study():
    """Should include seed study mean+/-std in results."""
    exps = [_make_exp("exp-001", "xgboost", accuracy=0.872)]
    seed_studies = {
        "exp-001": {"experiment_id": "exp-001", "metric": "accuracy", "mean": 0.868, "std": 0.007},
    }
    table = draft_results_table(exps, ["accuracy"], "accuracy", False, seed_studies, "latex")
    assert "0.868" in table
    assert "0.007" in table


def test_results_number_accuracy():
    """Numbers in the table must match experiment data exactly."""
    exps = [_make_exp("exp-001", "xgboost", accuracy=0.87234)]
    table = draft_results_table(exps, ["accuracy"], "accuracy", False, {}, "markdown")
    assert "0.8723" in table  # Should round to 4dp


# --- draft_ablation_table ---


def test_ablation_latex():
    """Should produce LaTeX ablation table."""
    studies = {
        "exp-001": {
            "experiment_id": "exp-001",
            "metric": "accuracy",
            "full_model_metric": 0.872,
            "results": [
                {"configuration": "- dropout", "metric_value": 0.863, "delta": -0.009, "status": "completed"},
            ],
        }
    }
    table = draft_ablation_table(studies, "latex")
    assert r"\begin{table}" in table
    assert "0.8720" in table
    assert "dropout" in table


def test_ablation_markdown():
    """Should produce markdown ablation table."""
    studies = {
        "exp-001": {
            "experiment_id": "exp-001",
            "metric": "accuracy",
            "full_model_metric": 0.872,
            "results": [
                {"configuration": "- dropout", "metric_value": 0.863, "delta": -0.009, "status": "completed"},
            ],
        }
    }
    table = draft_ablation_table(studies, "markdown")
    assert "## Ablation" in table


def test_ablation_no_studies():
    """Should report no ablation studies available."""
    result = draft_ablation_table({})
    assert "No ablation" in result


# --- draft_hyperparameter_table ---


def test_hyperparams_latex():
    """Should produce LaTeX hyperparameter table."""
    exps = [_make_exp("exp-001", "xgboost", max_depth=6, n_estimators=100)]
    table = draft_hyperparameter_table(exps, "latex")
    assert r"\begin{table}" in table
    assert "max" in table.lower().replace(r"\_", "_")  # max_depth


def test_hyperparams_markdown():
    """Should produce markdown hyperparameter table."""
    exps = [_make_exp("exp-001", "xgboost", max_depth=6)]
    table = draft_hyperparameter_table(exps, "markdown")
    assert "| xgboost |" in table


# --- draft_paper (integration) ---


def test_draft_paper_all_sections(tmp_path):
    """Should generate all four sections."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({
            "evaluation": {"primary_metric": "accuracy", "metrics": ["accuracy"], "lower_is_better": False},
            "data": {"source": "test.csv"},
        }, f)

    log_path = tmp_path / "log.jsonl"
    with open(log_path, "w") as f:
        f.write(json.dumps(_make_exp("exp-001")) + "\n")

    result = draft_paper(
        sections_str=None,
        output_format="latex",
        config_path=str(config_path),
        log_path=str(log_path),
    )
    assert "setup" in result["sections"]
    assert "results" in result["sections"]
    assert "ablation" in result["sections"]
    assert "hyperparameters" in result["sections"]


def test_draft_paper_specific_sections(tmp_path):
    """Should only generate requested sections."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"evaluation": {"primary_metric": "accuracy", "metrics": ["accuracy"]}}, f)

    log_path = tmp_path / "log.jsonl"
    with open(log_path, "w") as f:
        f.write(json.dumps(_make_exp("exp-001")) + "\n")

    result = draft_paper("setup,results", "markdown", str(config_path), str(log_path))
    assert "setup" in result["sections"]
    assert "results" in result["sections"]
    assert "ablation" not in result["sections"]


# --- save_paper_sections ---


def test_save_sections(tmp_path):
    """Should save each section to its own file."""
    sections = {
        "setup": r"\subsection{Setup} test",
        "results": "## Results\n| col |",
    }
    saved = save_paper_sections(sections, str(tmp_path))
    assert len(saved) == 2
    tex_files = [f for f in saved if f.suffix == ".tex"]
    md_files = [f for f in saved if f.suffix == ".md"]
    assert len(tex_files) == 1  # LaTeX content
    assert len(md_files) == 1   # Markdown content
