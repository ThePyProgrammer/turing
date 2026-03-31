"""Tests for research briefing generator (generate_brief.py).

Verifies the intelligence reporting that closes the taste-leverage loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.generate_brief import (
    compute_campaign_summary,
    compute_trajectory,
    find_best,
    format_brief,
    generate_brief,
    identify_model_types,
    load_experiments,
    load_reproductions,
    load_seed_studies,
)


def _make_exp(exp_id: str, status: str = "kept", accuracy: float = 0.85, model_type: str = "xgboost") -> dict:
    return {
        "experiment_id": exp_id,
        "timestamp": "2026-03-31T12:00:00+00:00",
        "status": status,
        "config": {"model_type": model_type},
        "metrics": {"accuracy": accuracy},
        "description": f"Test {exp_id}",
    }


def _write_log(path: Path, experiments: list[dict]) -> None:
    with open(path, "w") as f:
        for e in experiments:
            f.write(json.dumps(e) + "\n")


# --- compute_campaign_summary ---


def test_campaign_summary_basic():
    exps = [
        _make_exp("exp-001", "kept"),
        _make_exp("exp-002", "discarded"),
        _make_exp("exp-003", "kept"),
    ]
    summary = compute_campaign_summary(exps)
    assert summary["total"] == 3
    assert summary["kept"] == 2
    assert summary["discarded"] == 1
    assert summary["keep_rate"] == 0.67


def test_campaign_summary_empty():
    summary = compute_campaign_summary([])
    assert summary["total"] == 0
    assert summary["keep_rate"] == 0


# --- find_best ---


def test_find_best_higher_is_better():
    exps = [
        _make_exp("exp-001", "kept", accuracy=0.80),
        _make_exp("exp-002", "kept", accuracy=0.90),
        _make_exp("exp-003", "kept", accuracy=0.85),
    ]
    best = find_best(exps, "accuracy", lower_is_better=False)
    assert best["experiment_id"] == "exp-002"


def test_find_best_skips_discarded():
    exps = [
        _make_exp("exp-001", "kept", accuracy=0.80),
        _make_exp("exp-002", "discarded", accuracy=0.99),
    ]
    best = find_best(exps, "accuracy", lower_is_better=False)
    assert best["experiment_id"] == "exp-001"


def test_find_best_empty():
    assert find_best([], "accuracy", False) is None


# --- compute_trajectory ---


def test_trajectory_tracks_improvement():
    exps = [
        _make_exp("exp-001", "kept", accuracy=0.80),
        _make_exp("exp-002", "kept", accuracy=0.85),
        _make_exp("exp-003", "kept", accuracy=0.83),  # regression, but best_so_far stays 0.85
    ]
    traj = compute_trajectory(exps, "accuracy", lower_is_better=False)
    assert len(traj) == 3
    assert traj[0]["best_so_far"] == 0.80
    assert traj[1]["best_so_far"] == 0.85
    assert traj[2]["best_so_far"] == 0.85  # didn't regress


# --- identify_model_types ---


def test_model_types_grouped():
    exps = [
        _make_exp("exp-001", "kept", model_type="xgboost"),
        _make_exp("exp-002", "discarded", model_type="xgboost"),
        _make_exp("exp-003", "kept", model_type="lightgbm"),
    ]
    types = identify_model_types(exps)
    assert types["xgboost"]["total"] == 2
    assert types["xgboost"]["kept"] == 1
    assert types["lightgbm"]["total"] == 1


# --- generate_brief (integration) ---


def test_generate_brief_with_data(tmp_path: Path):
    """Full brief generation with sample data."""
    # Write config
    config = {
        "evaluation": {"primary_metric": "accuracy", "lower_is_better": False},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Write experiments
    log_path = tmp_path / "log.jsonl"
    _write_log(log_path, [
        _make_exp("exp-001", "kept", 0.80),
        _make_exp("exp-002", "discarded", 0.75),
        _make_exp("exp-003", "kept", 0.85),
    ])

    # Write hypotheses
    hyp_path = tmp_path / "hypotheses.yaml"
    with open(hyp_path, "w") as f:
        yaml.dump([{"id": "hyp-001", "description": "Try LightGBM", "status": "queued", "source": "human", "priority": "high"}], f)

    brief = generate_brief(str(config_path), str(log_path), str(hyp_path))

    assert "# Research Briefing" in brief
    assert "Campaign Summary" in brief
    assert "Current Best" in brief
    assert "exp-003" in brief  # best experiment
    assert "Hypothesis Queue" in brief
    assert "Try LightGBM" in brief
    assert "Recommendations" in brief


def test_generate_brief_empty(tmp_path: Path):
    """Brief with no data should not crash."""
    brief = generate_brief(
        str(tmp_path / "nonexistent.yaml"),
        str(tmp_path / "nonexistent.jsonl"),
        str(tmp_path / "nonexistent_hyp.yaml"),
    )
    assert "# Research Briefing" in brief
    assert "No kept experiments yet" in brief


# --- Seed study integration ---


def test_brief_includes_seed_studies():
    """Brief should include seed study section when studies exist."""
    seed_studies = [
        {
            "experiment_id": "exp-042",
            "metric": "accuracy",
            "mean": 0.8678,
            "std": 0.0071,
            "cv_percent": 0.82,
            "seed_sensitive": False,
            "ci_95": [0.859, 0.877],
            "seeds_run": [42, 123, 456, 789, 1024],
        },
    ]
    brief = format_brief(
        {"total": 5, "kept": 3, "discarded": 2, "keep_rate": 0.6,
         "first_experiment": "2026-01-01T00:00:00", "last_experiment": "2026-03-31T00:00:00"},
        _make_exp("exp-042", "kept", 0.87),
        [{"experiment_id": "exp-042", "value": 0.87, "best_so_far": 0.87}],
        {"xgboost": {"total": 5, "kept": 3, "discarded": 2}},
        [],
        "accuracy", False,
        seed_studies=seed_studies,
    )
    assert "Seed Studies" in brief
    assert "exp-042" in brief
    assert "STABLE" in brief


def test_brief_includes_sensitive_seed_warning():
    """Brief should warn about seed-sensitive results."""
    seed_studies = [
        {
            "experiment_id": "exp-001",
            "metric": "accuracy",
            "mean": 0.80,
            "std": 0.10,
            "cv_percent": 12.5,
            "seed_sensitive": True,
            "ci_95": [0.55, 1.05],
            "seeds_run": [42, 123, 456],
        },
    ]
    brief = format_brief(
        {"total": 1, "kept": 1, "discarded": 0, "keep_rate": 1.0,
         "first_experiment": "2026-01-01T00:00:00", "last_experiment": "2026-01-01T00:00:00"},
        _make_exp("exp-001", "kept"),
        [],
        {},
        [],
        "accuracy", False,
        seed_studies=seed_studies,
    )
    assert "SEED-SENSITIVE" in brief
    assert "Report distributions" in brief.lower() or "report distributions" in brief.lower()


def test_brief_includes_reproductions():
    """Brief should include reproducibility section when reports exist."""
    reproductions = [
        {
            "experiment_id": "exp-042",
            "verdict": "approximately_reproducible",
            "reason": "Within 2.0% tolerance",
        },
    ]
    brief = format_brief(
        {"total": 1, "kept": 1, "discarded": 0, "keep_rate": 1.0,
         "first_experiment": "2026-01-01T00:00:00", "last_experiment": "2026-01-01T00:00:00"},
        _make_exp("exp-042", "kept"),
        [],
        {},
        [],
        "accuracy", False,
        reproductions=reproductions,
    )
    assert "Reproducibility" in brief
    assert "PASS" in brief


def test_brief_no_seed_studies():
    """Brief should omit seed study section when no studies exist."""
    brief = format_brief(
        {"total": 1, "kept": 1, "discarded": 0, "keep_rate": 1.0,
         "first_experiment": "2026-01-01T00:00:00", "last_experiment": "2026-01-01T00:00:00"},
        _make_exp("exp-001", "kept"),
        [],
        {},
        [],
        "accuracy", False,
    )
    assert "Seed Studies" not in brief
    assert "Reproducibility" not in brief


# --- load_seed_studies / load_reproductions ---


def test_load_seed_studies_from_dir(tmp_path: Path):
    """Should load YAML files from seed studies directory."""
    seed_dir = tmp_path / "seed_studies"
    seed_dir.mkdir()
    study = {"experiment_id": "exp-001", "mean": 0.85, "std": 0.01}
    with open(seed_dir / "exp-001-seeds.yaml", "w") as f:
        yaml.dump(study, f)
    studies = load_seed_studies(str(seed_dir))
    assert len(studies) == 1
    assert studies[0]["experiment_id"] == "exp-001"


def test_load_seed_studies_empty_dir(tmp_path: Path):
    """Should return empty list for nonexistent directory."""
    studies = load_seed_studies(str(tmp_path / "nonexistent"))
    assert studies == []


def test_load_reproductions_from_dir(tmp_path: Path):
    """Should load YAML files from reproductions directory."""
    repro_dir = tmp_path / "reproductions"
    repro_dir.mkdir()
    report = {"experiment_id": "exp-001", "verdict": "reproducible"}
    with open(repro_dir / "exp-001-repro.yaml", "w") as f:
        yaml.dump(report, f)
    reports = load_reproductions(str(repro_dir))
    assert len(reports) == 1
    assert reports[0]["verdict"] == "reproducible"


def test_load_reproductions_empty_dir(tmp_path: Path):
    """Should return empty list for nonexistent directory."""
    reports = load_reproductions(str(tmp_path / "nonexistent"))
    assert reports == []
