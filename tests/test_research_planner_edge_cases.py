"""Edge case tests for research planning assistant (research_planner.py).

Phase 29.3: Zero budget, no history, single family, extreme values.
"""

from __future__ import annotations

import pytest
import yaml

from scripts.research_planner import (
    compute_family_roi,
    adjust_priorities,
    allocate_budget,
    generate_plan,
    create_research_plan,
    save_plan_report,
    format_plan_report,
    STRATEGIES,
)


# --- ROI edge cases ---

def test_roi_single_experiment():
    exps = [{"family": "tuning", "metrics": {"accuracy": 0.85}, "status": "kept"}]
    roi = compute_family_roi(exps, "accuracy")
    assert roi["tuning"]["roi_per_experiment"] == 0

def test_roi_lower_is_better():
    exps = [
        {"family": "tuning", "metrics": {"loss": 0.20}, "status": "kept"},
        {"family": "tuning", "metrics": {"loss": 0.15}, "status": "kept"},
        {"family": "tuning", "metrics": {"loss": 0.10}, "status": "kept"},
    ]
    roi = compute_family_roi(exps, "loss", lower_is_better=True)
    assert roi["tuning"]["total_improvement"] > 0

def test_roi_missing_metrics():
    exps = [{"family": "tuning", "config": {}, "status": "kept"}] * 5
    roi = compute_family_roi(exps, "accuracy")
    assert roi["tuning"]["best_metric"] is None


# --- Priority edge cases ---

def test_priorities_empty_roi():
    priorities = adjust_priorities(STRATEGIES, {}, [], "accuracy")
    assert abs(sum(priorities.values()) - 1.0) < 0.01

def test_priorities_all_exhausted():
    roi = {"tuning": {"exhausted": True, "roi_per_experiment": 0, "n_experiments": 20}}
    priorities = adjust_priorities(STRATEGIES, roi, [{}] * 20, "accuracy")
    assert abs(sum(priorities.values()) - 1.0) < 0.01


# --- Allocation edge cases ---

def test_allocate_large_budget():
    priorities = {"a": 0.6, "b": 0.4}
    allocation = allocate_budget(priorities, 100)
    assert sum(allocation.values()) == 100

def test_allocate_budget_1():
    priorities = {"a": 0.5, "b": 0.3, "c": 0.2}
    allocation = allocate_budget(priorities, 1)
    assert sum(allocation.values()) == 1

def test_allocate_many_strategies():
    priorities = {f"s{i}": 1/10 for i in range(10)}
    allocation = allocate_budget(priorities, 20)
    assert sum(allocation.values()) == 20


# --- Plan edge cases ---

def test_plan_single_phase():
    plan = generate_plan({"feature_engineering": 10}, STRATEGIES, {}, 0.85)
    assert len(plan["phases"]) == 1
    assert plan["total_experiments"] == 10

def test_plan_zero_allocation():
    allocation = {"feature_engineering": 0, "model_search": 0}
    plan = generate_plan(allocation, STRATEGIES, {})
    assert plan["total_experiments"] == 0

def test_plan_large_allocation():
    allocation = {"feature_engineering": 50}
    plan = generate_plan(allocation, STRATEGIES, {}, 0.85)
    assert plan["total_experiments"] == 50
    # Templates should cycle
    assert len(plan["phases"][0]["experiments"]) == 50


# --- create_research_plan edge cases ---

def test_plan_missing_config(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"metrics": {"accuracy": 0.85}, "status": "kept", "family": "tuning", "config": {}}\n')
    result = create_research_plan(budget=5, config_path=str(tmp_path / "missing.yaml"), log_path=str(log))
    # Should work with defaults
    assert "plan" in result

def test_plan_budget_1(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-001", "status": "kept", "family": "tuning", "config": {}, "metrics": {"accuracy": 0.85}}\n' * 5)
    result = create_research_plan(budget=1, config_path=str(config), log_path=str(log))
    assert result["plan"]["total_experiments"] == 1


# --- save_plan_report ---

def test_save_report(tmp_path):
    report = {"budget": 20, "plan": {"phases": []}, "generated_at": "2026-04-01"}
    path = save_plan_report(report, str(tmp_path / "plans"))
    assert path.exists()
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data["budget"] == 20


# --- Format edge cases ---

def test_format_with_no_best():
    report = {
        "budget": 10,
        "goal": None,
        "plan": {
            "phases": [{"label": "Search", "n_experiments": 10, "budget_pct": 100,
                         "rationale": "Explore", "experiments": [{"number": 1, "description": "Try"}],
                         "expected_gain": 0.03}],
            "total_experiments": 10,
            "expected_metric": None,
            "expected_gain": 0.03,
        },
        "generated_at": "2026-04-01",
    }
    text = format_plan_report(report)
    assert "+0.03" in text

def test_format_many_phases():
    phases = [
        {"label": f"Phase {i}", "n_experiments": 2, "budget_pct": 20,
         "rationale": "Test", "experiments": [{"number": i, "description": f"Exp {i}"}],
         "expected_gain": 0.01}
        for i in range(5)
    ]
    report = {
        "budget": 10, "goal": "test",
        "plan": {"phases": phases, "total_experiments": 10, "expected_metric": None, "expected_gain": 0.05},
        "generated_at": "2026-04-01",
    }
    text = format_plan_report(report)
    assert "Phase A:" in text
    assert "Phase E:" in text
