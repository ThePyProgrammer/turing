"""Tests for research planning assistant (research_planner.py).

Phase 29.3: Verifies ROI computation, priority adjustment, budget allocation, plan generation.
"""

from __future__ import annotations

import pytest

from scripts.research_planner import (
    compute_family_roi,
    adjust_priorities,
    allocate_budget,
    generate_plan,
    create_research_plan,
    format_plan_report,
    STRATEGIES,
)


def _make_experiments(n, family="tuning", base_metric=0.80):
    return [{"experiment_id": f"exp-{i:03d}", "status": "kept",
             "family": family, "config": {"family": family},
             "metrics": {"accuracy": base_metric + 0.005 * i}} for i in range(n)]


# --- compute_family_roi ---

def test_roi_basic():
    exps = _make_experiments(10, "tuning", 0.80)
    roi = compute_family_roi(exps, "accuracy")
    assert "tuning" in roi
    assert roi["tuning"]["n_experiments"] == 10
    assert roi["tuning"]["roi_per_experiment"] > 0

def test_roi_multiple_families():
    exps = _make_experiments(5, "tuning") + _make_experiments(5, "feature")
    roi = compute_family_roi(exps, "accuracy")
    assert "tuning" in roi
    assert "feature" in roi

def test_roi_exhausted():
    # Last 3 experiments with nearly identical metrics → exhausted
    exps = [{"family": "tuning", "metrics": {"accuracy": 0.85}, "status": "kept"}] * 5
    roi = compute_family_roi(exps, "accuracy")
    assert roi["tuning"]["exhausted"]

def test_roi_empty():
    assert compute_family_roi([], "accuracy") == {}

def test_roi_no_metrics():
    exps = [{"family": "tuning", "config": {}, "status": "kept"}] * 3
    roi = compute_family_roi(exps, "accuracy")
    assert roi["tuning"]["n_experiments"] == 3


# --- adjust_priorities ---

def test_priorities_default():
    roi = compute_family_roi(_make_experiments(10), "accuracy")
    priorities = adjust_priorities(STRATEGIES, roi, _make_experiments(10), "accuracy")
    assert abs(sum(priorities.values()) - 1.0) < 0.01

def test_priorities_production_goal():
    roi = compute_family_roi(_make_experiments(10), "accuracy")
    priorities = adjust_priorities(STRATEGIES, roi, _make_experiments(10), "accuracy", goal="deploy to production")
    assert priorities["calibration"] > STRATEGIES["calibration"]["base_priority"]

def test_priorities_many_experiments():
    exps = _make_experiments(25)
    roi = compute_family_roi(exps, "accuracy")
    priorities = adjust_priorities(STRATEGIES, roi, exps, "accuracy")
    # Verification should be boosted
    assert priorities["verification"] >= STRATEGIES["verification"]["base_priority"]


# --- allocate_budget ---

def test_allocate_basic():
    priorities = {"a": 0.5, "b": 0.3, "c": 0.2}
    allocation = allocate_budget(priorities, 20)
    assert sum(allocation.values()) == 20

def test_allocate_small_budget():
    priorities = {"a": 0.5, "b": 0.3, "c": 0.2}
    allocation = allocate_budget(priorities, 3)
    assert sum(allocation.values()) == 3

def test_allocate_zero():
    allocation = allocate_budget({"a": 0.5, "b": 0.5}, 0)
    assert all(v == 0 for v in allocation.values())

def test_allocate_single_strategy():
    allocation = allocate_budget({"a": 1.0}, 10)
    assert allocation["a"] == 10


# --- generate_plan ---

def test_plan_basic():
    allocation = {"feature_engineering": 5, "model_search": 3, "verification": 2}
    plan = generate_plan(allocation, STRATEGIES, {}, current_best=0.85)
    assert plan["total_experiments"] == 10
    assert len(plan["phases"]) == 3
    assert plan["expected_metric"] is not None

def test_plan_no_current_best():
    allocation = {"model_search": 5}
    plan = generate_plan(allocation, STRATEGIES, {})
    assert plan["expected_metric"] is None
    assert plan["total_experiments"] == 5

def test_plan_empty_allocation():
    plan = generate_plan({}, STRATEGIES, {})
    assert plan["total_experiments"] == 0
    assert len(plan["phases"]) == 0

def test_plan_experiment_numbering():
    allocation = {"feature_engineering": 3, "ensemble": 2}
    plan = generate_plan(allocation, STRATEGIES, {})
    all_numbers = [e["number"] for p in plan["phases"] for e in p["experiments"]]
    assert all_numbers == [1, 2, 3, 4, 5]


# --- create_research_plan ---

def test_plan_no_history(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text("")
    result = create_research_plan(budget=10, config_path=str(config), log_path=str(log))
    assert "message" in result

def test_plan_with_history(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    lines = [f'{{"experiment_id": "exp-{i:03d}", "status": "kept", "family": "tuning", "config": {{"family": "tuning"}}, "metrics": {{"accuracy": {0.80 + 0.005 * i}}}}}\n' for i in range(10)]
    log.write_text("".join(lines))
    result = create_research_plan(budget=20, config_path=str(config), log_path=str(log))
    assert "plan" in result
    assert result["plan"]["total_experiments"] == 20
    assert "priorities" in result

def test_plan_with_goal(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    lines = [f'{{"experiment_id": "exp-{i}", "status": "kept", "family": "tuning", "config": {{}}, "metrics": {{"accuracy": 0.85}}}}\n' for i in range(10)]
    log.write_text("".join(lines))
    result = create_research_plan(budget=15, goal="maximize F1 for production", config_path=str(config), log_path=str(log))
    assert result["goal"] == "maximize F1 for production"


# --- format_plan_report ---

def test_format_plan():
    report = {
        "budget": 20,
        "goal": "maximize F1",
        "current_best": 0.891,
        "primary_metric": "accuracy",
        "plan": {
            "phases": [
                {"label": "Feature Engineering", "n_experiments": 8, "budget_pct": 40,
                 "rationale": "High ROI", "experiments": [{"number": 1, "description": "Test"}],
                 "expected_gain": 0.04},
            ],
            "total_experiments": 8,
            "expected_metric": 0.931,
            "expected_gain": 0.04,
        },
        "generated_at": "2026-04-01",
    }
    text = format_plan_report(report)
    assert "Research Plan" in text
    assert "Feature Engineering" in text
    assert "0.891" in text

def test_format_no_history():
    report = {"message": "No experiments", "budget": 10}
    text = format_plan_report(report)
    assert "No experiments" in text
