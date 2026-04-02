#!/usr/bin/env python3
"""Research planning assistant for the autoresearch pipeline.

Given the current project state, generates a strategic research plan
that allocates experiments across strategies by expected ROI. Operates
one level above individual hypotheses — designs campaigns.

Usage:
    python scripts/research_planner.py --budget 20
    python scripts/research_planner.py --budget 20 --goal "maximize F1"
    python scripts/research_planner.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_BUDGET = 20

# Strategy definitions with base ROI and experiment templates
STRATEGIES = {
    "feature_engineering": {
        "label": "Feature Engineering",
        "base_priority": 0.4,
        "typical_gain": 0.005,
        "templates": [
            "Automated feature selection (top consensus features)",
            "Interaction feature generation",
            "Domain-specific feature engineering",
            "Feature ablation to prune dead weight",
        ],
    },
    "model_search": {
        "label": "Model Architecture Search",
        "base_priority": 0.25,
        "typical_gain": 0.003,
        "templates": [
            "Try alternative model family",
            "Hyperparameter optimization",
            "Architecture modification",
        ],
    },
    "ensemble": {
        "label": "Ensemble & Composition",
        "base_priority": 0.15,
        "typical_gain": 0.008,
        "templates": [
            "Build stacking ensemble from top diverse models",
            "Model soup from top checkpoints",
            "Pipeline stitch: swap preprocessing into ensemble",
        ],
    },
    "calibration": {
        "label": "Production Readiness",
        "base_priority": 0.1,
        "typical_gain": 0.001,
        "templates": [
            "Probability calibration (Platt/isotonic)",
            "Post-training quantization (INT8)",
            "Weight pruning (find sparsity knee point)",
            "Full seed study on final model",
        ],
    },
    "verification": {
        "label": "Verification & Documentation",
        "base_priority": 0.1,
        "typical_gain": 0.0,
        "templates": [
            "Reproduce final model",
            "Run methodology audit",
            "Generate model card",
        ],
    },
}


# --- ROI Analysis ---


def compute_family_roi(
    experiments: list[dict],
    primary_metric: str,
    lower_is_better: bool = False,
) -> dict[str, dict]:
    """Compute ROI (improvement per experiment) for each experiment family.

    Returns:
        Dict of {family: {experiments, total_improvement, roi, exhausted}}.
    """
    families = {}

    for exp in experiments:
        family = exp.get("family", exp.get("config", {}).get("family", "unknown"))
        if family not in families:
            families[family] = {"experiments": [], "metrics": []}
        families[family]["experiments"].append(exp)

        val = exp.get("metrics", {}).get(primary_metric)
        if val is not None:
            families[family]["metrics"].append(float(val))

    result = {}
    for family, data in families.items():
        metrics = data["metrics"]
        n_exps = len(data["experiments"])

        if len(metrics) < 2:
            roi = 0
            exhausted = False
        else:
            if lower_is_better:
                improvement = metrics[0] - min(metrics)
            else:
                improvement = max(metrics) - metrics[0]
            roi = improvement / n_exps if n_exps > 0 else 0

            # Check if last 3 experiments showed no improvement
            recent = metrics[-3:]
            exhausted = len(recent) >= 3 and (max(recent) - min(recent)) < 0.002

        result[family] = {
            "n_experiments": n_exps,
            "total_improvement": round(float(max(metrics) - min(metrics)) if metrics else 0, 6),
            "roi_per_experiment": round(float(roi), 6),
            "exhausted": exhausted,
            "best_metric": round(float(max(metrics)), 6) if metrics else None,
        }

    return result


def adjust_priorities(
    base_strategies: dict,
    family_roi: dict[str, dict],
    experiments: list[dict],
    primary_metric: str,
    goal: str | None = None,
) -> dict[str, float]:
    """Adjust strategy priorities based on project state.

    Returns:
        Dict of {strategy_name: adjusted_priority}.
    """
    priorities = {name: s["base_priority"] for name, s in base_strategies.items()}

    n_total = len(experiments)

    # Boost feature engineering if it has high ROI
    fe_families = [f for f, data in family_roi.items()
                   if "feature" in f.lower() and not data["exhausted"]]
    if fe_families:
        priorities["feature_engineering"] *= 1.3

    # Reduce model search if exhausted
    model_families = [f for f, data in family_roi.items()
                      if ("tuning" in f.lower() or "architecture" in f.lower()) and data["exhausted"]]
    if model_families:
        priorities["model_search"] *= 0.5

    # Boost ensemble if enough diverse models exist
    n_kept = sum(1 for e in experiments if e.get("status") == "kept")
    if n_kept >= 5:
        priorities["ensemble"] *= 1.2

    # Boost verification if many experiments done
    if n_total >= 20:
        priorities["verification"] *= 1.5

    # Goal-based adjustments
    if goal:
        goal_lower = goal.lower()
        if "production" in goal_lower or "deploy" in goal_lower:
            priorities["calibration"] *= 2.0
            priorities["verification"] *= 1.5
        if "f1" in goal_lower or "accuracy" in goal_lower:
            priorities["feature_engineering"] *= 1.2
            priorities["ensemble"] *= 1.2

    # Normalize
    total = sum(priorities.values())
    if total > 0:
        priorities = {k: round(v / total, 3) for k, v in priorities.items()}

    return priorities


# --- Plan Generation ---


def allocate_budget(
    priorities: dict[str, float],
    budget: int,
    min_per_strategy: int = 1,
) -> dict[str, int]:
    """Allocate experiment budget across strategies.

    Args:
        priorities: Strategy priorities (sum to ~1.0).
        budget: Total experiment budget.
        min_per_strategy: Minimum experiments per active strategy.

    Returns:
        Dict of {strategy: n_experiments}.
    """
    if budget <= 0:
        return {k: 0 for k in priorities}

    # Initial allocation by priority
    allocation = {}
    remaining = budget

    for strategy, priority in sorted(priorities.items(), key=lambda x: -x[1]):
        n = max(min_per_strategy, round(budget * priority))
        n = min(n, remaining)
        allocation[strategy] = n
        remaining -= n
        if remaining <= 0:
            break

    # Distribute any remaining
    if remaining > 0:
        top_strategy = max(priorities, key=priorities.get)
        allocation[top_strategy] = allocation.get(top_strategy, 0) + remaining

    return allocation


def generate_plan(
    allocation: dict[str, int],
    strategies: dict,
    family_roi: dict[str, dict],
    current_best: float | None = None,
    primary_metric: str = "accuracy",
) -> dict:
    """Generate a structured research plan from budget allocation.

    Returns:
        Plan with phases, experiment descriptions, and expected outcome.
    """
    phases = []
    exp_counter = 1

    for strategy_name, n_exps in allocation.items():
        if n_exps <= 0:
            continue

        strategy = strategies.get(strategy_name, {})
        templates = strategy.get("templates", [])
        typical_gain = strategy.get("typical_gain", 0)

        experiments = []
        for i in range(n_exps):
            template = templates[i % len(templates)] if templates else f"Experiment {exp_counter}"
            experiments.append({
                "number": exp_counter,
                "description": template,
            })
            exp_counter += 1

        pct = round(n_exps / sum(allocation.values()) * 100) if sum(allocation.values()) > 0 else 0

        phases.append({
            "name": strategy_name,
            "label": strategy.get("label", strategy_name),
            "n_experiments": n_exps,
            "budget_pct": pct,
            "rationale": _phase_rationale(strategy_name, family_roi),
            "experiments": experiments,
            "expected_gain": round(typical_gain * n_exps, 4),
        })

    # Estimate expected outcome
    total_expected_gain = sum(p["expected_gain"] for p in phases)
    expected_metric = round(current_best + total_expected_gain, 4) if current_best else None

    return {
        "phases": phases,
        "total_experiments": exp_counter - 1,
        "expected_metric": expected_metric,
        "expected_gain": round(total_expected_gain, 4),
        "primary_metric": primary_metric,
    }


def _phase_rationale(strategy_name: str, family_roi: dict) -> str:
    """Generate rationale for a phase allocation."""
    rationales = {
        "feature_engineering": "Highest ROI direction — feature improvements compound across models",
        "model_search": "Explore alternative architectures for potential step-change improvement",
        "ensemble": "Combine existing models for 1-3% improvement at zero additional training cost",
        "calibration": "Required for production deployment — probability calibration and model compression",
        "verification": "Final validation — reproduce results, audit methodology, generate model card",
    }
    return rationales.get(strategy_name, "Strategic allocation")


# --- Full Pipeline ---


def create_research_plan(
    budget: int = DEFAULT_BUDGET,
    goal: str | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Create a strategic research plan.

    Args:
        budget: Total experiment budget.
        goal: Optional goal description.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.

    Returns:
        Research plan with phases, allocation, and expected outcome.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)

    if not experiments:
        return {
            "budget": budget,
            "goal": goal,
            "message": "No experiment history — start with /turing:train first",
            "plan": generate_plan(
                {"model_search": budget},
                STRATEGIES, {},
                primary_metric=primary_metric,
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Compute family ROI
    family_roi = compute_family_roi(experiments, primary_metric, lower_is_better)

    # Get current best
    best_metrics = [exp.get("metrics", {}).get(primary_metric)
                    for exp in experiments if exp.get("metrics", {}).get(primary_metric) is not None]
    current_best = max(best_metrics) if best_metrics and not lower_is_better else (min(best_metrics) if best_metrics else None)

    # Adjust priorities
    priorities = adjust_priorities(STRATEGIES, family_roi, experiments, primary_metric, goal)

    # Allocate budget
    allocation = allocate_budget(priorities, budget)

    # Generate plan
    plan = generate_plan(allocation, STRATEGIES, family_roi, current_best, primary_metric)

    return {
        "budget": budget,
        "goal": goal,
        "current_best": current_best,
        "primary_metric": primary_metric,
        "n_experiments_so_far": len(experiments),
        "family_roi": family_roi,
        "priorities": priorities,
        "allocation": allocation,
        "plan": plan,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# --- Report Formatting ---


def save_plan_report(report: dict, output_dir: str = "experiments/plans") -> Path:
    """Save research plan to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filepath = out_path / f"plan-{ts}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_plan_report(report: dict) -> str:
    """Format research plan as readable markdown."""
    if "message" in report and "plan" not in report:
        return report["message"]

    plan = report.get("plan", {})
    budget = report.get("budget", 0)
    goal = report.get("goal", "maximize primary metric")

    lines = [
        f"# Research Plan ({budget} experiments, goal: {goal or 'maximize metric'})",
        "",
    ]

    if report.get("current_best"):
        lines.append(f"**Current best:** {report['primary_metric']}={report['current_best']}")
        lines.append("")

    phases = plan.get("phases", [])
    phase_label = "A"

    for phase in phases:
        lines.append(f"## Phase {phase_label}: {phase['label']} ({phase['n_experiments']} experiments, {phase['budget_pct']}% of budget)")
        lines.append(f"*Rationale: {phase['rationale']}*")
        lines.append("")

        for exp in phase.get("experiments", []):
            lines.append(f"  {exp['number']}. {exp['description']}")

        lines.append("")
        phase_label = chr(ord(phase_label) + 1)

    expected = plan.get("expected_metric")
    gain = plan.get("expected_gain", 0)
    if expected:
        lines.append(f"**Expected outcome:** {report.get('primary_metric', 'metric')} {report.get('current_best', '?')} → {expected} (+{gain})")
    else:
        lines.append(f"**Expected gain:** +{gain}")

    lines.extend(["", f"*Generated: {report.get('generated_at', 'N/A')}*"])
    return "\n".join(lines)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Research planning assistant — strategic experiment campaign design"
    )
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help="Total experiment budget")
    parser.add_argument("--goal", help="Goal description (e.g., 'maximize F1 for production')")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    report = create_research_plan(
        budget=args.budget,
        goal=args.goal,
        config_path=args.config,
        log_path=args.log,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_plan_report(report))

    if "error" not in report:
        saved = save_plan_report(report)
        if not args.json:
            print(f"\nSaved: {saved}")


if __name__ == "__main__":
    main()
