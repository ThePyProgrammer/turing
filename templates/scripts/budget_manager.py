#!/usr/bin/env python3
"""Compute budget manager for the autoresearch pipeline.

Sets a total compute budget (experiments, hours, or cost) and allocates
across exploration vs exploitation. Auto-shifts to exploit mode when
budget runs low. Prevents runaway compute spend.

Usage:
    python scripts/budget_manager.py set --experiments 50 --hours 8
    python scripts/budget_manager.py status
    python scripts/budget_manager.py reset
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_STATE_PATH = "experiment_state.yaml"

# Budget allocation policy thresholds
EXPLORE_PHASE_END = 0.50    # 0-50% budget: explore
MIXED_PHASE_END = 0.80      # 50-80%: mixed
# 80-100%: exploit only


# --- Budget State ---


def load_budget(state_path: str = DEFAULT_STATE_PATH) -> dict | None:
    """Load budget from experiment state file."""
    path = Path(state_path)
    if not path.exists():
        return None

    with open(path) as f:
        state = yaml.safe_load(f) or {}

    return state.get("budget")


def save_budget(budget: dict, state_path: str = DEFAULT_STATE_PATH) -> None:
    """Save budget to experiment state file."""
    path = Path(state_path)

    state = {}
    if path.exists():
        with open(path) as f:
            state = yaml.safe_load(f) or {}

    state["budget"] = budget

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False)


# --- Budget Operations ---


def set_budget(
    max_experiments: int | None = None,
    max_hours: float | None = None,
    state_path: str = DEFAULT_STATE_PATH,
) -> dict:
    """Set a compute budget.

    Args:
        max_experiments: Maximum number of experiments.
        max_hours: Maximum wall-clock hours.
        state_path: Path to experiment state file.

    Returns:
        Budget status dict.
    """
    if max_experiments is None and max_hours is None:
        return {"error": "Specify at least one constraint: --experiments or --hours"}

    budget = {
        "set_at": datetime.now(timezone.utc).isoformat(),
        "max_experiments": max_experiments,
        "max_hours": max_hours,
        "active": True,
    }

    save_budget(budget, state_path)

    return {
        "action": "set",
        "budget": budget,
        "message": _format_budget_constraints(budget),
    }


def get_budget_status(
    state_path: str = DEFAULT_STATE_PATH,
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Get current budget status with usage and projections.

    Args:
        state_path: Path to experiment state file.
        log_path: Path to experiment log.

    Returns:
        Complete budget status dict.
    """
    budget = load_budget(state_path)
    if not budget or not budget.get("active"):
        return {"error": "No active budget. Use `/turing:budget set` first."}

    experiments = load_experiments(log_path)

    # Count experiments since budget was set
    budget_set_at = budget.get("set_at", "")
    experiments_since = [
        e for e in experiments
        if e.get("timestamp", "") >= budget_set_at
    ]

    used_experiments = len(experiments_since)
    max_experiments = budget.get("max_experiments")

    # Compute time usage
    total_seconds = sum(
        e.get("metrics", {}).get("train_seconds", 0)
        for e in experiments_since
        if isinstance(e.get("metrics", {}).get("train_seconds"), (int, float))
    )
    used_hours = total_seconds / 3600
    max_hours = budget.get("max_hours")

    # Compute budget fraction used
    fractions = []
    if max_experiments and max_experiments > 0:
        fractions.append(used_experiments / max_experiments)
    if max_hours and max_hours > 0:
        fractions.append(used_hours / max_hours)

    budget_used = max(fractions) if fractions else 0.0

    # Determine current phase and recommended mode
    phase = determine_phase(budget_used)
    recommended_mode = phase_to_mode(phase)

    # Burn rate
    burn_rate = None
    if used_hours > 0 and used_experiments > 0:
        burn_rate = used_experiments / used_hours

    # Projection
    remaining_experiments = (max_experiments - used_experiments) if max_experiments else None
    remaining_hours = (max_hours - used_hours) if max_hours else None
    projected_exhaustion_hours = None
    if burn_rate and burn_rate > 0 and remaining_experiments:
        projected_exhaustion_hours = remaining_experiments / burn_rate

    # Check if exhausted
    exhausted = budget_used >= 1.0

    # Allocation breakdown
    explore_count = sum(1 for e in experiments_since if _is_explore(e))
    exploit_count = used_experiments - explore_count

    status = {
        "action": "status",
        "budget": budget,
        "usage": {
            "experiments_used": used_experiments,
            "experiments_max": max_experiments,
            "experiments_remaining": remaining_experiments,
            "hours_used": round(used_hours, 2),
            "hours_max": max_hours,
            "hours_remaining": round(remaining_hours, 2) if remaining_hours is not None else None,
            "budget_fraction": round(budget_used, 4),
        },
        "phase": phase,
        "recommended_mode": recommended_mode,
        "allocation": {
            "explore": explore_count,
            "exploit": exploit_count,
        },
        "burn_rate": round(burn_rate, 2) if burn_rate else None,
        "projected_exhaustion_hours": round(projected_exhaustion_hours, 2) if projected_exhaustion_hours else None,
        "exhausted": exhausted,
    }

    if exhausted:
        status["warning"] = "Budget exhausted. /turing:train will refuse to start new experiments."

    return status


def reset_budget(state_path: str = DEFAULT_STATE_PATH) -> dict:
    """Reset (deactivate) the current budget."""
    budget = load_budget(state_path)
    if not budget:
        return {"action": "reset", "message": "No budget to reset."}

    budget["active"] = False
    budget["reset_at"] = datetime.now(timezone.utc).isoformat()
    save_budget(budget, state_path)

    return {"action": "reset", "message": "Budget deactivated."}


def check_budget_allows(state_path: str = DEFAULT_STATE_PATH, log_path: str = DEFAULT_LOG_PATH) -> dict:
    """Check if the budget allows another experiment.

    Returns dict with allowed (bool) and reason.
    """
    budget = load_budget(state_path)
    if not budget or not budget.get("active"):
        return {"allowed": True, "reason": "No active budget"}

    status = get_budget_status(state_path, log_path)
    if "error" in status:
        return {"allowed": True, "reason": "Budget status unavailable"}

    if status.get("exhausted"):
        return {
            "allowed": False,
            "reason": f"Budget exhausted ({status['usage']['budget_fraction']:.0%} used)",
        }

    return {
        "allowed": True,
        "reason": f"Budget at {status['usage']['budget_fraction']:.0%}",
        "recommended_mode": status.get("recommended_mode"),
    }


# --- Phase Logic ---


def determine_phase(budget_fraction: float) -> str:
    """Determine budget phase from fraction used.

    Returns: 'explore', 'mixed', or 'exploit'.
    """
    if budget_fraction < EXPLORE_PHASE_END:
        return "explore"
    elif budget_fraction < MIXED_PHASE_END:
        return "mixed"
    else:
        return "exploit"


def phase_to_mode(phase: str) -> str:
    """Map budget phase to recommended research mode."""
    return {
        "explore": "explore",
        "mixed": "explore",  # Still explore promising, but start exploiting
        "exploit": "exploit",
    }.get(phase, "explore")


def _is_explore(experiment: dict) -> bool:
    """Heuristic: classify experiment as explore vs exploit."""
    config = experiment.get("config", {})
    # New model types or significantly different configs = explore
    # Similar to prior experiments = exploit
    # Simple heuristic: if experiment has a novel model_type, it's exploration
    return config.get("model_type", "") != config.get("base_model_type", config.get("model_type", ""))


def _format_budget_constraints(budget: dict) -> str:
    """Format budget constraints as human-readable string."""
    parts = []
    if budget.get("max_experiments"):
        parts.append(f"{budget['max_experiments']} experiments")
    if budget.get("max_hours"):
        parts.append(f"{budget['max_hours']} hours")
    return "Budget set: " + ", ".join(parts) if parts else "Budget set (no constraints)"


# --- Report Formatting ---


def format_budget_report(report: dict) -> str:
    """Format budget report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    action = report.get("action", "?")

    if action == "set":
        return f"# Budget Set\n\n{report.get('message', '')}"

    if action == "reset":
        return f"# Budget Reset\n\n{report.get('message', '')}"

    if action != "status":
        return f"Unknown action: {action}"

    usage = report.get("usage", {})
    phase = report.get("phase", "?")

    lines = [
        "# Budget Status",
        "",
    ]

    # Experiments
    if usage.get("experiments_max"):
        pct = usage["experiments_used"] / usage["experiments_max"] * 100
        lines.append(
            f"**Experiments:** {usage['experiments_used']}/{usage['experiments_max']} "
            f"used ({pct:.0f}%), {usage.get('experiments_remaining', 0)} remaining"
        )

    # Time
    if usage.get("hours_max"):
        pct = usage["hours_used"] / usage["hours_max"] * 100
        lines.append(
            f"**Time:** {usage['hours_used']:.1f}/{usage['hours_max']:.1f}h "
            f"used ({pct:.0f}%), {usage.get('hours_remaining', 0):.1f}h remaining"
        )

    # Burn rate
    burn = report.get("burn_rate")
    if burn:
        lines.append(f"**Burn rate:** {burn:.1f} experiments/hour")

    proj = report.get("projected_exhaustion_hours")
    if proj:
        lines.append(f"**Projected exhaustion:** ~{proj:.1f} hours")

    lines.append("")

    # Phase
    phase_labels = {
        "explore": "EXPLORE (try diverse hypotheses)",
        "mixed": "MIXED (explore promising, exploit best)",
        "exploit": "EXPLOIT ONLY (refine the winner)",
    }
    lines.append(f"**Phase:** {phase_labels.get(phase, phase)}")
    lines.append(f"**Recommended mode:** {report.get('recommended_mode', '?')}")

    # Allocation
    alloc = report.get("allocation", {})
    if alloc:
        lines.extend([
            "",
            "**Allocation:**",
            f"- Explore: {alloc.get('explore', 0)} experiments",
            f"- Exploit: {alloc.get('exploit', 0)} experiments",
        ])

    # Warning
    if report.get("warning"):
        lines.extend(["", f"**WARNING:** {report['warning']}"])

    # Auto-mode shift info
    if phase == "explore":
        shift_at = usage.get("experiments_max", 0) * MIXED_PHASE_END
        lines.extend(["", f"*Auto-shift to mixed mode at experiment {shift_at:.0f}*"])

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute budget manager",
    )
    parser.add_argument(
        "action", choices=["set", "status", "reset", "check"],
        help="Budget action",
    )
    parser.add_argument(
        "--experiments", type=int,
        help="Max experiments budget",
    )
    parser.add_argument(
        "--hours", type=float,
        help="Max hours budget",
    )
    parser.add_argument(
        "--state", default=DEFAULT_STATE_PATH,
        help=f"Path to experiment state (default: {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--log", default=DEFAULT_LOG_PATH,
        help="Path to experiment log",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    if args.action == "set":
        report = set_budget(args.experiments, args.hours, args.state)
    elif args.action == "status":
        report = get_budget_status(args.state, args.log)
    elif args.action == "reset":
        report = reset_budget(args.state)
    elif args.action == "check":
        report = check_budget_allows(args.state, args.log)
    else:
        report = {"error": f"Unknown action: {args.action}"}

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_budget_report(report))


if __name__ == "__main__":
    main()
