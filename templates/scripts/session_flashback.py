#!/usr/bin/env python3
"""Session flashback — context restoration after time away.

Reads recent experiments, current best, pending hypotheses, annotations,
and budget status to produce a compact "where was I?" summary. Designed
for the moment you return to a project after hours or days away.

Usage:
    python scripts/session_flashback.py [--config config.yaml] [--log experiments/log.jsonl]
    python scripts/session_flashback.py --last 10         # Last 10 experiments
    python scripts/session_flashback.py --days 3          # Last 3 days
    python scripts/session_flashback.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments, load_hypotheses

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_LAST_N = 10
DEFAULT_DAYS = 7


# --- Data Loading ---


def load_annotations(path: str = "experiments/annotations.yaml") -> list[dict]:
    """Load experiment annotations from YAML."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    try:
        with open(p) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, list) else []
    except (yaml.YAMLError, OSError):
        return []


def load_budget_status(
    state_path: str = "experiment_state.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict | None:
    """Load budget status if available."""
    try:
        from scripts.budget_manager import get_budget_status
        result = get_budget_status(state_path, log_path)
        if "error" not in result:
            return result
    except (ImportError, Exception):
        pass
    return None


def load_experiment_state(path: str = "experiment_state.yaml") -> dict:
    """Load experiment state file."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        return {}


# --- Filtering ---


def filter_recent_experiments(
    experiments: list[dict],
    last_n: int | None = None,
    days: int | None = None,
) -> list[dict]:
    """Filter experiments to recent ones by count or time window.

    If both are given, uses whichever returns more experiments.
    """
    if not experiments:
        return []

    by_count = []
    by_time = []

    if last_n is not None:
        by_count = experiments[-last_n:]

    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff.isoformat()
        by_time = [
            e for e in experiments
            if e.get("timestamp", "") >= cutoff_str
        ]

    if by_count and by_time:
        return by_count if len(by_count) >= len(by_time) else by_time
    return by_count or by_time or experiments[-DEFAULT_LAST_N:]


def find_current_best(
    experiments: list[dict],
    metric: str,
    lower_is_better: bool,
) -> dict | None:
    """Find the current best kept experiment."""
    best = None
    best_val = float("inf") if lower_is_better else float("-inf")
    for exp in experiments:
        if exp.get("status") != "kept":
            continue
        val = exp.get("metrics", {}).get(metric)
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        if (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val = val
            best = exp
    return best


# --- Flashback Assembly ---


def assemble_flashback(
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    hypotheses_path: str = "hypotheses.yaml",
    annotations_path: str = "experiments/annotations.yaml",
    last_n: int | None = None,
    days: int | None = None,
) -> dict:
    """Assemble all context for a session flashback.

    Returns a structured dict with everything needed to resume work.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    all_experiments = load_experiments(log_path)
    if not all_experiments:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "No experiments found",
            "log_path": log_path,
        }

    recent = filter_recent_experiments(all_experiments, last_n, days)
    best = find_current_best(all_experiments, metric, lower_is_better)
    hypotheses = load_hypotheses(hypotheses_path)
    annotations = load_annotations(annotations_path)
    budget = load_budget_status(log_path=log_path)
    state = load_experiment_state()

    # Summarize recent activity
    recent_kept = sum(1 for e in recent if e.get("status") == "kept")
    recent_discarded = sum(1 for e in recent if e.get("status") == "discarded")

    # Time since last experiment
    last_ts = all_experiments[-1].get("timestamp", "")
    time_away = None
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            delta = datetime.now(timezone.utc) - last_dt
            time_away = {
                "hours": round(delta.total_seconds() / 3600, 1),
                "human": _format_timedelta(delta),
            }
        except (ValueError, TypeError):
            pass

    # Pending hypotheses
    pending = [h for h in hypotheses if h.get("status") == "queued"]
    high_priority = [h for h in pending if h.get("priority") == "high"]

    # Recent annotations
    recent_ids = {e.get("experiment_id") for e in recent}
    relevant_annotations = [
        a for a in annotations
        if a.get("experiment_id") in recent_ids
    ]

    # Current research direction from state
    current_direction = state.get("current_direction")
    iteration = state.get("iteration", 0)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
        "lower_is_better": lower_is_better,
        "total_experiments": len(all_experiments),
        "time_away": time_away,
        "current_best": {
            "experiment_id": best.get("experiment_id", "?"),
            "model_type": best.get("config", {}).get("model_type", "?"),
            "metrics": best.get("metrics", {}),
            "description": best.get("description", ""),
        } if best else None,
        "recent": {
            "count": len(recent),
            "kept": recent_kept,
            "discarded": recent_discarded,
            "experiments": [
                {
                    "experiment_id": e.get("experiment_id", "?"),
                    "status": e.get("status", "?"),
                    "metric_value": e.get("metrics", {}).get(metric),
                    "description": e.get("description", ""),
                    "timestamp": e.get("timestamp", ""),
                }
                for e in recent
            ],
        },
        "hypotheses": {
            "total_pending": len(pending),
            "high_priority": len(high_priority),
            "queue": [
                {
                    "id": h.get("id", "?"),
                    "description": h.get("description", ""),
                    "priority": h.get("priority", "normal"),
                    "source": h.get("source", ""),
                }
                for h in pending[:10]
            ],
        },
        "annotations": relevant_annotations,
        "budget": _summarize_budget(budget) if budget else None,
        "iteration": iteration,
        "current_direction": current_direction,
    }


def _summarize_budget(budget: dict) -> dict:
    """Extract compact budget summary."""
    usage = budget.get("usage", {})
    return {
        "phase": budget.get("phase", "?"),
        "fraction_used": usage.get("budget_fraction", 0),
        "experiments_remaining": usage.get("experiments_remaining"),
        "hours_remaining": usage.get("hours_remaining"),
        "exhausted": budget.get("exhausted", False),
    }


def _format_timedelta(delta: timedelta) -> str:
    """Format a timedelta as human-readable string."""
    total_hours = delta.total_seconds() / 3600
    if total_hours < 1:
        return f"{int(delta.total_seconds() / 60)} minutes"
    elif total_hours < 24:
        return f"{total_hours:.1f} hours"
    else:
        days = delta.days
        hours = (delta.seconds // 3600)
        return f"{days}d {hours}h"


# --- Report ---


def save_flashback_report(report: dict, output_dir: str = "experiments/flashbacks") -> Path:
    """Save flashback report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    filepath = out_path / f"flashback-{date_str}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_flashback_report(report: dict) -> str:
    """Format flashback as a compact markdown summary."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    lines = [
        "# Session Flashback",
        "",
        f"*Generated {report.get('timestamp', '?')[:19]} UTC*",
        "",
    ]

    # Time away
    time_away = report.get("time_away")
    if time_away:
        lines.append(f"You've been away **{time_away['human']}**.")
        lines.append("")

    # Current best
    best = report.get("current_best")
    if best:
        metric = report.get("metric", "?")
        direction = "lower" if report.get("lower_is_better") else "higher"
        metric_val = best.get("metrics", {}).get(metric, "?")
        lines.extend([
            "## Current Best",
            "",
            f"**{best['experiment_id']}** ({best['model_type']})",
            f"- {metric} = **{metric_val}** ({direction} is better)",
        ])
        other_metrics = {k: v for k, v in best.get("metrics", {}).items() if k != metric}
        if other_metrics:
            parts = ", ".join(f"{k}={v}" for k, v in other_metrics.items())
            lines.append(f"- Other: {parts}")
        if best.get("description"):
            lines.append(f"- *{best['description']}*")
        lines.append("")
    else:
        lines.extend(["## Current Best", "", "No kept experiments yet.", ""])

    # Recent activity
    recent = report.get("recent", {})
    if recent.get("count", 0) > 0:
        lines.extend([
            "## Recent Activity",
            "",
            f"**{recent['count']}** recent experiments: "
            f"{recent['kept']} kept, {recent['discarded']} discarded",
            "",
        ])
        for exp in recent.get("experiments", [])[-5:]:
            status_icon = "+" if exp["status"] == "kept" else "-"
            val = exp.get("metric_value", "?")
            desc = f" — {exp['description']}" if exp.get("description") else ""
            lines.append(f"  {status_icon} **{exp['experiment_id']}**: {val}{desc}")
        if recent["count"] > 5:
            lines.append(f"  *...and {recent['count'] - 5} more*")
        lines.append("")

    # Annotations
    annotations = report.get("annotations", [])
    if annotations:
        lines.extend(["## Notes & Annotations", ""])
        for ann in annotations[:5]:
            exp_id = ann.get("experiment_id", "?")
            note = ann.get("note", ann.get("text", ""))
            lines.append(f"- **{exp_id}**: {note}")
        lines.append("")

    # Hypothesis queue
    hyp = report.get("hypotheses", {})
    if hyp.get("total_pending", 0) > 0:
        lines.extend([
            "## Pending Hypotheses",
            "",
            f"**{hyp['total_pending']}** queued"
            + (f" ({hyp['high_priority']} high priority)" if hyp.get("high_priority") else ""),
            "",
        ])
        for h in hyp.get("queue", [])[:5]:
            priority = " **(HIGH)**" if h.get("priority") == "high" else ""
            source = f" [{h['source']}]" if h.get("source") else ""
            lines.append(f"- {h['id']}: {h['description']}{priority}{source}")
        if hyp["total_pending"] > 5:
            lines.append(f"  *...and {hyp['total_pending'] - 5} more*")
        lines.append("")
    else:
        lines.extend([
            "## Pending Hypotheses",
            "",
            "Queue is empty. Use `/turing:try` to inject ideas.",
            "",
        ])

    # Budget
    budget = report.get("budget")
    if budget:
        lines.extend(["## Budget", ""])
        if budget.get("exhausted"):
            lines.append("**EXHAUSTED** — no more experiments will run.")
        else:
            pct = budget.get("fraction_used", 0)
            lines.append(f"**{pct:.0%} used** (phase: {budget.get('phase', '?')})")
            if budget.get("experiments_remaining") is not None:
                lines.append(f"- {budget['experiments_remaining']} experiments remaining")
            if budget.get("hours_remaining") is not None:
                lines.append(f"- {budget['hours_remaining']:.1f}h remaining")
        lines.append("")

    # Research state
    direction = report.get("current_direction")
    iteration = report.get("iteration", 0)
    if direction or iteration:
        lines.extend(["## Research State", ""])
        if iteration:
            lines.append(f"- Iteration: {iteration}")
        if direction:
            lines.append(f"- Direction: {direction}")
        lines.append("")

    # Suggested next actions
    lines.extend([
        "---",
        "",
        "**Next steps:**",
    ])
    if hyp.get("high_priority"):
        lines.append("- Run `/turing:train` to execute high-priority hypotheses")
    elif hyp.get("total_pending", 0) > 0:
        lines.append("- Run `/turing:train` to execute queued hypotheses")
    else:
        lines.append("- Run `/turing:try` to inject a new hypothesis")
    lines.append("- Run `/turing:brief` for full research intelligence report")
    lines.append("- Run `/turing:trend` for long-term trend analysis")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Session flashback — where was I?")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--hypotheses", default="hypotheses.yaml", help="Path to hypotheses file")
    parser.add_argument("--annotations", default="experiments/annotations.yaml",
                        help="Path to annotations file")
    parser.add_argument("--last", type=int, default=None, help="Show last N experiments")
    parser.add_argument("--days", type=int, default=None, help="Show experiments from last N days")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    last_n = args.last if args.last is not None else (None if args.days is not None else DEFAULT_LAST_N)
    days = args.days

    report = assemble_flashback(
        config_path=args.config,
        log_path=args.log,
        hypotheses_path=args.hypotheses,
        annotations_path=args.annotations,
        last_n=last_n,
        days=days,
    )

    if "error" not in report:
        filepath = save_flashback_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_flashback_report(report))


if __name__ == "__main__":
    main()
