#!/usr/bin/env python3
"""Structured experiment state manager for the autoresearch pipeline.

Replaces free-text MEMORY.md with a validated experiment_state.yaml
that the agent reads and writes in a schema-enforced format. The state
is machine-readable and programmatically queryable.

MEMORY.md is kept as a human-readable companion, generated from the
structured state.

Usage:
    python scripts/update_state.py init                    # Create empty state
    python scripts/update_state.py show                    # Display current state
    python scripts/update_state.py set-best <exp-id> <metrics_json>
    python scripts/update_state.py add-observation <text> [--exp <exp-id>]
    python scripts/update_state.py add-failure <text> --exp <exp-id> --reason <text>
    python scripts/update_state.py add-direction <text> [--priority high|medium|low]
    python scripts/update_state.py log-session <n_experiments> <best_metric>
    python scripts/update_state.py generate-memory         # Write MEMORY.md from state
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

DEFAULT_STATE_PATH = "experiment_state.yaml"
DEFAULT_MEMORY_PATH = ".claude/agent-memory/ml-researcher/MEMORY.md"

EMPTY_STATE = {
    "goal": "",
    "primary_metric": "",
    "metric_direction": "",
    "best_result": None,
    "observations": [],
    "failed_approaches": [],
    "promising_directions": [],
    "session_history": [],
}

REQUIRED_KEYS = set(EMPTY_STATE.keys())


def load_state(path: str) -> dict:
    """Load experiment state from YAML, validating required keys."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return copy.deepcopy(EMPTY_STATE)
    with open(p) as f:
        state = yaml.safe_load(f)
    if not isinstance(state, dict):
        return copy.deepcopy(EMPTY_STATE)
    # Ensure all required keys exist
    for key in REQUIRED_KEYS:
        if key not in state:
            state[key] = copy.deepcopy(EMPTY_STATE[key])
    return state


def save_state(path: str, state: dict) -> None:
    """Save experiment state to YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def validate_state(state: dict) -> list[str]:
    """Validate state against schema. Returns list of error messages."""
    errors = []
    for key in REQUIRED_KEYS:
        if key not in state:
            errors.append(f"Missing required key: {key}")
    if state.get("best_result") is not None:
        br = state["best_result"]
        if not isinstance(br, dict):
            errors.append("best_result must be a dict or null")
        elif "experiment_id" not in br or "metrics" not in br:
            errors.append("best_result must have experiment_id and metrics")
    if not isinstance(state.get("observations", []), list):
        errors.append("observations must be a list")
    if not isinstance(state.get("failed_approaches", []), list):
        errors.append("failed_approaches must be a list")
    if not isinstance(state.get("promising_directions", []), list):
        errors.append("promising_directions must be a list")
    if not isinstance(state.get("session_history", []), list):
        errors.append("session_history must be a list")
    return errors


def set_best(state: dict, experiment_id: str, metrics: dict) -> dict:
    """Update the best result in state."""
    state["best_result"] = {
        "experiment_id": experiment_id,
        "metrics": metrics,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Check for seed study data
    from scripts.turing_io import load_seed_study
    seed_study = load_seed_study(experiment_id)
    if seed_study and "mean" in seed_study:
        state["best_result"]["seed_study"] = {
            "mean": seed_study["mean"],
            "std": seed_study.get("std", 0),
            "cv_percent": seed_study.get("cv_percent", 0),
            "seed_sensitive": seed_study.get("seed_sensitive", False),
            "seeds_tested": len(seed_study.get("seeds_run", [])),
        }

    return state


def add_observation(state: dict, text: str, experiment_id: str | None = None) -> dict:
    """Add an observation to state."""
    entry = {
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if experiment_id:
        entry["experiment_id"] = experiment_id
    state["observations"].append(entry)
    return state


def add_failure(state: dict, description: str, experiment_id: str, reason: str) -> dict:
    """Record a failed approach."""
    state["failed_approaches"].append({
        "description": description,
        "experiment_id": experiment_id,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return state


def add_direction(state: dict, description: str, priority: str = "medium") -> dict:
    """Add a promising direction."""
    state["promising_directions"].append({
        "description": description,
        "priority": priority,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return state


def log_session(state: dict, n_experiments: int, best_metric: float | None) -> dict:
    """Log a training session summary."""
    state["session_history"].append({
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "experiments_run": n_experiments,
        "best_metric": best_metric,
    })
    return state


def generate_memory(state: dict, goal: str = "", metric: str = "", direction: str = "") -> str:
    """Generate MEMORY.md content from structured state."""
    goal = goal or state.get("goal", "")
    metric = metric or state.get("primary_metric", "")
    direction = direction or state.get("metric_direction", "")

    lines = [
        "# ML Researcher Memory",
        "",
        "## Goal",
        "",
        goal or "(not set)",
        "",
        f"Primary metric: {metric} ({direction} is better).",
        "",
        "## Best Result",
        "",
    ]

    br = state.get("best_result")
    if br:
        metrics_str = ", ".join(f"{k}={v}" for k, v in br.get("metrics", {}).items())
        lines.append(f"Experiment {br.get('experiment_id', '?')}: {metrics_str}")
    else:
        lines.append("No experiments completed yet.")

    lines.extend(["", "## Observations", ""])
    for obs in state.get("observations", [])[-10:]:  # Last 10
        exp_tag = f" ({obs['experiment_id']})" if obs.get("experiment_id") else ""
        lines.append(f"- {obs['text']}{exp_tag}")
    if not state.get("observations"):
        lines.append("(none yet)")

    lines.extend(["", "## Failed Approaches", ""])
    for fail in state.get("failed_approaches", []):
        lines.append(f"- {fail['description']} ({fail['experiment_id']}): {fail['reason']}")
    if not state.get("failed_approaches"):
        lines.append("(none yet)")

    lines.extend(["", "## Promising Directions", ""])
    for d in state.get("promising_directions", []):
        priority_tag = f" [{d['priority']}]" if d.get("priority") != "medium" else ""
        lines.append(f"- {d['description']}{priority_tag}")
    if not state.get("promising_directions"):
        lines.append("(none yet)")

    lines.extend(["", "## Session History", ""])
    if state.get("session_history"):
        lines.append("| Session | Experiments | Best Metric | Notes |")
        lines.append("|---------|-------------|-------------|-------|")
        for s in state["session_history"]:
            bm = f"{s['best_metric']:.4f}" if s.get("best_metric") is not None else "N/A"
            lines.append(f"| {s.get('date', '?')} | {s.get('experiments_run', 0)} | {bm} | |")
    else:
        lines.append("(no sessions yet)")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage experiment state")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Create empty state")
    subparsers.add_parser("show", help="Display current state")
    subparsers.add_parser("validate", help="Validate state schema")

    sb = subparsers.add_parser("set-best")
    sb.add_argument("experiment_id")
    sb.add_argument("metrics_json")

    obs = subparsers.add_parser("add-observation")
    obs.add_argument("text")
    obs.add_argument("--exp", default=None)

    fail = subparsers.add_parser("add-failure")
    fail.add_argument("text")
    fail.add_argument("--exp", required=True)
    fail.add_argument("--reason", required=True)

    d = subparsers.add_parser("add-direction")
    d.add_argument("text")
    d.add_argument("--priority", default="medium", choices=["high", "medium", "low"])

    sess = subparsers.add_parser("log-session")
    sess.add_argument("n_experiments", type=int)
    sess.add_argument("best_metric", type=float, nargs="?", default=None)

    subparsers.add_parser("generate-memory", help="Write MEMORY.md from state")

    args = parser.parse_args()
    state = load_state(args.state)

    if args.command == "init":
        save_state(args.state, EMPTY_STATE)
        print(f"Initialized empty state at {args.state}")

    elif args.command == "show":
        print(yaml.dump(state, default_flow_style=False, sort_keys=False))

    elif args.command == "validate":
        errors = validate_state(state)
        if errors:
            for e in errors:
                print(f"  ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print("State is valid.")

    elif args.command == "set-best":
        metrics = json.loads(args.metrics_json)
        set_best(state, args.experiment_id, metrics)
        save_state(args.state, state)
        print(f"Best result set to {args.experiment_id}")

    elif args.command == "add-observation":
        add_observation(state, args.text, args.exp)
        save_state(args.state, state)
        print("Observation added.")

    elif args.command == "add-failure":
        add_failure(state, args.text, args.exp, args.reason)
        save_state(args.state, state)
        print("Failed approach recorded.")

    elif args.command == "add-direction":
        add_direction(state, args.text, args.priority)
        save_state(args.state, state)
        print("Promising direction added.")

    elif args.command == "log-session":
        log_session(state, args.n_experiments, args.best_metric)
        save_state(args.state, state)
        print("Session logged.")

    elif args.command == "generate-memory":
        memory_content = generate_memory(state)
        print(memory_content)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
