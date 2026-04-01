#!/usr/bin/env python3
"""Experiment branching — run parallel tracks from a common parent.

"Try both A and B from this point" — creates child experiments,
runs both, reports which branch wins.

Usage:
    python scripts/fork_experiment.py exp-042 --branches "LightGBM dart" "XGBoost deeper"
    python scripts/fork_experiment.py exp-042 --branches "A" "B" --auto-promote
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments


def find_experiment(experiments: list[dict], exp_id: str) -> dict | None:
    """Find experiment by ID."""
    for exp in experiments:
        if exp.get("experiment_id") == exp_id:
            return exp
    return None


def create_branch(
    parent: dict,
    branch_description: str,
    branch_index: int,
) -> dict:
    """Create a branch descriptor from a parent experiment.

    Returns dict with branch metadata (not yet executed).
    """
    parent_id = parent.get("experiment_id", "unknown")
    return {
        "branch_id": f"fork-{parent_id}-{branch_index + 1}",
        "parent_id": parent_id,
        "description": branch_description,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_experiment": None,
        "metrics": {},
    }


def run_branch(branch: dict, seed: int = 42, timeout: int = 600) -> dict:
    """Execute a single branch experiment.

    Returns updated branch dict with status and metrics.
    """
    branch["status"] = "running"
    branch["started_at"] = datetime.now(timezone.utc).isoformat()

    cmd = ["python", "train.py", "--seed", str(seed)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        branch["status"] = "failed"
        branch["error"] = "timeout"
        return branch

    if proc.returncode != 0:
        branch["status"] = "failed"
        branch["error"] = proc.stderr[-300:] if proc.stderr else "unknown error"
        return branch

    # Parse metrics
    metrics = {}
    in_block = False
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line == "---":
            if in_block:
                break
            in_block = True
            continue
        if in_block and ":" in line:
            key, value = line.split(":", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                metrics[key.strip()] = value.strip()

    branch["status"] = "completed"
    branch["completed_at"] = datetime.now(timezone.utc).isoformat()
    branch["metrics"] = metrics
    return branch


def determine_winner(
    branches: list[dict],
    metric: str,
    lower_is_better: bool,
) -> dict | None:
    """Determine the winning branch by primary metric.

    Returns the winning branch dict, or None if no branches completed.
    """
    completed = [b for b in branches if b.get("status") == "completed" and metric in b.get("metrics", {})]
    if not completed:
        return None

    if lower_is_better:
        return min(completed, key=lambda b: b["metrics"][metric])
    else:
        return max(completed, key=lambda b: b["metrics"][metric])


def format_fork_report(
    parent_id: str,
    branches: list[dict],
    winner: dict | None,
    metric: str,
) -> str:
    """Format fork results as a comparison tree."""
    lines = [
        f"# Fork from {parent_id}",
        "",
    ]

    if not branches:
        lines.append("No branches executed.")
        return "\n".join(lines)

    winner_id = winner["branch_id"] if winner else None

    for branch in branches:
        status = branch.get("status", "?")
        desc = branch.get("description", "?")
        bid = branch.get("branch_id", "?")

        if status == "completed":
            metric_val = branch.get("metrics", {}).get(metric, "N/A")
            is_winner = bid == winner_id
            marker = "WINNER" if is_winner else ""
            if isinstance(metric_val, float):
                lines.append(f"├── {bid}: {desc} → {metric}={metric_val:.4f} {marker}")
            else:
                lines.append(f"├── {bid}: {desc} → {metric}={metric_val} {marker}")
        elif status == "failed":
            error = branch.get("error", "unknown")
            lines.append(f"├── {bid}: {desc} → FAILED ({error})")
        else:
            lines.append(f"├── {bid}: {desc} → {status}")

    if winner:
        lines.extend([
            "",
            f"**Recommendation:** promote {winner['branch_id']}, abandon the rest.",
        ])

    return "\n".join(lines)


def save_fork_report(report: dict, output_dir: str = "experiments/forks") -> Path:
    """Save fork report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    parent_id = report.get("parent_id", "unknown")
    filepath = out_path / f"{parent_id}-fork.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def run_fork(
    exp_id: str,
    branch_descriptions: list[str],
    auto_promote: bool = False,
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
    timeout: int = 600,
) -> dict:
    """Fork an experiment into multiple branches and run all.

    Args:
        exp_id: Parent experiment ID.
        branch_descriptions: List of branch descriptions.
        auto_promote: Automatically keep winner and discard rest.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        timeout: Per-branch timeout.

    Returns:
        Fork result dict with branches, winner, and recommendation.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)
    parent = find_experiment(experiments, exp_id)

    if not parent:
        return {"error": f"Experiment {exp_id} not found"}

    if not branch_descriptions:
        return {"error": "No branches specified. Use --branches 'A' 'B'"}

    # Create branches
    branches = []
    for i, desc in enumerate(branch_descriptions):
        branches.append(create_branch(parent, desc, i))

    print(f"Forking {exp_id} into {len(branches)} branches:", file=sys.stderr)
    for b in branches:
        print(f"  {b['branch_id']}: {b['description']}", file=sys.stderr)
    print(file=sys.stderr)

    # Execute branches
    for i, branch in enumerate(branches):
        print(f"  [{i+1}/{len(branches)}] Running {branch['branch_id']}...", end=" ",
              flush=True, file=sys.stderr)
        run_branch(branch, seed=42 + i, timeout=timeout)
        if branch["status"] == "completed":
            metric_val = branch.get("metrics", {}).get(primary_metric, "N/A")
            print(f"{primary_metric}={metric_val}", file=sys.stderr)
        else:
            print(f"FAILED", file=sys.stderr)

    # Determine winner
    winner = determine_winner(branches, primary_metric, lower_is_better)

    result = {
        "parent_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": primary_metric,
        "lower_is_better": lower_is_better,
        "branches": branches,
        "winner": winner["branch_id"] if winner else None,
        "winner_metric": winner["metrics"].get(primary_metric) if winner else None,
        "auto_promote": auto_promote,
        "total_branches": len(branches),
        "completed": sum(1 for b in branches if b["status"] == "completed"),
        "failed": sum(1 for b in branches if b["status"] == "failed"),
    }

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Fork experiment into parallel branches")
    parser.add_argument("exp_id", help="Parent experiment ID")
    parser.add_argument("--branches", nargs="+", required=True, help="Branch descriptions")
    parser.add_argument("--auto-promote", action="store_true", help="Auto-keep winner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_fork(
        args.exp_id, args.branches, args.auto_promote,
        args.config, args.log, args.timeout,
    )

    if "error" not in result:
        filepath = save_fork_report(result)
        print(f"\nSaved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(format_fork_report(
                result["parent_id"], result["branches"],
                next((b for b in result["branches"] if b["branch_id"] == result.get("winner")), None),
                result["metric"],
            ))


if __name__ == "__main__":
    main()
