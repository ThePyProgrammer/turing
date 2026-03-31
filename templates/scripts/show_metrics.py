"""Display experiment metrics from experiments/log.jsonl.

Reads the JSONL experiment log and prints a formatted table,
highlighting the current best by primary metric. This is the
agent's primary observation tool at the start of each loop iteration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"


def find_best(experiments: list[dict], primary_metric: str, lower_is_better: bool) -> str | None:
    """Find experiment_id with best primary metric among 'keep' entries."""
    best_id = None
    best_value = float("inf") if lower_is_better else float("-inf")

    for exp in experiments:
        if exp.get("status") != "kept":
            continue
        value = exp.get("metrics", {}).get(primary_metric)
        if value is None:
            continue
        if lower_is_better and value < best_value:
            best_value = value
            best_id = exp["experiment_id"]
        elif not lower_is_better and value > best_value:
            best_value = value
            best_id = exp["experiment_id"]
    return best_id


def format_table(experiments: list[dict], best_id: str | None, metric_names: list[str]) -> str:
    """Format experiments as a text table."""
    if not experiments:
        return "No experiments logged yet."

    # Detect if any experiment has train_seconds
    has_train_seconds = any(
        exp.get("metrics", {}).get("train_seconds") is not None
        for exp in experiments
    )

    # Build dynamic header based on configured metrics
    metric_headers = "".join(f"{m:>12}" for m in metric_names)
    time_header = f"{'Time':>10}" if has_train_seconds else ""
    header = f"{'ID':<10} {'Status':<10} {'Model':<15}{metric_headers}{time_header} {'Timestamp':<22}"
    sep = "-" * len(header)
    lines = [header, sep]

    for exp in experiments:
        metrics = exp.get("metrics", {})
        model_type = exp.get("config", {}).get("model_type", "unknown")

        metric_values = ""
        for m in metric_names:
            val = metrics.get(m)
            if isinstance(val, (int, float)):
                metric_values += f"{val:>12.4f}"
            else:
                metric_values += f"{'N/A':>12}"

        time_col = ""
        if has_train_seconds:
            train_secs = metrics.get("train_seconds")
            if train_secs is not None:
                if train_secs < 60:
                    time_col = f"{train_secs:.1f}s"
                elif train_secs < 3600:
                    time_col = f"{train_secs / 60:.1f}m"
                else:
                    time_col = f"{train_secs / 3600:.1f}h"
                time_col = f"{time_col:>10}"
            else:
                time_col = f"{'N/A':>10}"

        ts = exp.get("timestamp", "")[:19]
        marker = " *BEST*" if exp.get("experiment_id") == best_id else ""
        line = f"{exp.get('experiment_id', '?'):<10} {exp.get('status', '?'):<10} {model_type:<15}{metric_values}{time_col} {ts}{marker}"
        lines.append(line)

    return "\n".join(lines)


def get_experiment_diffs(experiments: list[dict], max_diffs: int = 3) -> str:
    """Get git diffs for recent discarded experiments."""
    import subprocess

    discarded = [e for e in experiments if e.get("status") == "discarded"]
    if not discarded:
        return ""

    recent = discarded[-max_diffs:]
    lines = ["\n--- Recent Failed Experiment Diffs ---\n"]

    for exp in recent:
        exp_id = exp.get("experiment_id", "unknown")
        description = exp.get("description", "no description")
        lines.append(f"=== {exp_id}: {description} ===")

        # Try to find the experiment branch
        branch = f"exp/{exp_id.replace('exp-', '')}"
        try:
            result = subprocess.run(
                ["git", "diff", f"main...{branch}", "--", "train.py", "config.yaml"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Truncate long diffs
                diff_lines = result.stdout.strip().splitlines()
                if len(diff_lines) > 30:
                    diff_lines = diff_lines[:30] + [f"... ({len(diff_lines) - 30} more lines)"]
                lines.append("\n".join(diff_lines))
            else:
                lines.append("  (branch not found or no diff)")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            lines.append("  (git not available)")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Show experiment metrics")
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_PATH,
        help=f"Path to experiment log (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        help="Show only last N experiments",
    )
    parser.add_argument(
        "--with-diffs",
        action="store_true",
        help="Include git diffs for discarded experiments",
    )
    args = parser.parse_args()

    config = load_config()
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    metric_names = eval_cfg.get("metrics", ["accuracy", "f1_weighted"])
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(args.log)

    if args.last and args.last > 0:
        experiments = experiments[-args.last:]

    best_id = find_best(experiments, primary_metric, lower_is_better)
    print(format_table(experiments, best_id, metric_names))

    if args.with_diffs:
        all_experiments = load_experiments(args.log)
        diffs = get_experiment_diffs(all_experiments)
        if diffs:
            print(diffs)


if __name__ == "__main__":
    main()
