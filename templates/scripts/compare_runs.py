"""Side-by-side comparison of two experiments.

Usage: python scripts/compare_runs.py exp-001 exp-002 [--log path/to/log.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

DEFAULT_LOG_PATH = "experiments/log.jsonl"


def load_experiment(log_path: str, experiment_id: str) -> dict | None:
    """Load a single experiment by ID."""
    path = Path(log_path)
    if not path.exists():
        return None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("experiment_id") == experiment_id:
                    return entry
            except json.JSONDecodeError:
                continue
    return None


def load_config() -> dict:
    """Load config.yaml if present, else return defaults."""
    if Path("config.yaml").exists():
        with open("config.yaml") as f:
            return yaml.safe_load(f) or {}
    return {}


def format_comparison(exp_a: dict, exp_b: dict, config: dict) -> str:
    """Format side-by-side comparison of two experiments."""
    id_a = exp_a.get("experiment_id", "?")
    id_b = exp_b.get("experiment_id", "?")

    eval_cfg = config.get("evaluation", {})
    lower_is_better_metrics = set()
    if eval_cfg.get("lower_is_better", False):
        lower_is_better_metrics = set(eval_cfg.get("metrics", []))

    lines = [
        f"{'':20s} {id_a:<20s} {id_b:<20s}",
        "=" * 60,
        "",
        "## Config",
    ]

    config_a = exp_a.get("config", {})
    config_b = exp_b.get("config", {})
    all_config_keys = sorted(set(list(config_a.keys()) + list(config_b.keys())))
    for key in all_config_keys:
        val_a = config_a.get(key, "N/A")
        val_b = config_b.get(key, "N/A")
        marker = " <--" if val_a != val_b else ""
        lines.append(f"  {key:<18s} {str(val_a):<20s} {str(val_b):<20s}{marker}")

    lines.append("")
    lines.append("## Metrics")

    metrics_a = exp_a.get("metrics", {})
    metrics_b = exp_b.get("metrics", {})
    all_metric_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
    for key in all_metric_keys:
        val_a = metrics_a.get(key, "N/A")
        val_b = metrics_b.get(key, "N/A")
        diff_marker = ""
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            if key in lower_is_better_metrics:
                diff_marker = " (better)" if val_b < val_a else " (worse)" if val_b > val_a else ""
            else:
                diff_marker = " (better)" if val_b > val_a else " (worse)" if val_b < val_a else ""
        a_str = f"{val_a:.4f}" if isinstance(val_a, float) else str(val_a)
        b_str = f"{val_b:.4f}" if isinstance(val_b, float) else str(val_b)
        lines.append(f"  {key:<18s} {a_str:<20s} {b_str:<20s}{diff_marker}")

    lines.append("")
    lines.append("## Status")
    lines.append(f"  {'status':<18s} {exp_a.get('status', '?'):<20s} {exp_b.get('status', '?'):<20s}")
    lines.append(f"  {'timestamp':<18s} {exp_a.get('timestamp', '?')[:19]:<20s} {exp_b.get('timestamp', '?')[:19]:<20s}")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare two experiment runs")
    parser.add_argument("exp_a", help="First experiment ID (e.g., exp-001)")
    parser.add_argument("exp_b", help="Second experiment ID (e.g., exp-002)")
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_PATH,
        help=f"Path to experiment log (default: {DEFAULT_LOG_PATH})",
    )
    args = parser.parse_args()

    a = load_experiment(args.log, args.exp_a)
    b = load_experiment(args.log, args.exp_b)

    if a is None:
        print(f"Experiment {args.exp_a} not found in {args.log}")
        sys.exit(1)
    if b is None:
        print(f"Experiment {args.exp_b} not found in {args.log}")
        sys.exit(1)

    config = load_config()
    print(format_comparison(a, b, config))


if __name__ == "__main__":
    main()
