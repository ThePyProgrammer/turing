#!/usr/bin/env python3
"""Hyperparameter sweep tool for the autoresearch pipeline.

Generates cartesian product of sweep parameters and writes an experiment queue.
The agent processes the queue sequentially, one experiment per iteration.

Usage:
    python scripts/sweep.py [sweep_config.yaml]
    python scripts/sweep.py --status  # Show queue progress
    python scripts/sweep.py --next    # Print next pending experiment as JSON
    python scripts/sweep.py --mark <name> <status>  # Mark experiment complete/failed
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import yaml


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Apply dotted-path overrides to a config dict.

    Takes dotted-path keys like "model.hyperparams.n_estimators" and sets
    nested values. Returns a new config dict with overrides applied.

    Args:
        config: Base configuration dictionary.
        overrides: Dict mapping dotted-path keys to values.

    Returns:
        New config dict with overrides applied (original is not mutated).
    """
    result = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return result


def _make_experiment_name(overrides: dict) -> str:
    """Generate a short descriptive name from parameter overrides.

    Example: {"model.hyperparams.n_estimators": 100, "model.hyperparams.max_depth": 4}
    becomes "n100_d4"
    """
    abbreviations = {
        "n_estimators": "n",
        "max_depth": "d",
        "learning_rate": "lr",
        "min_child_weight": "mcw",
        "subsample": "ss",
        "colsample_bytree": "cs",
        "gamma": "g",
        "reg_alpha": "a",
        "reg_lambda": "l",
        "epochs": "ep",
        "batch_size": "bs",
        "hidden_size": "hs",
        "dropout": "do",
        "weight_decay": "wd",
    }

    parts = []
    for key, value in overrides.items():
        param_name = key.split(".")[-1]
        abbrev = abbreviations.get(param_name, param_name[:3])
        parts.append(f"{abbrev}{value}")

    return "_".join(parts)


def generate_queue(sweep_config_path: str) -> None:
    """Generate cartesian product experiment queue from sweep config.

    Reads sweep parameters, computes the cartesian product of all value lists,
    and writes a queue YAML file with one entry per combination.

    Args:
        sweep_config_path: Path to sweep config YAML file.
    """
    config_path = Path(sweep_config_path)
    if not config_path.exists():
        print(f"Error: Sweep config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    sweep_params = sweep_config.get("sweep", {})
    output_path = sweep_config.get("output", "experiments/queue.yaml")

    if not sweep_params:
        print("Error: No sweep parameters defined in config", file=sys.stderr)
        sys.exit(1)

    # Extract parameter names and value lists
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())

    # Generate cartesian product
    combinations = list(itertools.product(*param_values))

    # Build experiment queue
    queue = []
    for combo in combinations:
        overrides = dict(zip(param_names, combo))
        name = _make_experiment_name(overrides)
        queue.append({
            "experiment_name": name,
            "config_overrides": overrides,
            "status": "pending",
        })

    # Write queue
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False)

    num_params = len(param_names)
    num_experiments = len(queue)
    print(f"Generated {num_experiments} experiments from {num_params} parameters")
    print(f"Queue written to: {output_path}")


def show_status(queue_path: str) -> None:
    """Show queue progress: counts of pending/running/complete/failed experiments.

    Args:
        queue_path: Path to experiments/queue.yaml.
    """
    path = Path(queue_path)
    if not path.exists():
        print("No queue found. Run sweep.py to generate one.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        queue = yaml.safe_load(f) or []

    counts: dict[str, int] = {}
    for entry in queue:
        status = entry.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    total = len(queue)
    print(f"Queue: {total} experiments")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


def get_next(queue_path: str) -> None:
    """Print the next pending experiment as JSON for agent consumption.

    Args:
        queue_path: Path to experiments/queue.yaml.
    """
    path = Path(queue_path)
    if not path.exists():
        print("No queue found.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        queue = yaml.safe_load(f) or []

    for entry in queue:
        if entry.get("status") == "pending":
            print(json.dumps(entry, indent=2))
            return

    print("No pending experiments.", file=sys.stderr)
    sys.exit(1)


def mark_experiment(queue_path: str, name: str, new_status: str) -> None:
    """Mark an experiment as complete or failed in the queue.

    Args:
        queue_path: Path to experiments/queue.yaml.
        name: Experiment name to mark.
        new_status: New status (complete, failed, running).
    """
    path = Path(queue_path)
    if not path.exists():
        print("No queue found.", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        queue = yaml.safe_load(f) or []

    found = False
    for entry in queue:
        if entry.get("experiment_name") == name:
            entry["status"] = new_status
            found = True
            break

    if not found:
        print(f"Error: Experiment '{name}' not found in queue", file=sys.stderr)
        sys.exit(1)

    with open(path, "w") as f:
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False)

    print(f"Marked '{name}' as {new_status}")


def _find_queue_path(sweep_config_path: str) -> str:
    """Extract the queue output path from the sweep config."""
    config_path = Path(sweep_config_path)
    if config_path.exists():
        with open(config_path) as f:
            sweep_config = yaml.safe_load(f)
        return sweep_config.get("output", "experiments/queue.yaml")
    return "experiments/queue.yaml"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep tool for the autoresearch pipeline"
    )
    parser.add_argument(
        "sweep_config",
        nargs="?",
        default="sweep_config.yaml",
        help="Path to sweep config YAML (default: sweep_config.yaml)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show queue progress",
    )
    parser.add_argument(
        "--next",
        action="store_true",
        help="Print the next pending experiment as JSON",
    )
    parser.add_argument(
        "--mark",
        nargs=2,
        metavar=("NAME", "STATUS"),
        help="Mark an experiment as complete/failed",
    )

    args = parser.parse_args()
    queue_path = _find_queue_path(args.sweep_config)

    if args.status:
        show_status(queue_path)
    elif args.next:
        get_next(queue_path)
    elif args.mark:
        mark_experiment(queue_path, args.mark[0], args.mark[1])
    else:
        generate_queue(args.sweep_config)


if __name__ == "__main__":
    main()
