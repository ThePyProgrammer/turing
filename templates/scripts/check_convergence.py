#!/usr/bin/env python3
"""Convergence detection for the autoresearch pipeline.

Reads experiments/log.jsonl and checks if the last N experiments
(where N = convergence.patience) show insufficient improvement
over the best prior result.

This is a discrete analogue of early stopping from gradient descent,
adapted for the experiment loop context. The algorithm:

1. Load all "kept" experiments from the JSONL log
2. For each of the last N experiments, compute relative improvement
   over the prior best
3. If all N show < threshold improvement, declare convergence

Usage:
    python scripts/check_convergence.py [--config config.yaml] [--log experiments/log.jsonl]

Exit codes:
    0 = not converged, agent should continue
    2 = converged, agent should stop (signals /loop to halt)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_convergence_config(config_path: str) -> dict:
    """Load convergence parameters from config.yaml.

    Returns dict with keys: patience, improvement_threshold,
    primary_metric, lower_is_better.

    Falls back to conservative defaults if config cannot be loaded.
    """
    defaults = {
        "patience": 3,
        "improvement_threshold": 0.005,
        "primary_metric": "accuracy",
        "lower_is_better": False,
    }

    path = Path(config_path)
    if not path.exists():
        print(f"convergence: Config not found at {config_path}, using defaults.", file=sys.stderr)
        return defaults

    try:
        with open(path) as f:
            config = yaml.safe_load(f)

        convergence_cfg = config.get("convergence", {})
        eval_cfg = config.get("evaluation", {})

        return {
            "patience": convergence_cfg.get("patience", defaults["patience"]),
            "improvement_threshold": convergence_cfg.get(
                "improvement_threshold", defaults["improvement_threshold"]
            ),
            "primary_metric": eval_cfg.get("primary_metric", defaults["primary_metric"]),
            "lower_is_better": eval_cfg.get("lower_is_better", defaults["lower_is_better"]),
        }
    except (yaml.YAMLError, AttributeError) as e:
        print(f"convergence: Error reading config: {e}. Using defaults.", file=sys.stderr)
        return defaults


def load_kept_experiments(log_path: str, primary_metric: str) -> list[dict]:
    """Load all 'kept' experiments with valid primary metric values.

    Args:
        log_path: Path to experiments/log.jsonl.
        primary_metric: Metric name to extract.

    Returns:
        List of dicts with 'id' and 'value' keys, in chronological order.
    """
    path = Path(log_path)
    if not path.exists():
        return []

    experiments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                value = entry.get("metrics", {}).get(primary_metric)
                if value is not None and entry.get("status") == "kept":
                    experiments.append({
                        "id": entry.get("experiment_id", "?"),
                        "value": float(value),
                    })
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    return experiments


def compute_relative_improvement(
    current: float,
    prior_best: float,
    lower_is_better: bool,
) -> float:
    """Compute relative improvement of current value over prior best.

    Returns a float where positive = improvement, negative = regression.
    Returns 1.0 if prior_best is zero (any non-zero value is infinite improvement).
    """
    if prior_best == 0:
        return 1.0 if current != 0 else 0.0

    if lower_is_better:
        return (prior_best - current) / abs(prior_best)
    else:
        return (current - prior_best) / abs(prior_best)


def check_convergence(
    experiments: list[dict],
    patience: int,
    improvement_threshold: float,
    lower_is_better: bool,
) -> tuple[bool, int, str]:
    """Check if the experiment loop has converged.

    Args:
        experiments: List of dicts with 'id' and 'value' keys.
        patience: Number of consecutive non-improvements required.
        improvement_threshold: Minimum relative improvement to count.
        lower_is_better: True for metrics like MAE/MSE.

    Returns:
        Tuple of (converged: bool, non_improvements: int, message: str).
    """
    total = len(experiments)

    if total < patience:
        return (
            False,
            0,
            f"Only {total} experiments, need {patience} to check convergence.",
        )

    # Find best value across all experiments
    values = [e["value"] for e in experiments]
    best_value = min(values) if lower_is_better else max(values)

    # Check last N experiments for improvement over their respective prior bests
    non_improvements = 0
    for i in range(total - patience, total):
        prior_values = [e["value"] for e in experiments[:i]]
        if not prior_values:
            continue

        prior_best = min(prior_values) if lower_is_better else max(prior_values)
        current_value = experiments[i]["value"]
        improvement = compute_relative_improvement(current_value, prior_best, lower_is_better)

        if improvement < improvement_threshold:
            non_improvements += 1

    last_n = experiments[-patience:]
    last_values = [round(e["value"], 4) for e in last_n]
    primary_metric = "metric"  # Cosmetic — the caller knows the actual name

    if non_improvements >= patience:
        msg = (
            f"CONVERGED: {patience} consecutive non-improvements "
            f"(threshold: {improvement_threshold * 100:.1f}% relative gain). "
            f"Best={best_value:.4f}, last {patience} values={last_values}"
        )
        return True, non_improvements, msg
    else:
        msg = (
            f"Not converged ({non_improvements}/{patience} non-improvements). "
            f"Best={best_value:.4f}, last {patience} values={last_values}"
        )
        return False, non_improvements, msg


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Check experiment convergence for the autoresearch pipeline"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--log",
        default="experiments/log.jsonl",
        help="Path to experiment log (default: experiments/log.jsonl)",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_convergence_config(args.config)

    # Load experiments
    experiments = load_kept_experiments(args.log, cfg["primary_metric"])

    # Check convergence
    converged, non_improvements, message = check_convergence(
        experiments=experiments,
        patience=cfg["patience"],
        improvement_threshold=cfg["improvement_threshold"],
        lower_is_better=cfg["lower_is_better"],
    )

    print(f"convergence: {message}", file=sys.stderr)

    if converged:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
