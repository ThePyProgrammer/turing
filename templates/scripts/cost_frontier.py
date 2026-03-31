"""Compute cost-performance frontier from experiment history.

Answers the question every ML team should ask: "Is that 2% improvement
worth 800x the compute?" Reads experiment log, identifies Pareto-optimal
experiments (best metric for each compute budget), and produces a report
showing the cost-performance tradeoff.

Usage:
    python scripts/cost_frontier.py [--log experiments/log.jsonl] [--config config.yaml]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"


@dataclass
class CostRecord:
    """A single experiment's cost-performance data point."""

    experiment_id: str
    metric_value: float
    train_seconds: float
    model_type: str


def load_cost_data(log_path: str, metric: str) -> list[CostRecord]:
    """Extract cost-performance data from experiment log.

    Only includes kept experiments that have both the requested metric
    and train_seconds recorded in their metrics dict.

    Args:
        log_path: Path to experiments/log.jsonl.
        metric: Name of the metric to use (e.g. "accuracy").

    Returns:
        List of CostRecord with (experiment_id, metric_value, train_seconds, model_type).
    """
    experiments = load_experiments(log_path)
    records = []

    for exp in experiments:
        if exp.get("status") != "kept":
            continue

        metrics = exp.get("metrics", {})
        metric_val = metrics.get(metric)
        train_secs = metrics.get("train_seconds")

        if metric_val is None or train_secs is None:
            continue

        records.append(CostRecord(
            experiment_id=exp.get("experiment_id", "?"),
            metric_value=float(metric_val),
            train_seconds=float(train_secs),
            model_type=exp.get("config", {}).get("model_type", "unknown"),
        ))

    return records


def compute_pareto_frontier(
    data: list[CostRecord], lower_is_better: bool = False
) -> list[CostRecord]:
    """Find Pareto-optimal experiments on the cost-performance frontier.

    An experiment is Pareto-optimal if no other experiment is both
    faster AND has a better metric value.

    Args:
        data: List of CostRecord entries.
        lower_is_better: True if lower metric values are better (e.g. MSE).

    Returns:
        List of CostRecord on the Pareto frontier, sorted by train_seconds.
    """
    if not data:
        return []

    frontier = []
    for candidate in data:
        dominated = False
        for other in data:
            if other is candidate:
                continue
            # "other" dominates "candidate" if other is both faster AND better
            faster = other.train_seconds < candidate.train_seconds
            if lower_is_better:
                better_metric = other.metric_value < candidate.metric_value
            else:
                better_metric = other.metric_value > candidate.metric_value
            # Also dominate if equal time but better metric, or equal metric but faster
            equal_time = other.train_seconds == candidate.train_seconds
            equal_metric = other.metric_value == candidate.metric_value
            if (faster and better_metric) or (faster and equal_metric) or (equal_time and better_metric):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    frontier.sort(key=lambda r: r.train_seconds)
    return frontier


def compute_cost_efficiency(
    data: list[CostRecord], lower_is_better: bool = False
) -> list[dict]:
    """Compute metric improvement per second relative to baseline.

    Baseline is the worst-performing experiment. For each experiment,
    efficiency = (metric_improvement_over_baseline) / train_seconds.

    Args:
        data: List of CostRecord entries.
        lower_is_better: True if lower metric values are better.

    Returns:
        List of dicts with experiment_id, metric_value, train_seconds,
        metric_per_second, and model_type. Sorted by efficiency descending.
    """
    if not data:
        return []

    if lower_is_better:
        baseline = max(r.metric_value for r in data)
    else:
        baseline = min(r.metric_value for r in data)

    results = []
    for r in data:
        improvement = abs(r.metric_value - baseline)
        efficiency = improvement / r.train_seconds if r.train_seconds > 0 else 0.0
        results.append({
            "experiment_id": r.experiment_id,
            "metric_value": r.metric_value,
            "train_seconds": r.train_seconds,
            "model_type": r.model_type,
            "improvement_over_baseline": improvement,
            "metric_per_second": efficiency,
        })

    results.sort(key=lambda x: -x["metric_per_second"])
    return results


def _format_seconds(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_cost_report(
    data: list[CostRecord],
    frontier: list[CostRecord],
    metric_name: str,
    lower_is_better: bool = False,
) -> str:
    """Format cost-performance analysis as a text report.

    Shows all experiments with train_seconds, metric, and whether
    each is on the Pareto frontier. Includes a summary comparing
    the best metric vs the most cost-efficient alternative.

    Args:
        data: All CostRecord entries.
        frontier: Pareto-optimal CostRecord entries.
        metric_name: Name of the metric for display.
        lower_is_better: True if lower metric values are better.

    Returns:
        Formatted text report.
    """
    if not data:
        return "No cost-performance data available (no experiments have train_seconds)."

    frontier_ids = {r.experiment_id for r in frontier}
    efficiency = compute_cost_efficiency(data, lower_is_better)
    eff_map = {e["experiment_id"]: e for e in efficiency}

    lines = [
        "## Cost-Performance Frontier",
        "",
    ]

    # Table header
    header = f"{'ID':<10} {'Model':<15} {metric_name:>12} {'Time':>10} {'Eff.':>10} {'Pareto':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    # Sort by metric (best first)
    sorted_data = sorted(
        data,
        key=lambda r: r.metric_value,
        reverse=not lower_is_better,
    )

    for r in sorted_data:
        on_frontier = "  YES" if r.experiment_id in frontier_ids else ""
        eff = eff_map.get(r.experiment_id, {})
        eff_val = eff.get("metric_per_second", 0.0)
        lines.append(
            f"{r.experiment_id:<10} {r.model_type:<15} {r.metric_value:>12.4f} "
            f"{_format_seconds(r.train_seconds):>10} {eff_val:>10.6f} {on_frontier:>8}"
        )

    # Summary
    if lower_is_better:
        best_metric_exp = min(data, key=lambda r: r.metric_value)
    else:
        best_metric_exp = max(data, key=lambda r: r.metric_value)

    most_efficient = efficiency[0] if efficiency else None
    cheapest_frontier = frontier[0] if frontier else None

    lines.extend(["", "### Summary", ""])
    lines.append(
        f"Best {metric_name}: {best_metric_exp.experiment_id} "
        f"({best_metric_exp.metric_value:.4f}, "
        f"{_format_seconds(best_metric_exp.train_seconds)})"
    )

    if most_efficient:
        lines.append(
            f"Best cost-efficiency: {most_efficient['experiment_id']} "
            f"({most_efficient['metric_value']:.4f}, "
            f"{_format_seconds(most_efficient['train_seconds'])})"
        )

    # Compute the "is it worth it?" comparison
    if cheapest_frontier and best_metric_exp.experiment_id != cheapest_frontier.experiment_id:
        if best_metric_exp.metric_value != cheapest_frontier.metric_value:
            metric_diff = abs(best_metric_exp.metric_value - cheapest_frontier.metric_value)
            if lower_is_better:
                pct_diff = metric_diff / cheapest_frontier.metric_value * 100
            else:
                pct_diff = metric_diff / cheapest_frontier.metric_value * 100

            if cheapest_frontier.train_seconds > 0:
                compute_ratio = best_metric_exp.train_seconds / cheapest_frontier.train_seconds
            else:
                compute_ratio = float("inf")

            lines.append(
                f"The {pct_diff:.1f}% improvement costs "
                f"{compute_ratio:.0f}x more compute."
            )

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for cost-performance analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze cost-performance frontier from experiment log.",
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_PATH,
        help=f"Path to experiment log (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    data = load_cost_data(args.log, metric)
    frontier = compute_pareto_frontier(data, lower_is_better)
    report = format_cost_report(data, frontier, metric, lower_is_better)
    print(report)


if __name__ == "__main__":
    main()
