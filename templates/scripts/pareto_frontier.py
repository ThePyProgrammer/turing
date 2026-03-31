#!/usr/bin/env python3
"""Multi-objective Pareto frontier visualization.

Visualizes the Pareto frontier across multiple objectives from
experiment history. Answers "which model is actually best?" when
there are tradeoffs between accuracy, latency, model size, etc.

Extends the existing cost_frontier.py (2D: metric vs train_time) to
N-dimensional Pareto analysis across arbitrary metric sets.

Usage:
    python scripts/pareto_frontier.py                                    # Default metrics
    python scripts/pareto_frontier.py --metrics "accuracy,train_seconds" # Specific metrics
    python scripts/pareto_frontier.py --metrics "accuracy,train_seconds,n_params"  # 3D
    python scripts/pareto_frontier.py --ascii                            # ASCII scatter
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"


def extract_metric_vectors(
    experiments: list[dict],
    metrics: list[str],
    status_filter: str = "kept",
) -> list[dict]:
    """Extract metric vectors from experiments for Pareto analysis.

    Returns list of dicts with experiment_id, model_type, and metric values.
    Only includes experiments that have ALL requested metrics.
    """
    results = []
    for exp in experiments:
        if status_filter and exp.get("status") != status_filter:
            continue

        exp_metrics = exp.get("metrics", {})
        values = {}
        complete = True
        for m in metrics:
            val = exp_metrics.get(m)
            if val is None:
                complete = False
                break
            try:
                values[m] = float(val)
            except (ValueError, TypeError):
                complete = False
                break

        if complete:
            results.append({
                "experiment_id": exp.get("experiment_id", "?"),
                "model_type": exp.get("config", {}).get("model_type", "unknown"),
                "metrics": values,
                "description": exp.get("description", ""),
            })

    return results


def compute_pareto_set(
    data: list[dict],
    metrics: list[str],
    directions: dict[str, str],
) -> list[dict]:
    """Compute N-dimensional Pareto-optimal set.

    An experiment is Pareto-optimal if no other experiment is strictly
    better on ALL metrics simultaneously.

    Args:
        data: List of experiment dicts with metrics.
        metrics: List of metric names.
        directions: Dict mapping metric name to "higher" or "lower".

    Returns:
        List of Pareto-optimal experiment dicts.
    """
    if not data:
        return []

    frontier = []
    for i, candidate in enumerate(data):
        dominated = False
        for j, other in enumerate(data):
            if i == j:
                continue
            # Check if other dominates candidate
            # "other dominates candidate" means other is >= on all metrics and > on at least one
            all_at_least_as_good = True
            strictly_better_on_one = False

            for m in metrics:
                c_val = candidate["metrics"][m]
                o_val = other["metrics"][m]
                direction = directions.get(m, "higher")

                if direction == "higher":
                    if o_val < c_val:
                        all_at_least_as_good = False
                        break
                    if o_val > c_val:
                        strictly_better_on_one = True
                else:  # lower is better
                    if o_val > c_val:
                        all_at_least_as_good = False
                        break
                    if o_val < c_val:
                        strictly_better_on_one = True

            if all_at_least_as_good and strictly_better_on_one:
                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return frontier


def find_closest_pareto_neighbor(
    dominated: dict,
    frontier: list[dict],
    metrics: list[str],
    directions: dict[str, str],
) -> dict | None:
    """Find the closest Pareto-optimal experiment to a dominated one.

    Uses normalized Euclidean distance across all metrics.
    """
    if not frontier:
        return None

    # Compute normalization ranges
    all_points = frontier + [dominated]
    ranges = {}
    for m in metrics:
        values = [p["metrics"][m] for p in all_points]
        r = max(values) - min(values)
        ranges[m] = r if r > 0 else 1.0

    best_dist = float("inf")
    best_neighbor = None

    for fp in frontier:
        dist = 0
        for m in metrics:
            norm_diff = (dominated["metrics"][m] - fp["metrics"][m]) / ranges[m]
            dist += norm_diff ** 2
        dist = math.sqrt(dist)

        if dist < best_dist:
            best_dist = dist
            best_neighbor = fp

    return best_neighbor


def format_ascii_scatter(
    data: list[dict],
    frontier: list[dict],
    x_metric: str,
    y_metric: str,
    width: int = 60,
    height: int = 20,
) -> str:
    """Generate an ASCII scatter plot of two metrics with Pareto frontier marked."""
    if not data:
        return "No data to plot."

    frontier_ids = {e["experiment_id"] for e in frontier}

    x_vals = [d["metrics"][x_metric] for d in data]
    y_vals = [d["metrics"][y_metric] for d in data]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    # Add margin
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    x_min -= x_range * 0.05
    x_max += x_range * 0.05
    y_min -= y_range * 0.05
    y_max += y_range * 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Create grid
    grid = [[" "] * width for _ in range(height)]

    # Plot points
    for d in data:
        x = int((d["metrics"][x_metric] - x_min) / x_range * (width - 1))
        y = int((d["metrics"][y_metric] - y_min) / y_range * (height - 1))
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        y = height - 1 - y  # Flip y axis

        if d["experiment_id"] in frontier_ids:
            grid[y][x] = "*"
        elif grid[y][x] == " ":
            grid[y][x] = "·"

    # Build output
    lines = [f"  {y_metric} vs {x_metric}  (* = Pareto-optimal, · = dominated)", ""]

    # Y axis label
    y_top = f"{y_max:.3f}"
    y_bot = f"{y_min:.3f}"

    for i, row in enumerate(grid):
        label = ""
        if i == 0:
            label = y_top.rjust(8)
        elif i == height - 1:
            label = y_bot.rjust(8)
        else:
            label = " " * 8
        lines.append(f"{label} |{''.join(row)}|")

    # X axis
    lines.append(f"{'':>8} +{'-' * width}+")
    x_label = f"{x_min:.3f}{'':>{width - 12}}{x_max:.3f}"
    lines.append(f"{'':>9} {x_label}")

    return "\n".join(lines)


def format_frontier_report(
    data: list[dict],
    frontier: list[dict],
    metrics: list[str],
    directions: dict[str, str],
) -> str:
    """Format Pareto frontier analysis as a markdown report."""
    lines = [
        f"# Pareto Frontier Analysis",
        "",
        f"*{len(frontier)} Pareto-optimal of {len(data)} experiments across {len(metrics)} metrics*",
        "",
    ]

    # Directions
    dir_strs = [f"{m} ({'↓' if directions.get(m) == 'lower' else '↑'})" for m in metrics]
    lines.append(f"**Metrics:** {', '.join(dir_strs)}")
    lines.append("")

    # Pareto-optimal table
    lines.append("## Pareto-Optimal Experiments")
    lines.append("")

    header = "| Experiment | Model |"
    sep = "|------------|-------|"
    for m in metrics:
        header += f" {m} |"
        sep += f"{'---' * max(len(m) // 3, 1)}--|"
    header += " Notes |"
    sep += "-------|"
    lines.append(header)
    lines.append(sep)

    for exp in frontier:
        row = f"| {exp['experiment_id']} | {exp['model_type']} |"
        for m in metrics:
            row += f" {exp['metrics'][m]:.4f} |"
        # Determine what this experiment is best at
        best_at = []
        for m in metrics:
            is_best = True
            for other in frontier:
                if other is exp:
                    continue
                if directions.get(m) == "lower":
                    if other["metrics"][m] < exp["metrics"][m]:
                        is_best = False
                        break
                else:
                    if other["metrics"][m] > exp["metrics"][m]:
                        is_best = False
                        break
            if is_best:
                best_at.append(f"Best {m}")
        notes = ", ".join(best_at) if best_at else "Balanced"
        row += f" {notes} |"
        lines.append(row)

    # Dominated experiments with nearest neighbor
    dominated = [d for d in data if d not in frontier]
    if dominated:
        lines.extend([
            "",
            "## Dominated Experiments",
            "",
            "| Experiment | Model |",
        ])
        header2 = "|------------|-------|"
        for m in metrics:
            lines[-1] += f" {m} |"
            header2 += f"{'---' * max(len(m) // 3, 1)}--|"
        lines[-1] += " Nearest Pareto |"
        header2 += "----------------|"
        lines.append(header2)

        for exp in dominated[:10]:
            row = f"| {exp['experiment_id']} | {exp['model_type']} |"
            for m in metrics:
                row += f" {exp['metrics'][m]:.4f} |"
            neighbor = find_closest_pareto_neighbor(exp, frontier, metrics, directions)
            if neighbor:
                row += f" {neighbor['experiment_id']} |"
            else:
                row += " — |"
            lines.append(row)

        if len(dominated) > 10:
            lines.append(f"*...and {len(dominated) - 10} more dominated experiments*")

    return "\n".join(lines)


def save_frontier(frontier_data: dict, output_dir: str = "experiments/frontiers") -> Path:
    """Save frontier analysis to YAML file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = out_path / f"frontier-{date_str}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(frontier_data, f, default_flow_style=False, sort_keys=False)
    return filepath


def run_frontier_analysis(
    metrics_str: str | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    ascii_plot: bool = False,
) -> dict:
    """Run Pareto frontier analysis.

    Args:
        metrics_str: Comma-separated metric names (defaults to config metrics).
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        ascii_plot: Whether to generate ASCII scatter plot.

    Returns:
        Frontier analysis result dict.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    # Determine metrics to analyze
    if metrics_str:
        metrics = [m.strip() for m in metrics_str.split(",")]
    else:
        # Default: primary metric + train_seconds
        configured_metrics = eval_cfg.get("metrics", [primary_metric])
        metrics = list(dict.fromkeys(configured_metrics + ["train_seconds"]))

    # Determine direction for each metric
    lower_metrics = {"train_seconds", "latency", "latency_ms", "n_params", "model_size",
                     "mse", "rmse", "mae", "loss", "log_loss", "error_rate"}
    directions = {}
    for m in metrics:
        if m == primary_metric:
            directions[m] = "lower" if lower_is_better else "higher"
        elif m in lower_metrics:
            directions[m] = "lower"
        else:
            directions[m] = "higher"

    # Extract data
    experiments = load_experiments(log_path)
    data = extract_metric_vectors(experiments, metrics)

    if not data:
        return {
            "error": f"No experiments with all metrics: {metrics}",
            "metrics_requested": metrics,
            "hint": "Ensure experiments log all requested metrics.",
        }

    # Compute Pareto set
    frontier = compute_pareto_set(data, metrics, directions)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "directions": directions,
        "total_experiments": len(data),
        "pareto_optimal": len(frontier),
        "frontier": [
            {
                "experiment_id": e["experiment_id"],
                "model_type": e["model_type"],
                "metrics": e["metrics"],
            }
            for e in frontier
        ],
        "dominated": len(data) - len(frontier),
    }

    if ascii_plot and len(metrics) >= 2:
        result["ascii_plot"] = format_ascii_scatter(
            data, frontier, metrics[0], metrics[1],
        )

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Multi-objective Pareto frontier visualization")
    parser.add_argument("--metrics", default=None, help="Comma-separated metric names")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--ascii", action="store_true", help="Include ASCII scatter plot")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    result = run_frontier_analysis(
        metrics_str=args.metrics,
        config_path=args.config,
        log_path=args.log,
        ascii_plot=args.ascii,
    )

    if "error" not in result:
        filepath = save_frontier(result)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            # Load full data for formatting
            config = load_config(args.config)
            experiments = load_experiments(args.log)
            metrics = result["metrics"]
            data = extract_metric_vectors(experiments, metrics)
            frontier_exps = [d for d in data if d["experiment_id"] in
                           {f["experiment_id"] for f in result["frontier"]}]
            report = format_frontier_report(data, frontier_exps, metrics, result["directions"])
            print(report)

            if result.get("ascii_plot"):
                print()
                print(result["ascii_plot"])


if __name__ == "__main__":
    main()
