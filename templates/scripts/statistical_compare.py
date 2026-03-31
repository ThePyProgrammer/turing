#!/usr/bin/env python3
"""Multi-run statistical comparison for the autoresearch pipeline.

Runs the same configuration N times with different random seeds and
compares metric distributions rather than point estimates. This prevents
keep/discard decisions based on noise.

A model that scored 0.87 vs 0.86 on a single run might just be seed
variance. This script runs both configurations multiple times and
uses the Mann-Whitney U test to determine if the difference is real.

Usage:
    python scripts/statistical_compare.py --n-runs 3 [--config config.yaml]
    python scripts/statistical_compare.py --compare <log1.jsonl> <log2.jsonl>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def run_multi_seed(
    n_runs: int,
    config_path: str = "config.yaml",
    base_seed: int = 42,
) -> list[dict]:
    """Run training N times with different seeds, collect metrics.

    Args:
        n_runs: Number of runs to execute.
        config_path: Path to config.yaml.
        base_seed: Starting seed (incremented for each run).

    Returns:
        List of metric dicts from each run.
    """
    results = []
    for i in range(n_runs):
        seed = base_seed + i
        print(f"  Run {i+1}/{n_runs} (seed={seed})...", file=sys.stderr, end=" ")

        cmd = f"python train.py --config {config_path} --seed {seed}"
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=600,
        )

        if proc.returncode != 0:
            print(f"FAILED", file=sys.stderr)
            continue

        # Parse metrics from output (between --- delimiters)
        metrics = parse_metrics_from_output(proc.stdout)
        if metrics:
            metrics["seed"] = seed
            results.append(metrics)
            primary = next((v for k, v in metrics.items() if k not in ("seed", "model_type", "train_seconds")), None)
            print(f"done ({primary})", file=sys.stderr)
        else:
            print(f"no metrics parsed", file=sys.stderr)

    return results


def parse_metrics_from_output(output: str) -> dict:
    """Parse metrics from the --- delimited format in train.py output."""
    lines = output.strip().split("\n")
    in_block = False
    metrics = {}
    metadata_keys = {"model_type", "train_seconds"}

    for line in lines:
        line = line.strip()
        if line == "---":
            if in_block:
                break  # end of block
            in_block = True
            continue
        if in_block and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in metadata_keys:
                metrics[key] = value
            else:
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value

    return metrics


def compute_statistics(results: list[dict], metric_name: str) -> dict:
    """Compute summary statistics for a metric across runs.

    Returns dict with mean, std, min, max, ci_lower, ci_upper (95%).
    """
    values = [r[metric_name] for r in results if metric_name in r and isinstance(r[metric_name], (int, float))]

    if not values:
        return {"n": 0, "mean": None, "std": None}

    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # 95% confidence interval (t-distribution approximation for small n)
    if n > 1:
        from scipy import stats as scipy_stats
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * std / np.sqrt(n)
    else:
        margin = 0.0

    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "ci_lower": round(mean - margin, 6),
        "ci_upper": round(mean + margin, 6),
        "values": [round(float(v), 6) for v in arr],
    }


def mann_whitney_test(values_a: list[float], values_b: list[float]) -> dict:
    """Run Mann-Whitney U test comparing two sets of metric values.

    Returns dict with statistic, p_value, and verdict.
    """
    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "statistic": None,
            "p_value": None,
            "verdict": "insufficient_data",
            "detail": f"Need at least 2 values per group (got {len(values_a)} and {len(values_b)})",
        }

    from scipy import stats as scipy_stats
    stat, p_value = scipy_stats.mannwhitneyu(
        values_a, values_b, alternative="two-sided",
    )

    if p_value < 0.05:
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        verdict = "significantly_different"
        detail = f"p={p_value:.4f} < 0.05 — the difference is statistically significant"
    else:
        verdict = "not_significant"
        detail = f"p={p_value:.4f} >= 0.05 — the difference could be random noise"

    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "verdict": verdict,
        "detail": detail,
    }


def format_comparison_report(
    stats_a: dict,
    stats_b: dict,
    test_result: dict,
    label_a: str,
    label_b: str,
    metric_name: str,
    lower_is_better: bool,
) -> str:
    """Format a statistical comparison report."""
    direction = "lower" if lower_is_better else "higher"
    lines = [
        f"# Statistical Comparison: {metric_name} ({direction} is better)",
        "",
        f"| Statistic | {label_a} | {label_b} |",
        f"|-----------|{'---' * len(label_a)}--|{'---' * len(label_b)}--|",
        f"| N runs | {stats_a['n']} | {stats_b['n']} |",
        f"| Mean | {stats_a['mean']:.4f} | {stats_b['mean']:.4f} |",
        f"| Std | {stats_a['std']:.4f} | {stats_b['std']:.4f} |",
        f"| Min | {stats_a['min']:.4f} | {stats_b['min']:.4f} |",
        f"| Max | {stats_a['max']:.4f} | {stats_b['max']:.4f} |",
        f"| 95% CI | [{stats_a['ci_lower']:.4f}, {stats_a['ci_upper']:.4f}] | [{stats_b['ci_lower']:.4f}, {stats_b['ci_upper']:.4f}] |",
        "",
        "## Mann-Whitney U Test",
        "",
        f"- **Verdict:** {test_result['verdict']}",
        f"- **Detail:** {test_result['detail']}",
    ]

    if test_result["statistic"] is not None:
        lines.append(f"- **U statistic:** {test_result['statistic']}")
        lines.append(f"- **p-value:** {test_result['p_value']}")

    lines.extend(["", "## Recommendation", ""])

    if test_result["verdict"] == "significantly_different":
        mean_a = stats_a["mean"]
        mean_b = stats_b["mean"]
        if lower_is_better:
            better = label_a if mean_a < mean_b else label_b
        else:
            better = label_a if mean_a > mean_b else label_b
        lines.append(f"**{better}** is statistically better. Safe to keep.")
    elif test_result["verdict"] == "not_significant":
        lines.append("The difference is not statistically significant. Consider:")
        lines.append("- Running more trials (increase n_runs)")
        lines.append("- Keeping the simpler/faster configuration")
        lines.append("- Treating this as a tie and moving to a different hypothesis")
    else:
        lines.append("Insufficient data for statistical comparison. Run more trials.")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-run statistical comparison for ML experiments"
    )
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    print(f"Running {args.n_runs} trials with seeds {args.seed} to {args.seed + args.n_runs - 1}...", file=sys.stderr)
    results = run_multi_seed(args.n_runs, args.config, args.seed)

    if not results:
        print("No successful runs.", file=sys.stderr)
        sys.exit(1)

    # Load config for metric info
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    eval_cfg = config.get("evaluation", {})
    metric = eval_cfg.get("primary_metric", "accuracy")

    stats = compute_statistics(results, metric)
    print(f"\n{metric} over {stats['n']} runs:")
    print(f"  Mean:   {stats['mean']:.4f}")
    print(f"  Std:    {stats['std']:.4f}")
    print(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
    print(f"  Values: {stats['values']}")


if __name__ == "__main__":
    main()
