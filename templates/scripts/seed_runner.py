#!/usr/bin/env python3
"""Multi-seed experiment runner for statistical rigor.

Runs the same experiment configuration across N random seeds, computes
mean/std/confidence intervals, and flags seed-sensitive results.

Prevents publishing lucky seeds by requiring distributional evidence
before claiming a result.

Usage:
    python scripts/seed_runner.py                          # 5 seeds, best experiment
    python scripts/seed_runner.py --quick                  # 3 seeds for fast checks
    python scripts/seed_runner.py --seeds 10               # Custom seed count
    python scripts/seed_runner.py --exp-id exp-042         # Specific experiment
    python scripts/seed_runner.py --seed-list 42,123,456   # Custom seed values
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments

# Default seed values (diverse primes and powers for good coverage)
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3141, 4096, 7919]
DEFAULT_N_SEEDS = 5
QUICK_N_SEEDS = 3
CV_THRESHOLD = 5.0  # Percent — above this, result is seed-sensitive


def get_experiment_config(
    experiments: list[dict],
    exp_id: str | None,
    primary_metric: str,
    lower_is_better: bool,
) -> dict | None:
    """Retrieve experiment config from log by ID, or find the best experiment."""
    if exp_id:
        for exp in experiments:
            if exp.get("experiment_id") == exp_id:
                return exp
        return None

    # Find best kept experiment
    best = None
    best_val = float("inf") if lower_is_better else float("-inf")
    for exp in experiments:
        if exp.get("status") != "kept":
            continue
        val = exp.get("metrics", {}).get(primary_metric)
        if val is None:
            continue
        if (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val = val
            best = exp
    return best


def run_single_seed(seed: int, timeout: int = 600) -> dict | None:
    """Run train.py with a specific seed and parse metrics from output.

    Returns dict of parsed metrics, or None on failure.
    """
    cmd = ["python", "train.py", "--seed", str(seed)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None

    if proc.returncode != 0:
        return None

    # Parse metrics from --- delimited block
    metrics = {}
    in_block = False
    metadata_keys = {"model_type", "train_seconds"}

    for line in proc.stdout.splitlines():
        line = line.strip()
        if line == "---":
            if in_block:
                break
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

    return metrics if metrics else None


def compute_seed_statistics(
    values: list[float],
    seeds: list[int],
) -> dict:
    """Compute statistical summary for seed study results.

    Returns dict with mean, std, 95% CI, CV%, and sensitivity flag.
    """
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # 95% CI using t-distribution
    if n > 1:
        from scipy import stats as scipy_stats
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * std / np.sqrt(n)
    else:
        margin = 0.0

    ci_lower = mean - margin
    ci_upper = mean + margin

    # Coefficient of variation
    cv = (std / abs(mean) * 100) if mean != 0 else float("inf")

    # Identify best and worst seeds
    best_idx = int(np.argmax(arr))
    worst_idx = int(np.argmin(arr))

    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci_95": [round(ci_lower, 6), round(ci_upper, 6)],
        "cv_percent": round(cv, 2),
        "seed_sensitive": cv > CV_THRESHOLD,
        "best_seed": seeds[best_idx],
        "best_value": round(float(arr[best_idx]), 6),
        "worst_seed": seeds[worst_idx],
        "worst_value": round(float(arr[worst_idx]), 6),
        "range": round(float(np.max(arr) - np.min(arr)), 6),
    }


def run_seed_study(
    n_seeds: int = DEFAULT_N_SEEDS,
    seed_list: list[int] | None = None,
    exp_id: str | None = None,
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
    timeout: int = 600,
) -> dict:
    """Run a complete multi-seed study.

    Args:
        n_seeds: Number of seeds to run.
        seed_list: Explicit seed values (overrides n_seeds if provided).
        exp_id: Specific experiment ID to study (defaults to best).
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        timeout: Per-run timeout in seconds.

    Returns:
        Complete seed study result dict.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    # Determine seeds to use
    configured_seeds = eval_cfg.get("seed_seeds", DEFAULT_SEEDS)
    if seed_list:
        seeds = seed_list
    else:
        seeds = configured_seeds[:n_seeds]

    # Find the target experiment
    experiments = load_experiments(log_path)
    target_exp = get_experiment_config(experiments, exp_id, primary_metric, lower_is_better)

    if not target_exp:
        return {
            "error": f"No experiment found{f' with ID {exp_id}' if exp_id else ''}",
            "experiment_id": exp_id,
        }

    target_id = target_exp.get("experiment_id", "unknown")
    print(f"Seed study for {target_id}", file=sys.stderr)
    print(f"Primary metric: {primary_metric} ({'lower' if lower_is_better else 'higher'} is better)", file=sys.stderr)
    print(f"Seeds: {seeds}", file=sys.stderr)
    print(file=sys.stderr)

    # Run each seed
    results = []
    failed_seeds = []
    for i, seed in enumerate(seeds):
        print(f"  Run {i + 1}/{len(seeds)} (seed={seed})...", end=" ", flush=True, file=sys.stderr)
        metrics = run_single_seed(seed, timeout=timeout)
        if metrics and primary_metric in metrics:
            value = metrics[primary_metric]
            results.append({"seed": seed, "value": value, "metrics": metrics})
            print(f"{primary_metric}={value:.4f}", file=sys.stderr)
        else:
            failed_seeds.append(seed)
            print("FAILED", file=sys.stderr)

    if len(results) < 2:
        return {
            "error": f"Only {len(results)} successful runs — need at least 2 for statistics",
            "experiment_id": target_id,
            "successful_runs": len(results),
            "failed_seeds": failed_seeds,
        }

    # Compute statistics
    values = [r["value"] for r in results]
    seeds_run = [r["seed"] for r in results]
    stats = compute_seed_statistics(values, seeds_run)

    # Build result
    study = {
        "experiment_id": target_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": primary_metric,
        "lower_is_better": lower_is_better,
        "seeds_run": seeds_run,
        "results": [round(v, 6) for v in values],
        "failed_seeds": failed_seeds,
        **stats,
    }

    return study


def save_seed_study(study: dict, output_dir: str = "experiments/seed_studies") -> Path:
    """Save seed study results to YAML file.

    Returns path to the saved file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exp_id = study.get("experiment_id", "unknown")
    filename = f"{exp_id}-seeds.yaml"
    filepath = out_path / filename

    with open(filepath, "w") as f:
        yaml.dump(study, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_seed_report(study: dict) -> str:
    """Format seed study results as a human-readable report."""
    if "error" in study:
        return f"ERROR: {study['error']}"

    exp_id = study["experiment_id"]
    metric = study["metric"]
    direction = "lower" if study.get("lower_is_better", False) else "higher"
    sensitive = study["seed_sensitive"]

    lines = [
        f"# Seed Study: {exp_id}",
        "",
        f"*{metric} ({direction} is better) across {len(study['seeds_run'])} seeds*",
        "",
        "## Results",
        "",
        "| Seed | Value |",
        "|------|-------|",
    ]

    for seed, value in zip(study["seeds_run"], study["results"]):
        marker = ""
        if seed == study["best_seed"]:
            marker = " (best)"
        elif seed == study["worst_seed"]:
            marker = " (worst)"
        lines.append(f"| {seed} | {value:.4f}{marker} |")

    lines.extend([
        "",
        "## Statistics",
        "",
        f"| Statistic | Value |",
        f"|-----------|-------|",
        f"| Mean | {study['mean']:.4f} |",
        f"| Std | {study['std']:.4f} |",
        f"| 95% CI | [{study['ci_95'][0]:.4f}, {study['ci_95'][1]:.4f}] |",
        f"| CV | {study['cv_percent']:.2f}% |",
        f"| Range | {study['range']:.4f} |",
        f"| Best seed | {study['best_seed']} ({study['best_value']:.4f}) |",
        f"| Worst seed | {study['worst_seed']} ({study['worst_value']:.4f}) |",
        "",
        "## Verdict",
        "",
    ])

    if sensitive:
        lines.extend([
            f"**SEED-SENSITIVE** (CV={study['cv_percent']:.2f}% > {CV_THRESHOLD}%)",
            "",
            "This result varies significantly across seeds. Do not report a single-seed result.",
            "Report as: **{metric} = {mean:.4f} +/- {std:.4f}** (mean +/- std over {n} seeds)".format(
                metric=metric,
                mean=study["mean"],
                std=study["std"],
                n=len(study["seeds_run"]),
            ),
        ])
    else:
        lines.extend([
            f"**STABLE** (CV={study['cv_percent']:.2f}% < {CV_THRESHOLD}%)",
            "",
            "Result is robust across seeds. Safe to report.",
            "Report as: **{metric} = {mean:.4f} +/- {std:.4f}** (mean +/- std over {n} seeds, 95% CI [{ci_lo:.4f}, {ci_hi:.4f}])".format(
                metric=metric,
                mean=study["mean"],
                std=study["std"],
                n=len(study["seeds_run"]),
                ci_lo=study["ci_95"][0],
                ci_hi=study["ci_95"][1],
            ),
        ])

    if study.get("failed_seeds"):
        lines.extend([
            "",
            f"**Warning:** {len(study['failed_seeds'])} seeds failed: {study['failed_seeds']}",
        ])

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-seed experiment runner for statistical rigor",
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_N_SEEDS,
        help=f"Number of seeds to run (default: {DEFAULT_N_SEEDS})",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=f"Quick mode: run {QUICK_N_SEEDS} seeds instead of {DEFAULT_N_SEEDS}",
    )
    parser.add_argument(
        "--seed-list", type=str, default=None,
        help="Comma-separated list of specific seed values",
    )
    parser.add_argument(
        "--exp-id", type=str, default=None,
        help="Experiment ID to study (defaults to best experiment)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--log", default="experiments/log.jsonl",
        help="Path to experiment log",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Per-run timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    n_seeds = QUICK_N_SEEDS if args.quick else args.seeds
    seed_list = None
    if args.seed_list:
        seed_list = [int(s.strip()) for s in args.seed_list.split(",")]

    study = run_seed_study(
        n_seeds=n_seeds,
        seed_list=seed_list,
        exp_id=args.exp_id,
        config_path=args.config,
        log_path=args.log,
        timeout=args.timeout,
    )

    # Save results
    if "error" not in study:
        filepath = save_seed_study(study)
        print(f"\nSaved to {filepath}", file=sys.stderr)

    # Output
    if args.json:
        print(json.dumps(study, indent=2))
    else:
        print(format_seed_report(study))


if __name__ == "__main__":
    main()
