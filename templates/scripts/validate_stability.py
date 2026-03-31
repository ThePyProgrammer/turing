"""Stability validation for experiment metrics.

Runs the training pipeline N times and computes coefficient of variation.
If CV > 5%, the metric is too noisy for reliable single-run comparison.

Usage:
    python scripts/validate_stability.py              # Check stability (5 runs)
    python scripts/validate_stability.py --runs 10    # Custom run count
    python scripts/validate_stability.py --auto       # Auto-fix if unstable
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_RUNS = 5
CV_THRESHOLD = 5.0  # Percent


def run_experiment(seed: int) -> float | None:
    """Run a single training pass and extract the primary metric."""
    config = load_config()
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")

    result = subprocess.run(
        ["python", "train.py", "--seed", str(seed)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Parse metric from stdout (between --- delimiters)
    output = result.stdout
    in_block = False
    for line in output.splitlines():
        if line.strip() == "---":
            in_block = not in_block
            continue
        if in_block:
            match = re.match(rf"^{re.escape(primary_metric)}:\s*(.+)$", line.strip())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
    return None


def load_config() -> dict:
    """Load config.yaml."""
    if Path("config.yaml").exists():
        with open("config.yaml") as f:
            return yaml.safe_load(f) or {}
    return {}


def check_stability(n_runs: int = DEFAULT_RUNS) -> dict:
    """Run N experiments and compute stability metrics."""
    config = load_config()
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")

    print(f"Running {n_runs} stability validation passes...")
    print(f"Primary metric: {primary_metric}")
    print()

    results = []
    for i in range(n_runs):
        seed = 42 + i  # Deterministic but different seeds
        print(f"  Run {i + 1}/{n_runs} (seed={seed})...", end=" ", flush=True)
        value = run_experiment(seed)
        if value is not None:
            results.append(value)
            print(f"{primary_metric}={value:.4f}")
        else:
            print("FAILED (no metric)")

    if len(results) < 2:
        return {
            "stable": False,
            "reason": f"Only {len(results)} successful runs out of {n_runs}",
            "results": results,
            "recommendation": "Fix pipeline errors before validating stability",
        }

    mean = statistics.mean(results)
    stdev = statistics.stdev(results)
    cv = (stdev / abs(mean) * 100) if mean != 0 else float("inf")

    stable = cv < CV_THRESHOLD

    return {
        "stable": stable,
        "n_runs": len(results),
        "mean": round(mean, 4),
        "stdev": round(stdev, 4),
        "cv_percent": round(cv, 2),
        "min": round(min(results), 4),
        "max": round(max(results), 4),
        "results": results,
        "recommendation": (
            "Metric is stable — single-run evaluation is reliable."
            if stable
            else f"CV={cv:.1f}% exceeds {CV_THRESHOLD}% threshold. "
            f"Recommend setting evaluation.n_runs: 3 in config.yaml "
            f"to use median of 3 runs per experiment."
        ),
    }


def auto_fix_config() -> bool:
    """Set evaluation.n_runs: 3 in config.yaml if not already set."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    eval_cfg = config.setdefault("evaluation", {})
    if eval_cfg.get("n_runs", 1) >= 3:
        return False  # Already configured

    eval_cfg["n_runs"] = 3
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate experiment metric stability")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help=f"Number of runs (default: {DEFAULT_RUNS})")
    parser.add_argument("--auto", action="store_true", help="Auto-configure multi-run if unstable")
    args = parser.parse_args()

    result = check_stability(args.runs)

    print()
    print("=" * 50)
    print(f"  Stability: {'STABLE' if result['stable'] else 'UNSTABLE'}")
    if "mean" in result:
        print(f"  Mean:      {result['mean']}")
        print(f"  Stdev:     {result['stdev']}")
        print(f"  CV:        {result['cv_percent']}%")
        print(f"  Range:     [{result['min']}, {result['max']}]")
    print(f"  Runs:      {result.get('n_runs', 0)}")
    print()
    print(f"  {result['recommendation']}")
    print("=" * 50)

    if args.auto and not result["stable"]:
        if auto_fix_config():
            print()
            print("  AUTO-FIX: Set evaluation.n_runs: 3 in config.yaml")
            print("  Each experiment will now run 3 times and report the median.")
        else:
            print()
            print("  evaluation.n_runs already >= 3, no changes made.")


if __name__ == "__main__":
    main()
