#!/usr/bin/env python3
"""Performance regression gate for the autoresearch pipeline.

After any code or dependency change, re-runs the best experiment and
verifies metrics haven't degraded. CI for your model — catches silent
regressions from library upgrades, data pipeline changes, or accidental
train.py edits.

Verdicts:
    pass     — all metrics within tolerance
    warning  — some metrics degraded within 2x tolerance
    fail     — any metric degraded beyond tolerance

Usage:
    python scripts/regression_gate.py
    python scripts/regression_gate.py --tolerance 0.01
    python scripts/regression_gate.py --against exp-042
    python scripts/regression_gate.py --quick
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_TOLERANCE = 0.01  # 1% relative
DEFAULT_RUNS = 3
QUICK_RUNS = 1


def find_best_experiment(
    experiments: list[dict],
    primary_metric: str,
    lower_is_better: bool = False,
) -> dict | None:
    """Find the best experiment by primary metric."""
    kept = [e for e in experiments if e.get("status") == "kept"]
    if not kept:
        # Fall back to all experiments with the metric
        kept = [e for e in experiments if primary_metric in e.get("metrics", {})]

    if not kept:
        return None

    def metric_val(exp):
        return exp.get("metrics", {}).get(primary_metric, float("inf") if lower_is_better else float("-inf"))

    if lower_is_better:
        return min(kept, key=metric_val)
    return max(kept, key=metric_val)


def capture_environment() -> dict:
    """Capture current environment for regression report."""
    env = {"python_version": sys.version.split()[0]}

    try:
        result = subprocess.run(
            ["pip", "freeze"], capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            packages = {}
            for line in result.stdout.strip().splitlines():
                if "==" in line:
                    pkg, ver = line.split("==", 1)
                    packages[pkg.lower()] = ver
            env["packages"] = packages
    except (subprocess.TimeoutExpired, FileNotFoundError):
        env["packages"] = {}

    # Git info
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            env["git_commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "diff", "--stat"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            env["git_dirty"] = bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return env


def diff_environments(original: dict | None, current: dict) -> list[dict]:
    """Compare environments and return list of differences."""
    if not original:
        return [{"field": "environment", "detail": "No original environment snapshot"}]

    diffs = []
    orig_pkgs = original.get("packages", {})
    curr_pkgs = current.get("packages", {})

    critical_packages = {
        "numpy", "scipy", "scikit-learn", "sklearn", "pandas",
        "torch", "tensorflow", "xgboost", "lightgbm", "catboost",
    }

    for pkg in sorted(set(orig_pkgs) | set(curr_pkgs)):
        orig_ver = orig_pkgs.get(pkg)
        curr_ver = curr_pkgs.get(pkg)
        if orig_ver and curr_ver and orig_ver != curr_ver:
            severity = "critical" if pkg in critical_packages else "info"
            diffs.append({
                "field": f"package:{pkg}",
                "original": orig_ver,
                "current": curr_ver,
                "severity": severity,
                "detail": f"{pkg}: {orig_ver} -> {curr_ver}",
            })

    orig_py = original.get("python_version")
    curr_py = current.get("python_version")
    if orig_py and curr_py and orig_py != curr_py:
        diffs.append({
            "field": "python_version",
            "original": orig_py,
            "current": curr_py,
            "severity": "warning",
            "detail": f"Python: {orig_py} -> {curr_py}",
        })

    return diffs


def run_regression_check(
    seed: int,
    timeout: int = 600,
) -> dict | None:
    """Run train.py once and return parsed metrics."""
    try:
        result = subprocess.run(
            ["python", "train.py", "--seed", str(seed)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    metrics = {}
    in_block = False
    metadata_keys = {"model_type", "train_seconds"}

    for line in result.stdout.splitlines():
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


def determine_verdict(
    original_metrics: dict,
    new_metrics_list: list[dict],
    primary_metric: str,
    tolerance: float,
    lower_is_better: bool = False,
) -> dict:
    """Determine regression verdict by comparing metrics.

    Args:
        original_metrics: Original experiment metrics.
        new_metrics_list: List of metric dicts from re-run(s).
        primary_metric: Name of primary metric.
        tolerance: Relative tolerance threshold.
        lower_is_better: Whether lower metric is better.

    Returns:
        Verdict dict with pass/warning/fail, per-metric details.
    """
    per_metric = {}
    overall_verdict = "pass"

    # Get all numeric metric keys
    all_keys = set()
    for nm in new_metrics_list:
        all_keys.update(nm.keys())
    all_keys &= set(original_metrics.keys())

    # Filter to numeric metrics only
    numeric_keys = sorted(
        k for k in all_keys
        if isinstance(original_metrics.get(k), (int, float))
        and k not in {"model_type", "train_seconds"}
    )

    for key in numeric_keys:
        orig_val = original_metrics[key]
        new_vals = [nm[key] for nm in new_metrics_list if key in nm and isinstance(nm.get(key), (int, float))]

        if not new_vals:
            continue

        new_mean = float(np.mean(new_vals))
        delta = new_mean - orig_val
        rel_diff = abs(delta) / abs(orig_val) if orig_val != 0 else abs(delta)

        # Determine direction (did it get worse?)
        if lower_is_better:
            degraded = delta > 0  # Higher is worse
        else:
            degraded = delta < 0  # Lower is worse

        # Determine per-metric verdict
        if not degraded or rel_diff <= 0:
            metric_verdict = "pass"
        elif rel_diff <= tolerance:
            metric_verdict = "pass"
        elif rel_diff <= 2 * tolerance:
            metric_verdict = "warning"
        else:
            metric_verdict = "fail"

        entry = {
            "original": round(orig_val, 6),
            "new_mean": round(new_mean, 6),
            "new_values": [round(v, 6) for v in new_vals],
            "delta": round(delta, 6),
            "relative_diff": round(rel_diff, 6),
            "degraded": degraded,
            "verdict": metric_verdict,
        }

        if len(new_vals) > 1:
            entry["new_std"] = round(float(np.std(new_vals, ddof=1)), 6)

        per_metric[key] = entry

        # Update overall verdict
        if metric_verdict == "fail" and overall_verdict != "fail":
            overall_verdict = "fail"
        elif metric_verdict == "warning" and overall_verdict == "pass":
            overall_verdict = "warning"

    return {
        "verdict": overall_verdict,
        "per_metric": per_metric,
        "primary_metric": primary_metric,
        "tolerance": tolerance,
    }


def regression_gate(
    tolerance: float = DEFAULT_TOLERANCE,
    against: str | None = None,
    quick: bool = False,
    n_runs: int = DEFAULT_RUNS,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    timeout: int = 600,
) -> dict:
    """Run a complete regression check.

    Args:
        tolerance: Relative tolerance for metric degradation.
        against: Specific experiment ID to check against (default: best).
        quick: Quick mode — 1 run instead of full seed study.
        n_runs: Number of runs (overridden by quick).
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        timeout: Per-run timeout in seconds.

    Returns:
        Complete regression check report.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)

    if against:
        # Find specific experiment
        baseline = None
        for exp in experiments:
            if exp.get("experiment_id") == against:
                baseline = exp
                break
        if not baseline:
            return {"error": f"Experiment {against} not found in {log_path}"}
    else:
        baseline = find_best_experiment(experiments, primary_metric, lower_is_better)
        if not baseline:
            return {"error": f"No experiments found in {log_path}"}

    baseline_metrics = baseline.get("metrics", {})
    baseline_id = baseline.get("experiment_id", "unknown")
    baseline_value = baseline_metrics.get(primary_metric)

    if baseline_value is None:
        return {"error": f"Experiment {baseline_id} has no {primary_metric} metric"}

    # Determine number of runs
    actual_runs = QUICK_RUNS if quick else n_runs

    print(f"Regression check against {baseline_id}", file=sys.stderr)
    print(f"Baseline {primary_metric}: {baseline_value:.4f}", file=sys.stderr)
    print(f"Tolerance: {tolerance * 100:.1f}%", file=sys.stderr)
    print(f"Runs: {actual_runs} ({'quick' if quick else 'full'})", file=sys.stderr)
    print(file=sys.stderr)

    # Capture current environment
    current_env = capture_environment()
    original_env = baseline.get("environment")
    env_diffs = diff_environments(original_env, current_env)

    # Run checks
    seed = baseline.get("config", {}).get("hyperparams", {}).get("seed", 42)
    new_metrics_list = []
    failed_runs = 0

    for i in range(actual_runs):
        run_seed = seed + i
        print(f"  Run {i + 1}/{actual_runs} (seed={run_seed})...", end=" ", flush=True, file=sys.stderr)
        metrics = run_regression_check(run_seed, timeout=timeout)
        if metrics and primary_metric in metrics:
            new_metrics_list.append(metrics)
            print(f"{primary_metric}={metrics[primary_metric]:.4f}", file=sys.stderr)
        else:
            failed_runs += 1
            print("FAILED", file=sys.stderr)

    if not new_metrics_list:
        return {
            "error": f"All {actual_runs} regression runs failed",
            "baseline_id": baseline_id,
        }

    # Determine verdict
    verdict_info = determine_verdict(
        baseline_metrics, new_metrics_list, primary_metric, tolerance, lower_is_better,
    )

    report = {
        "baseline_id": baseline_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict_info["verdict"],
        "primary_metric": primary_metric,
        "tolerance": tolerance,
        "mode": "quick" if quick else "full",
        "n_runs": len(new_metrics_list),
        "failed_runs": failed_runs,
        "per_metric": verdict_info["per_metric"],
        "environment_diffs": env_diffs,
        "current_environment": current_env,
    }

    return report


def save_regression_report(report: dict, output_dir: str = "experiments/regressions") -> Path:
    """Save regression report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = out_path / f"check-{date}.yaml"

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_regression_report(report: dict) -> str:
    """Format regression report as human-readable markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    verdict = report["verdict"]
    verdict_markers = {
        "pass": "PASS — No regression detected",
        "warning": "WARNING — Minor regression, investigate",
        "fail": "FAIL — REGRESSION DETECTED",
    }
    marker = verdict_markers.get(verdict, verdict)

    lines = [
        f"# Regression Check: {report.get('baseline_id', '?')}",
        "",
        f"**{marker}**",
        "",
        f"*Checked {report.get('checked_at', 'N/A')[:19]}*",
        f"*Mode: {report.get('mode', '?')}, Tolerance: {report.get('tolerance', 0) * 100:.1f}%*",
        "",
        "## Metric Comparison",
        "",
        f"| Metric | Baseline | Current | Delta | Rel Diff | Verdict |",
        f"|--------|----------|---------|-------|----------|---------|",
    ]

    per_metric = report.get("per_metric", {})
    for key, m in per_metric.items():
        orig = m.get("original", "N/A")
        new = m.get("new_mean", "N/A")
        delta = m.get("delta", 0)
        rel = m.get("relative_diff", 0)
        mv = m.get("verdict", "?").upper()

        orig_str = f"{orig:.4f}" if isinstance(orig, float) else str(orig)
        new_str = f"{new:.4f}" if isinstance(new, float) else str(new)

        lines.append(
            f"| {key} | {orig_str} | {new_str} | {delta:+.4f} | {rel * 100:.2f}% | {mv} |"
        )

    # Environment diffs
    env_diffs = report.get("environment_diffs", [])
    critical_env = [d for d in env_diffs if d.get("severity") in ("critical", "warning")]
    if critical_env:
        lines.extend(["", "## Environment Changes", ""])
        for d in critical_env:
            lines.append(f"- **[{d.get('severity', 'info').upper()}]** {d.get('detail', 'N/A')}")
        if verdict == "fail":
            lines.append("")
            lines.append("*Environment changes may explain the regression.*")

    # Run details
    n_runs = report.get("n_runs", 0)
    failed = report.get("failed_runs", 0)
    if n_runs > 1 or failed > 0:
        lines.extend([
            "",
            "## Run Details",
            "",
            f"- **Successful runs:** {n_runs}",
            f"- **Failed runs:** {failed}",
        ])
        for key, m in per_metric.items():
            if "new_std" in m:
                lines.append(f"- **{key} std:** {m['new_std']:.6f}")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Performance regression gate for ML experiments",
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help=f"Relative tolerance for regression (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--against",
        help="Specific experiment ID to check against (default: best)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 1 run instead of full seed study",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Number of regression runs (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--log", default=DEFAULT_LOG_PATH,
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

    report = regression_gate(
        tolerance=args.tolerance,
        against=args.against,
        quick=args.quick,
        n_runs=args.runs,
        config_path=args.config,
        log_path=args.log,
        timeout=args.timeout,
    )

    # Save report
    if "error" not in report:
        filepath = save_regression_report(report)
        print(f"\nSaved to {filepath}", file=sys.stderr)

    # Output
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_regression_report(report))

    # Exit code based on verdict
    if report.get("verdict") == "fail":
        sys.exit(1)
    elif report.get("verdict") == "warning":
        sys.exit(2)


if __name__ == "__main__":
    main()
