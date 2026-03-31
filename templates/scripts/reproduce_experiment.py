#!/usr/bin/env python3
"""Reproducibility verification for ML experiments.

Given an experiment ID, re-runs it from the logged config and verifies
metrics fall within tolerance of the original. Catches non-determinism,
environment drift, and silent data changes.

Verdicts:
    reproducible              — metrics match within float tolerance (1e-6)
    approximately_reproducible — metrics within user-specified tolerance
    not_reproducible          — metrics outside tolerance/CI
    environment_changed       — different library versions detected

Usage:
    python scripts/reproduce_experiment.py exp-042
    python scripts/reproduce_experiment.py exp-042 --tolerance 0.02
    python scripts/reproduce_experiment.py exp-042 --strict
    python scripts/reproduce_experiment.py exp-042 --runs 5
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

FLOAT_TOLERANCE = 1e-6
DEFAULT_TOLERANCE = 0.02  # 2% relative
DEFAULT_REPRO_RUNS = 3


def find_experiment(experiments: list[dict], exp_id: str) -> dict | None:
    """Find an experiment by ID in the log."""
    for exp in experiments:
        if exp.get("experiment_id") == exp_id:
            return exp
    return None


def capture_current_environment() -> dict:
    """Capture the current Python environment for comparison."""
    env = {}

    # Python version
    env["python_version"] = sys.version.split()[0]

    # Installed packages via pip freeze
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

    return env


def compare_environments(original_env: dict | None, current_env: dict) -> list[dict]:
    """Compare original experiment environment against current.

    Returns list of diffs with severity (info, warning, critical).
    """
    if not original_env:
        return [{"field": "environment", "severity": "info",
                 "detail": "No environment snapshot in original experiment"}]

    diffs = []

    # Python version
    orig_py = original_env.get("python_version", "unknown")
    curr_py = current_env.get("python_version", "unknown")
    if orig_py != curr_py:
        diffs.append({
            "field": "python_version",
            "original": orig_py,
            "current": curr_py,
            "severity": "warning",
            "detail": f"Python version changed: {orig_py} -> {curr_py}",
        })

    # Package versions
    orig_pkgs = original_env.get("packages", {})
    curr_pkgs = current_env.get("packages", {})

    # Key ML packages that affect reproducibility
    critical_packages = {
        "numpy", "scipy", "scikit-learn", "sklearn", "pandas",
        "torch", "tensorflow", "xgboost", "lightgbm", "catboost",
    }

    all_pkgs = set(orig_pkgs) | set(curr_pkgs)
    for pkg in sorted(all_pkgs):
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
        elif orig_ver and not curr_ver:
            severity = "warning" if pkg in critical_packages else "info"
            diffs.append({
                "field": f"package:{pkg}",
                "original": orig_ver,
                "current": "missing",
                "severity": severity,
                "detail": f"{pkg} {orig_ver} was present but is now missing",
            })

    return diffs


def run_single_reproduction(seed: int, timeout: int = 600) -> dict | None:
    """Run train.py with given seed and return parsed metrics."""
    cmd = ["python", "train.py", "--seed", str(seed)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None

    if proc.returncode != 0:
        return None

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


def determine_verdict(
    original_value: float,
    new_values: list[float],
    tolerance: float,
    strict: bool,
) -> dict:
    """Determine reproducibility verdict.

    Args:
        original_value: Metric from the original experiment.
        new_values: Metrics from reproduction run(s).
        tolerance: Relative tolerance for approximate match.
        strict: If True, require exact match within float tolerance.

    Returns:
        Dict with verdict, reason, and statistical details.
    """
    arr = np.array(new_values)
    new_mean = float(np.mean(arr))
    n = len(arr)

    result = {
        "original_value": round(original_value, 6),
        "new_mean": round(new_mean, 6),
        "new_values": [round(v, 6) for v in new_values],
        "n_runs": n,
    }

    if n > 1:
        new_std = float(np.std(arr, ddof=1))
        from scipy import stats as scipy_stats
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * new_std / np.sqrt(n)
        ci_lower = new_mean - margin
        ci_upper = new_mean + margin
        result["new_std"] = round(new_std, 6)
        result["ci_95"] = [round(ci_lower, 6), round(ci_upper, 6)]
    else:
        ci_lower = new_values[0]
        ci_upper = new_values[0]
        result["new_std"] = 0.0
        result["ci_95"] = [round(ci_lower, 6), round(ci_upper, 6)]

    # Check exact match (deterministic case)
    if strict or n == 1:
        diff = abs(original_value - new_mean)
        if diff < FLOAT_TOLERANCE:
            result["verdict"] = "reproducible"
            result["reason"] = f"Exact match within float tolerance ({FLOAT_TOLERANCE})"
            return result
        if strict:
            result["verdict"] = "not_reproducible"
            result["reason"] = f"Strict mode: difference {diff:.6f} exceeds float tolerance {FLOAT_TOLERANCE}"
            return result

    # Check approximate match (within tolerance)
    relative_diff = abs(original_value - new_mean) / abs(original_value) if original_value != 0 else abs(new_mean)
    result["relative_difference"] = round(relative_diff, 6)

    if relative_diff <= tolerance:
        result["verdict"] = "approximately_reproducible"
        result["reason"] = (
            f"Within {tolerance*100:.1f}% tolerance "
            f"(actual difference: {relative_diff*100:.2f}%)"
        )
        return result

    # Check if original falls within CI of new distribution
    if n > 1 and ci_lower <= original_value <= ci_upper:
        result["verdict"] = "approximately_reproducible"
        result["reason"] = (
            f"Original value {original_value:.4f} falls within 95% CI "
            f"[{ci_lower:.4f}, {ci_upper:.4f}] of reproduction distribution"
        )
        return result

    # Not reproducible
    result["verdict"] = "not_reproducible"
    result["reason"] = (
        f"Difference {relative_diff*100:.2f}% exceeds {tolerance*100:.1f}% tolerance, "
        f"and original {original_value:.4f} outside 95% CI "
        f"[{ci_lower:.4f}, {ci_upper:.4f}]"
    )
    return result


def reproduce_experiment(
    exp_id: str,
    tolerance: float = DEFAULT_TOLERANCE,
    strict: bool = False,
    n_runs: int = DEFAULT_REPRO_RUNS,
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
    timeout: int = 600,
) -> dict:
    """Run a complete reproducibility verification.

    Args:
        exp_id: Experiment ID to reproduce.
        tolerance: Relative tolerance for approximate match.
        strict: Require exact match (overrides tolerance).
        n_runs: Number of reproduction runs (1 for strict/deterministic).
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        timeout: Per-run timeout in seconds.

    Returns:
        Complete reproduction report dict.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")

    experiments = load_experiments(log_path)
    original = find_experiment(experiments, exp_id)

    if not original:
        return {
            "error": f"Experiment {exp_id} not found in {log_path}",
            "experiment_id": exp_id,
        }

    original_metrics = original.get("metrics", {})
    original_value = original_metrics.get(primary_metric)
    if original_value is None:
        return {
            "error": f"Experiment {exp_id} has no {primary_metric} metric",
            "experiment_id": exp_id,
        }

    # Extract seed from original experiment
    original_seed = original.get("config", {}).get("hyperparams", {}).get("seed", 42)
    # Also check environment for seed
    if original.get("environment", {}).get("seed") is not None:
        original_seed = original["environment"]["seed"]

    # If strict, run once with same seed
    if strict:
        n_runs = 1

    print(f"Reproducing {exp_id}", file=sys.stderr)
    print(f"Original {primary_metric}: {original_value:.4f}", file=sys.stderr)
    print(f"Mode: {'strict (exact match)' if strict else f'tolerance={tolerance*100:.1f}%'}", file=sys.stderr)
    print(f"Runs: {n_runs} (seed={original_seed})", file=sys.stderr)
    print(file=sys.stderr)

    # Capture current environment
    current_env = capture_current_environment()
    original_env = original.get("environment")
    env_diffs = compare_environments(original_env, current_env)

    # Run reproductions
    new_values = []
    failed_runs = 0
    for i in range(n_runs):
        seed = original_seed if strict else original_seed + i
        print(f"  Run {i + 1}/{n_runs} (seed={seed})...", end=" ", flush=True, file=sys.stderr)
        metrics = run_single_reproduction(seed, timeout=timeout)
        if metrics and primary_metric in metrics:
            val = metrics[primary_metric]
            new_values.append(val)
            print(f"{primary_metric}={val:.4f}", file=sys.stderr)
        else:
            failed_runs += 1
            print("FAILED", file=sys.stderr)

    if not new_values:
        return {
            "error": f"All {n_runs} reproduction runs failed",
            "experiment_id": exp_id,
        }

    # Determine verdict
    verdict_info = determine_verdict(original_value, new_values, tolerance, strict)

    # Check for environment changes
    has_env_changes = any(d["severity"] in ("warning", "critical") for d in env_diffs)
    if has_env_changes and verdict_info["verdict"] == "not_reproducible":
        verdict_info["verdict"] = "environment_changed"
        verdict_info["reason"] += " (environment differences detected — this may be the cause)"

    # Build full report
    report = {
        "experiment_id": exp_id,
        "reproduced_at": datetime.now(timezone.utc).isoformat(),
        "original_metrics": {primary_metric: round(original_value, 6)},
        "original_config": original.get("config", {}),
        "original_git_commit": original.get("git_commit"),
        "new_metrics": {primary_metric: verdict_info["new_mean"]},
        "verdict": verdict_info["verdict"],
        "reason": verdict_info["reason"],
        "statistical_details": {
            k: v for k, v in verdict_info.items()
            if k not in ("verdict", "reason")
        },
        "strictness": f"exact (1e-6)" if strict else f"tolerance={tolerance}",
        "n_runs": len(new_values),
        "failed_runs": failed_runs,
        "environment_changed": has_env_changes,
        "environment_diffs": env_diffs if env_diffs else [],
    }

    return report


def save_reproduction_report(report: dict, output_dir: str = "experiments/reproductions") -> Path:
    """Save reproduction report to YAML file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exp_id = report.get("experiment_id", "unknown")
    filename = f"{exp_id}-repro.yaml"
    filepath = out_path / filename

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_reproduction_report(report: dict) -> str:
    """Format reproduction report as human-readable markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    exp_id = report["experiment_id"]
    verdict = report["verdict"]
    details = report.get("statistical_details", {})

    # Verdict emoji/marker
    verdict_markers = {
        "reproducible": "PASS",
        "approximately_reproducible": "PASS (approx)",
        "not_reproducible": "FAIL",
        "environment_changed": "WARN",
    }
    marker = verdict_markers.get(verdict, verdict)

    lines = [
        f"# Reproducibility Report: {exp_id}",
        "",
        f"**Verdict: {marker}**",
        "",
        f"*{report['reason']}*",
        "",
        "## Comparison",
        "",
        "| Metric | Original | Reproduced |",
        "|--------|----------|------------|",
    ]

    for metric, orig_val in report.get("original_metrics", {}).items():
        new_val = report.get("new_metrics", {}).get(metric, "N/A")
        if isinstance(orig_val, float) and isinstance(new_val, float):
            lines.append(f"| {metric} | {orig_val:.4f} | {new_val:.4f} |")
        else:
            lines.append(f"| {metric} | {orig_val} | {new_val} |")

    if details.get("n_runs", 0) > 1:
        lines.extend([
            "",
            "## Statistical Details",
            "",
            f"- **Reproduction runs:** {details.get('n_runs', 'N/A')}",
            f"- **New values:** {details.get('new_values', [])}",
        ])
        if "new_std" in details:
            lines.append(f"- **New std:** {details['new_std']:.6f}")
        if "ci_95" in details:
            ci = details["ci_95"]
            lines.append(f"- **95% CI:** [{ci[0]:.4f}, {ci[1]:.4f}]")
        if "relative_difference" in details:
            lines.append(f"- **Relative difference:** {details['relative_difference']*100:.2f}%")

    # Environment section
    env_diffs = report.get("environment_diffs", [])
    if env_diffs:
        lines.extend([
            "",
            "## Environment",
            "",
        ])
        has_changes = False
        for diff in env_diffs:
            if diff["severity"] == "info" and "No environment snapshot" in diff.get("detail", ""):
                lines.append(f"- {diff['detail']}")
            elif diff["severity"] != "info":
                has_changes = True
                sev = diff["severity"].upper()
                lines.append(f"- **[{sev}]** {diff['detail']}")

        if not has_changes and not any("No environment" in d.get("detail", "") for d in env_diffs):
            lines.append("All packages match original environment.")
    else:
        lines.extend([
            "",
            "## Environment",
            "",
            "All packages match original environment.",
        ])

    lines.extend([
        "",
        f"*Strictness: {report.get('strictness', 'N/A')}*",
    ])

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reproducibility verification for ML experiments",
    )
    parser.add_argument(
        "exp_id",
        help="Experiment ID to reproduce (e.g., exp-042)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help=f"Relative tolerance for approximate match (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Strict mode: require exact match within float tolerance (1e-6)",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_REPRO_RUNS,
        help=f"Number of reproduction runs (default: {DEFAULT_REPRO_RUNS})",
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

    report = reproduce_experiment(
        exp_id=args.exp_id,
        tolerance=args.tolerance,
        strict=args.strict,
        n_runs=args.runs,
        config_path=args.config,
        log_path=args.log,
        timeout=args.timeout,
    )

    # Save report
    if "error" not in report:
        filepath = save_reproduction_report(report)
        print(f"\nSaved to {filepath}", file=sys.stderr)

    # Output
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_reproduction_report(report))

    # Exit code based on verdict
    if report.get("verdict") == "not_reproducible":
        sys.exit(1)
    elif report.get("verdict") == "environment_changed":
        sys.exit(2)


if __name__ == "__main__":
    main()
