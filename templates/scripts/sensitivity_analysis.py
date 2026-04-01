#!/usr/bin/env python3
"""Hyperparameter sensitivity analysis for the autoresearch pipeline.

Varies each hyperparameter individually while holding others fixed,
measures the metric response, and ranks hyperparameters by sensitivity.
Answers "which hyperparameters actually matter?"

Usage:
    python scripts/sensitivity_analysis.py exp-042
    python scripts/sensitivity_analysis.py --params "learning_rate,max_depth"
    python scripts/sensitivity_analysis.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_N_POINTS = 5
SENSITIVITY_THRESHOLDS = {"HIGH": 0.02, "MED": 0.005, "LOW": 0.002}
DEFAULT_MULTIPLIERS = [0.5, 0.75, 1.0, 1.5, 2.0]


# --- Sweep Generation ---


def generate_sweep(
    param_name: str,
    current_value: float | int,
    n_points: int = DEFAULT_N_POINTS,
    multipliers: list[float] | None = None,
) -> list[dict]:
    """Generate sweep values for a hyperparameter.

    Returns list of {value, multiplier} dicts.
    """
    if multipliers is None:
        multipliers = DEFAULT_MULTIPLIERS[:n_points]

    points = []
    for m in multipliers:
        if isinstance(current_value, int):
            val = max(1, int(current_value * m))
        else:
            val = current_value * m
        points.append({
            "value": val,
            "multiplier": round(m, 2),
            "is_current": abs(m - 1.0) < 0.01,
        })

    return points


def extract_tunable_params(config: dict) -> dict:
    """Extract tunable hyperparameters from config."""
    hyperparams = config.get("model", {}).get("hyperparams", {})

    tunable = {}
    for key, val in hyperparams.items():
        if isinstance(val, (int, float)) and key not in ("seed", "random_state", "verbose"):
            tunable[key] = val

    return tunable


# --- Sensitivity Scoring ---


def compute_sensitivity(
    param_name: str,
    sweep_results: list[dict],
    primary_metric: str,
) -> dict:
    """Compute sensitivity score for a hyperparameter.

    Args:
        param_name: Hyperparameter name.
        sweep_results: List of {value, metric_value} dicts.
        primary_metric: Name of the primary metric.

    Returns:
        Sensitivity dict with score, level, range, best value, monotonicity.
    """
    if not sweep_results or len(sweep_results) < 2:
        return {"param": param_name, "sensitivity": 0, "level": "NONE",
                "reason": "Insufficient sweep data"}

    values = [r.get("value") for r in sweep_results]
    metrics = [r.get("metric_value") for r in sweep_results
               if r.get("metric_value") is not None]

    if len(metrics) < 2:
        return {"param": param_name, "sensitivity": 0, "level": "NONE",
                "reason": "Insufficient metric data"}

    metric_range = max(metrics) - min(metrics)
    metric_mean = np.mean(metrics)

    # Normalized sensitivity
    sensitivity = metric_range / abs(metric_mean) if metric_mean != 0 else metric_range

    # Classify level
    if sensitivity > SENSITIVITY_THRESHOLDS["HIGH"]:
        level = "HIGH"
    elif sensitivity > SENSITIVITY_THRESHOLDS["MED"]:
        level = "MED"
    elif sensitivity > SENSITIVITY_THRESHOLDS["LOW"]:
        level = "LOW"
    else:
        level = "NONE"

    # Check monotonicity
    monotonic = _check_monotonicity(metrics)

    # Best value
    best_idx = np.argmax(metrics)
    best_value = values[best_idx] if best_idx < len(values) else None

    return {
        "param": param_name,
        "current_value": next((r["value"] for r in sweep_results if r.get("is_current")), None),
        "sensitivity": round(float(sensitivity), 6),
        "metric_range": round(float(metric_range), 6),
        "metric_min": round(float(min(metrics)), 6),
        "metric_max": round(float(max(metrics)), 6),
        "level": level,
        "best_value": best_value,
        "monotonic": monotonic,
    }


def _check_monotonicity(values: list[float]) -> str:
    """Check if values are monotonically increasing, decreasing, or non-monotonic."""
    if len(values) < 2:
        return "unknown"

    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    all_pos = all(d >= 0 for d in diffs)
    all_neg = all(d <= 0 for d in diffs)

    if all_pos:
        return "increasing"
    elif all_neg:
        return "decreasing"
    else:
        return "non_monotonic"


def rank_sensitivities(sensitivities: list[dict]) -> list[dict]:
    """Rank parameters by sensitivity (highest first)."""
    return sorted(sensitivities, key=lambda s: s.get("sensitivity", 0), reverse=True)


# --- Recommendations ---


def generate_recommendations(ranked: list[dict]) -> list[str]:
    """Generate tuning recommendations from sensitivity ranking."""
    recs = []

    high = [s for s in ranked if s["level"] == "HIGH"]
    none = [s for s in ranked if s["level"] == "NONE"]

    if high:
        names = ", ".join(s["param"] for s in high)
        recs.append(f"Focus tuning on {names}")

    if none:
        names = ", ".join(s["param"] for s in none)
        recs.append(f"Stop tuning {names} — they don't matter for this model")

    non_mono = [s for s in ranked if s.get("monotonic") == "non_monotonic" and s["level"] in ("HIGH", "MED")]
    if non_mono:
        for s in non_mono:
            recs.append(f"{s['param']} has a non-monotonic relationship — there's an optimal sweet spot around {s.get('best_value')}")

    return recs


# --- Full Pipeline ---


def sensitivity_analysis(
    exp_id: str | None = None,
    params: list[str] | None = None,
    sweep_data: dict[str, list[dict]] | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Run sensitivity analysis.

    Args:
        exp_id: Experiment ID to analyze.
        params: Specific parameters to analyze.
        sweep_data: Pre-computed sweep results {param: [{value, metric_value}]}.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.

    Returns:
        Sensitivity analysis report.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")

    sensitivities = []

    if sweep_data:
        # Analyze pre-computed sweep data
        for param, results in sweep_data.items():
            sens = compute_sensitivity(param, results, primary_metric)
            sensitivities.append(sens)
    else:
        # Generate sweep plan (actual execution done by agent)
        tunable = extract_tunable_params(config)
        if params:
            tunable = {k: v for k, v in tunable.items() if k in params}

        if not tunable:
            return {"error": "No tunable hyperparameters found in config"}

        sweep_plans = {}
        for param, value in tunable.items():
            sweep_plans[param] = generate_sweep(param, value)

        return {
            "action": "plan",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "primary_metric": primary_metric,
            "experiment_id": exp_id,
            "sweep_plans": sweep_plans,
            "n_experiments_needed": sum(len(s) for s in sweep_plans.values()),
            "message": f"Sweep {len(sweep_plans)} parameters × {DEFAULT_N_POINTS} values each",
        }

    ranked = rank_sensitivities(sensitivities)
    recommendations = generate_recommendations(ranked)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": primary_metric,
        "experiment_id": exp_id,
        "sensitivities": ranked,
        "recommendations": recommendations,
    }


# --- Report Formatting ---


def save_sensitivity_report(report: dict, output_dir: str = "experiments/sensitivity") -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("experiment_id", "unknown")
    filepath = out_path / f"{exp_id}-sensitivity.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_sensitivity_report(report: dict) -> str:
    if "error" in report:
        return f"ERROR: {report['error']}"

    if report.get("action") == "plan":
        plans = report.get("sweep_plans", {})
        lines = ["# Sensitivity Analysis Plan", "",
                 f"**{report.get('n_experiments_needed', 0)} experiments** needed for {len(plans)} parameters", ""]
        for param, points in plans.items():
            vals = ", ".join(str(p["value"]) for p in points)
            lines.append(f"- **{param}:** [{vals}]")
        return "\n".join(lines)

    metric = report.get("primary_metric", "metric")
    exp_id = report.get("experiment_id", "?")

    lines = [f"# Hyperparameter Sensitivity Analysis ({exp_id})", "",
             f"*Generated {report.get('generated_at', 'N/A')[:19]}*", "",
             f"| Parameter | Current | Range Tested | {metric} Range | Sensitivity |",
             "|-----------|---------|-------------|----------------|-------------|"]

    for s in report.get("sensitivities", []):
        current = s.get("current_value", "?")
        metric_range = f"{s['metric_min']:.4f}–{s['metric_max']:.4f}" if s.get("metric_min") is not None else "N/A"
        sens = f"{s['level']} ({s['sensitivity']:.4f})"
        lines.append(f"| {s['param']} | {current} | — | {metric_range} | {sens} |")

    recs = report.get("recommendations", [])
    if recs:
        lines.extend(["", "## Recommendations", ""])
        for r in recs:
            lines.append(f"- {r}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sensitivity analysis")
    parser.add_argument("exp_id", nargs="?", help="Experiment ID")
    parser.add_argument("--params", help="Comma-separated parameter names")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    params = [p.strip() for p in args.params.split(",")] if args.params else None

    report = sensitivity_analysis(
        exp_id=args.exp_id, params=params,
        config_path=args.config, log_path=args.log,
    )

    if "error" not in report:
        filepath = save_sensitivity_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_sensitivity_report(report))


if __name__ == "__main__":
    main()
