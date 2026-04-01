#!/usr/bin/env python3
"""Scaling law estimator for the autoresearch pipeline.

Runs experiments at different data/compute/model sizes, fits a power-law
curve, and predicts full-scale performance. Answers "is it worth training
on the full dataset?" before committing the compute.

Usage:
    python scripts/scaling_estimator.py --axis data
    python scripts/scaling_estimator.py --axis compute --points 4
    python scripts/scaling_estimator.py --analyze experiments/scaling/results.yaml
    python scripts/scaling_estimator.py --json
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
DEFAULT_SCALE_POINTS = [0.10, 0.25, 0.50, 0.75]
SCALE_AXES = {"data", "compute", "params"}


# --- Power Law Fitting ---


def fit_power_law(
    scales: list[float],
    values: list[float],
) -> dict:
    """Fit a power law: performance = a * scale^b + c.

    Uses log-linear regression on (log(scale), value) for the a*x^b part,
    then estimates c as the residual offset.

    Args:
        scales: Scale fractions (e.g., [0.1, 0.25, 0.5, 0.75]).
        values: Metric values at each scale.

    Returns:
        Dict with a, b, c coefficients, r_squared, and residuals.
    """
    if len(scales) < 2 or len(values) < 2:
        return {"a": 0, "b": 0, "c": 0, "r_squared": 0, "error": "Need at least 2 points"}

    x = np.array(scales, dtype=float)
    y = np.array(values, dtype=float)

    # Filter out non-positive scales
    mask = x > 0
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return {"a": 0, "b": 0, "c": 0, "r_squared": 0, "error": "Need at least 2 positive scale points"}

    # Log-linear regression: log(y - c) = log(a) + b * log(x)
    # First estimate without c (assume c=0)
    log_x = np.log(x)
    log_y = np.log(np.clip(y, 1e-10, None))

    # Linear regression in log space
    n = len(log_x)
    sum_lx = np.sum(log_x)
    sum_ly = np.sum(log_y)
    sum_lx2 = np.sum(log_x ** 2)
    sum_lxly = np.sum(log_x * log_y)

    denom = n * sum_lx2 - sum_lx ** 2
    if abs(denom) < 1e-12:
        b = 0.0
        log_a = np.mean(log_y)
    else:
        b = (n * sum_lxly - sum_lx * sum_ly) / denom
        log_a = (sum_ly - b * sum_lx) / n

    a = math.exp(log_a)
    c = 0.0  # Simplified: no offset term

    # Compute R²
    y_pred = a * x ** b + c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "a": round(float(a), 6),
        "b": round(float(b), 6),
        "c": round(float(c), 6),
        "r_squared": round(float(r_squared), 4),
        "residuals": [round(float(r), 6) for r in (y - y_pred)],
    }


def extrapolate(
    fit: dict,
    target_scales: list[float],
) -> list[dict]:
    """Extrapolate from fitted power law to target scales.

    Args:
        fit: Power law fit dict from fit_power_law.
        target_scales: Scale values to predict.

    Returns:
        List of prediction dicts with scale, predicted_value.
    """
    a = fit.get("a", 0)
    b = fit.get("b", 0)
    c = fit.get("c", 0)

    predictions = []
    for scale in target_scales:
        if scale <= 0:
            predictions.append({"scale": scale, "predicted_value": None})
            continue
        predicted = a * (scale ** b) + c
        predictions.append({
            "scale": round(scale, 4),
            "predicted_value": round(float(predicted), 6),
        })

    return predictions


# --- Scale Point Generation ---


def generate_scale_points(
    axis: str,
    fractions: list[float] | None = None,
    config: dict | None = None,
) -> list[dict]:
    """Generate experiment configurations for each scale point.

    Args:
        axis: Scaling axis (data, compute, params).
        fractions: Scale fractions (default: [0.1, 0.25, 0.5, 0.75]).
        config: Current model config.

    Returns:
        List of scale point dicts with fraction, description, config_overrides.
    """
    if fractions is None:
        fractions = DEFAULT_SCALE_POINTS

    if config is None:
        config = {}

    hyperparams = config.get("model", {}).get("hyperparams", {})
    points = []

    for frac in fractions:
        point = {
            "fraction": frac,
            "percentage": f"{frac * 100:.0f}%",
            "config_overrides": {},
        }

        if axis == "data":
            point["description"] = f"Train on {frac * 100:.0f}% of dataset"
            point["config_overrides"]["data_fraction"] = frac

        elif axis == "compute":
            max_epochs = hyperparams.get("n_estimators", hyperparams.get("epochs", 100))
            scaled_epochs = max(1, int(max_epochs * frac))
            point["description"] = f"Train for {scaled_epochs} epochs ({frac * 100:.0f}%)"
            point["config_overrides"]["n_estimators"] = scaled_epochs

        elif axis == "params":
            n_estimators = hyperparams.get("n_estimators", 100)
            max_depth = hyperparams.get("max_depth", 6)
            point["description"] = f"Model at {frac * 100:.0f}% capacity"
            point["config_overrides"]["n_estimators"] = max(1, int(n_estimators * frac))
            point["config_overrides"]["max_depth"] = max(1, int(max_depth * frac))

        points.append(point)

    return points


# --- Verdict ---


def compute_verdict(
    observed: list[dict],
    predictions: list[dict],
    primary_metric: str,
) -> dict:
    """Compute a verdict on whether scaling is worth it.

    Args:
        observed: List of {fraction, metric_value} from actual runs.
        predictions: Extrapolation predictions.
        primary_metric: Name of the primary metric.

    Returns:
        Verdict dict with recommendation and reasoning.
    """
    if not observed or not predictions:
        return {"verdict": "insufficient_data", "reason": "Not enough data points"}

    # Find the highest observed fraction and its value
    observed_sorted = sorted(observed, key=lambda x: x.get("fraction", 0))
    last_observed = observed_sorted[-1]
    last_fraction = last_observed.get("fraction", 0)
    last_value = last_observed.get("metric_value", 0)

    # Find full-scale prediction
    full_pred = None
    for p in predictions:
        if abs(p["scale"] - 1.0) < 0.01:
            full_pred = p
            break

    if not full_pred or full_pred["predicted_value"] is None:
        return {"verdict": "no_prediction", "reason": "Cannot predict full-scale performance"}

    predicted_gain = full_pred["predicted_value"] - last_value
    relative_gain = abs(predicted_gain / last_value) if last_value != 0 else 0

    if relative_gain < 0.005:  # < 0.5% improvement
        return {
            "verdict": "diminishing_returns",
            "predicted_gain": round(predicted_gain, 6),
            "relative_gain": round(relative_gain, 6),
            "reason": (
                f"Full-scale gains only {predicted_gain:+.4f} ({relative_gain:.1%}) "
                f"over {last_fraction:.0%} data. Consider feature engineering instead."
            ),
        }
    elif relative_gain < 0.02:  # < 2% improvement
        return {
            "verdict": "marginal_gains",
            "predicted_gain": round(predicted_gain, 6),
            "relative_gain": round(relative_gain, 6),
            "reason": (
                f"Full-scale gains {predicted_gain:+.4f} ({relative_gain:.1%}). "
                f"Worth running if compute is cheap."
            ),
        }
    else:
        return {
            "verdict": "worth_scaling",
            "predicted_gain": round(predicted_gain, 6),
            "relative_gain": round(relative_gain, 6),
            "reason": (
                f"Full-scale gains {predicted_gain:+.4f} ({relative_gain:.1%}). "
                f"Significant improvement expected — proceed with full-scale training."
            ),
        }


# --- Analysis ---


def analyze_scaling(
    scale_results: list[dict],
    primary_metric: str,
) -> dict:
    """Analyze completed scaling study results.

    Args:
        scale_results: List of {fraction, metric_value, std} dicts.
        primary_metric: Name of primary metric.

    Returns:
        Complete analysis report.
    """
    if not scale_results:
        return {"error": "No scaling results to analyze"}

    fractions = [r["fraction"] for r in scale_results]
    values = [r["metric_value"] for r in scale_results]

    # Fit power law
    fit = fit_power_law(fractions, values)

    # Extrapolate to 100% and 200%
    predictions = extrapolate(fit, [1.0, 1.5, 2.0])

    # Compute verdict
    verdict = compute_verdict(scale_results, predictions, primary_metric)

    return {
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": primary_metric,
        "scale_points": scale_results,
        "power_law_fit": fit,
        "predictions": predictions,
        "verdict": verdict,
    }


# --- Report Formatting ---


def save_scaling_report(report: dict, output_dir: str = "experiments/scaling") -> Path:
    """Save scaling report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = out_path / f"scale-{date}.yaml"

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_scaling_report(report: dict) -> str:
    """Format scaling report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    metric = report.get("primary_metric", "metric")
    fit = report.get("power_law_fit", {})

    lines = [
        "# Scaling Analysis",
        "",
        f"*Analyzed {report.get('analyzed_at', 'N/A')[:19]}*",
        "",
    ]

    # Scale points table
    points = report.get("scale_points", [])
    if points:
        lines.extend(["## Observed Scale Points", ""])
        has_std = any("std" in p for p in points)
        if has_std:
            lines.append(f"| Data % | {metric} (mean±std) |")
            lines.append("|--------|---------------------|")
            for p in points:
                std = p.get("std", 0)
                lines.append(f"| {p['fraction'] * 100:.0f}% | {p['metric_value']:.4f} ± {std:.4f} |")
        else:
            lines.append(f"| Data % | {metric} |")
            lines.append("|--------|---------|")
            for p in points:
                lines.append(f"| {p['fraction'] * 100:.0f}% | {p['metric_value']:.4f} |")
        lines.append("")

    # Power law fit
    if fit and "error" not in fit:
        lines.extend([
            "## Power Law Fit",
            "",
            f"**{metric} = {fit['a']:.4f} × n^{fit['b']:.4f}** (R²={fit['r_squared']:.4f})",
            "",
        ])

    # Predictions
    predictions = report.get("predictions", [])
    if predictions:
        lines.extend(["## Predictions", ""])
        for p in predictions:
            if p["predicted_value"] is not None:
                lines.append(f"- **{p['scale'] * 100:.0f}% data** → {metric} = {p['predicted_value']:.4f}")
        lines.append("")

    # Verdict
    verdict = report.get("verdict", {})
    if verdict:
        v = verdict.get("verdict", "?")
        verdict_labels = {
            "diminishing_returns": "DIMINISHING RETURNS",
            "marginal_gains": "MARGINAL GAINS",
            "worth_scaling": "WORTH SCALING",
        }
        lines.extend([
            "## Verdict",
            "",
            f"**{verdict_labels.get(v, v.upper())}**",
            "",
            verdict.get("reason", ""),
            "",
        ])

    return "\n".join(lines)


def format_ascii_plot(
    scale_results: list[dict],
    predictions: list[dict],
    metric: str,
    width: int = 50,
    height: int = 15,
) -> str:
    """Generate an ASCII scatter plot of the scaling curve."""
    all_points = []
    for r in scale_results:
        all_points.append((r["fraction"], r["metric_value"], "o"))
    for p in predictions:
        if p["predicted_value"] is not None:
            all_points.append((p["scale"], p["predicted_value"], "*"))

    if not all_points:
        return "(no data to plot)"

    x_vals = [p[0] for p in all_points]
    y_vals = [p[1] for p in all_points]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 0.01

    grid = [[" " for _ in range(width)] for _ in range(height)]

    for x, y, marker in all_points:
        col = int((x - x_min) / (x_max - x_min) * (width - 1))
        row = height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = marker

    lines = [f"  {metric} vs Scale (o=observed, *=predicted)", ""]
    lines.append(f"  {y_max:.3f} |")
    for row in grid:
        lines.append(f"          |{''.join(row)}|")
    lines.append(f"  {y_min:.3f} |{'_' * width}|")
    lines.append(f"           {x_min:.0%}{' ' * (width - 8)}{x_max:.0%}")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scaling law estimator",
    )
    parser.add_argument(
        "--axis", choices=list(SCALE_AXES), default="data",
        help="Scaling axis (default: data)",
    )
    parser.add_argument(
        "--points", type=int, default=len(DEFAULT_SCALE_POINTS),
        help=f"Number of scale points (default: {len(DEFAULT_SCALE_POINTS)})",
    )
    parser.add_argument(
        "--analyze",
        help="Analyze existing scaling results YAML",
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
        "--plot", action="store_true",
        help="Include ASCII plot",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    if args.analyze:
        # Analyze existing results
        with open(args.analyze) as f:
            data = yaml.safe_load(f)
        config = load_config(args.config)
        metric = config.get("evaluation", {}).get("primary_metric", "accuracy")
        report = analyze_scaling(data.get("scale_points", data), metric)
    else:
        # Generate scale points (actual execution would be done by the agent)
        config = load_config(args.config)
        metric = config.get("evaluation", {}).get("primary_metric", "accuracy")
        fractions = DEFAULT_SCALE_POINTS[:args.points]
        points = generate_scale_points(args.axis, fractions, config)
        report = {
            "action": "plan",
            "axis": args.axis,
            "primary_metric": metric,
            "scale_points": points,
            "message": f"Run {len(points)} experiments at scale points: {', '.join(p['percentage'] for p in points)}",
        }

    if "error" not in report:
        filepath = save_scaling_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        if report.get("action") == "plan":
            lines = ["# Scaling Plan", "", f"**Axis:** {report['axis']}", ""]
            for p in report["scale_points"]:
                lines.append(f"- {p['percentage']}: {p['description']}")
            lines.append("")
            lines.append(report["message"])
            print("\n".join(lines))
        else:
            text = format_scaling_report(report)
            if args.plot:
                text += "\n\n" + format_ascii_plot(
                    report.get("scale_points", []),
                    report.get("predictions", []),
                    metric,
                )
            print(text)


if __name__ == "__main__":
    main()
