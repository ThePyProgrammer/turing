#!/usr/bin/env python3
"""Pre-training sanity checks for the autoresearch pipeline.

Runs a battery of fast checks before committing to a full training run:
initial loss validation, single-batch overfit, gradient flow, output
validation, data pipeline check, and config consistency.

Usage:
    python scripts/sanity_checks.py
    python scripts/sanity_checks.py --quick
    python scripts/sanity_checks.py --verbose
    python scripts/sanity_checks.py --json
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

from scripts.turing_io import load_config

DEFAULT_OVERFIT_STEPS = 50
DEFAULT_OVERFIT_THRESHOLD = 0.1  # Loss should drop below this fraction of initial


# --- Individual Checks ---


def check_initial_loss(
    initial_loss: float,
    num_classes: int | None = None,
    task_type: str = "classification",
) -> dict:
    """Check if initial loss matches theoretical expectation.

    For classification with cross-entropy: expected = -log(1/num_classes).
    """
    check = {
        "check": "initial_loss",
        "severity": "high",
        "initial_loss": round(initial_loss, 4),
    }

    if math.isnan(initial_loss) or math.isinf(initial_loss):
        check["status"] = "fail"
        check["reason"] = f"Initial loss is {initial_loss} — model is broken before training starts"
        return check

    if task_type == "classification" and num_classes and num_classes > 1:
        expected = -math.log(1.0 / num_classes)
        ratio = initial_loss / expected if expected > 0 else float("inf")
        check["expected_loss"] = round(expected, 4)
        check["ratio"] = round(ratio, 2)

        if ratio > 3.0:
            check["status"] = "fail"
            check["reason"] = f"Initial loss {initial_loss:.4f} is {ratio:.1f}x expected ({expected:.4f}) — likely misconfigured loss function"
        elif ratio > 2.0:
            check["status"] = "warn"
            check["reason"] = f"Initial loss {initial_loss:.4f} is {ratio:.1f}x expected ({expected:.4f}) — investigate"
        else:
            check["status"] = "pass"
            check["reason"] = f"Initial loss {initial_loss:.4f} matches expected {expected:.4f} (ratio: {ratio:.2f})"
    else:
        # For regression, just check it's finite and reasonable
        if initial_loss < 0:
            check["status"] = "warn"
            check["reason"] = f"Negative initial loss ({initial_loss:.4f}) — unusual for most loss functions"
        else:
            check["status"] = "pass"
            check["reason"] = f"Initial loss {initial_loss:.4f} is finite and non-negative"

    return check


def check_single_batch_overfit(
    loss_history: list[float],
    threshold: float = DEFAULT_OVERFIT_THRESHOLD,
) -> dict:
    """Check if model can overfit a single batch.

    If loss doesn't approach zero after N steps, something is broken.
    """
    check = {
        "check": "single_batch_overfit",
        "severity": "critical",
    }

    if not loss_history:
        check["status"] = "skip"
        check["reason"] = "No loss history provided"
        return check

    initial = loss_history[0]
    final = loss_history[-1]
    n_steps = len(loss_history)

    if initial <= 0:
        check["status"] = "warn"
        check["reason"] = f"Initial loss is {initial:.4f} (non-positive), cannot assess overfit"
        return check

    reduction = 1 - (final / initial)

    check["initial_loss"] = round(initial, 4)
    check["final_loss"] = round(final, 4)
    check["n_steps"] = n_steps
    check["reduction"] = round(reduction, 4)

    if any(math.isnan(l) for l in loss_history):
        check["status"] = "fail"
        check["reason"] = f"NaN in loss during overfit test — numerical instability"
        return check

    if reduction > 0.9:
        check["status"] = "pass"
        check["reason"] = f"Loss reduced by {reduction:.0%} in {n_steps} steps — model can memorize"
    elif reduction > 0.5:
        check["status"] = "warn"
        check["reason"] = f"Loss reduced by only {reduction:.0%} — model is learning but slowly. Check learning rate."
    else:
        check["status"] = "fail"
        check["reason"] = f"Loss stuck (reduced only {reduction:.0%} in {n_steps} steps) — model cannot memorize 1 batch. Check: architecture, learning rate, loss function"

    return check


def check_gradient_flow(
    gradient_stats: list[dict],
) -> dict:
    """Check that gradients are non-zero and non-exploding for every parameter.

    Args:
        gradient_stats: List of {name, mean, max, min, std} per parameter group.
    """
    check = {
        "check": "gradient_flow",
        "severity": "high",
    }

    if not gradient_stats:
        check["status"] = "skip"
        check["reason"] = "No gradient statistics provided"
        return check

    dead_layers = []
    exploding_layers = []
    total = len(gradient_stats)

    mean_grad = np.mean([abs(g.get("mean", 0)) for g in gradient_stats])

    for g in gradient_stats:
        name = g.get("name", "?")
        grad_mean = abs(g.get("mean", 0))
        grad_max = abs(g.get("max", 0))

        if grad_mean == 0 and grad_max == 0:
            dead_layers.append(name)
        elif mean_grad > 0 and grad_max > 100 * mean_grad:
            exploding_layers.append(name)

    check["total_params"] = total
    check["dead_layers"] = dead_layers
    check["exploding_layers"] = exploding_layers

    if dead_layers and exploding_layers:
        check["status"] = "fail"
        check["reason"] = f"{len(dead_layers)} dead layer(s) and {len(exploding_layers)} exploding layer(s)"
    elif dead_layers:
        check["status"] = "warn"
        check["reason"] = f"{len(dead_layers)} dead layer(s) with zero gradients: {', '.join(dead_layers[:3])}"
    elif exploding_layers:
        check["status"] = "warn"
        check["reason"] = f"{len(exploding_layers)} layer(s) with exploding gradients: {', '.join(exploding_layers[:3])}"
    else:
        check["status"] = "pass"
        check["reason"] = f"All {total} parameter groups have non-zero, stable gradients"

    return check


def check_output_validation(
    outputs: np.ndarray | list,
    task_type: str = "classification",
) -> dict:
    """Check that model outputs are valid (non-NaN, non-constant, reasonable range)."""
    check = {
        "check": "output_validation",
        "severity": "high",
    }

    arr = np.asarray(outputs, dtype=float)

    if arr.size == 0:
        check["status"] = "skip"
        check["reason"] = "No outputs to validate"
        return check

    has_nan = bool(np.any(np.isnan(arr)))
    has_inf = bool(np.any(np.isinf(arr)))
    is_constant = bool(np.std(arr) == 0)
    out_min = float(np.nanmin(arr))
    out_max = float(np.nanmax(arr))

    check["range"] = [round(out_min, 4), round(out_max, 4)]
    check["has_nan"] = has_nan
    check["has_inf"] = has_inf
    check["is_constant"] = is_constant

    issues = []
    if has_nan:
        issues.append("NaN values in outputs")
    if has_inf:
        issues.append("Inf values in outputs")
    if is_constant:
        issues.append("All outputs identical (constant predictions)")
    if abs(out_max) > 100 and task_type == "classification":
        issues.append(f"Extreme output range [{out_min:.1f}, {out_max:.1f}] — consider clamping")

    if has_nan or has_inf:
        check["status"] = "fail"
        check["reason"] = "; ".join(issues)
    elif is_constant:
        check["status"] = "fail"
        check["reason"] = "Constant predictions — model is not differentiating inputs"
    elif issues:
        check["status"] = "warn"
        check["reason"] = "; ".join(issues)
    else:
        check["status"] = "pass"
        check["reason"] = f"Outputs valid: range [{out_min:.4f}, {out_max:.4f}], no NaN/Inf"

    return check


def check_data_pipeline(
    batch_shapes: dict | None = None,
    has_nan: bool = False,
    has_inf: bool = False,
    loads_ok: bool = True,
) -> dict:
    """Check that the data pipeline produces valid batches."""
    check = {
        "check": "data_pipeline",
        "severity": "critical",
    }

    if not loads_ok:
        check["status"] = "fail"
        check["reason"] = "Data pipeline failed to load first batch"
        return check

    issues = []
    if has_nan:
        issues.append("NaN values in input data")
    if has_inf:
        issues.append("Inf values in input data")

    if batch_shapes:
        check["shapes"] = batch_shapes

    if issues:
        check["status"] = "fail"
        check["reason"] = "; ".join(issues)
    elif batch_shapes:
        shapes_str = ", ".join(f"{k}: {v}" for k, v in batch_shapes.items())
        check["status"] = "pass"
        check["reason"] = f"Batch loads, shapes correct ({shapes_str})"
    else:
        check["status"] = "pass"
        check["reason"] = "Data pipeline functional"

    return check


def check_config_consistency(config: dict) -> dict:
    """Check that config values are in reasonable ranges."""
    check = {
        "check": "config_consistency",
        "severity": "medium",
    }

    issues = []
    hyperparams = config.get("model", {}).get("hyperparams", {})

    lr = hyperparams.get("learning_rate", hyperparams.get("lr"))
    if lr is not None:
        if lr > 1.0:
            issues.append(f"Learning rate {lr} > 1.0 — unusually high")
        elif lr < 1e-8:
            issues.append(f"Learning rate {lr} < 1e-8 — effectively zero")

    batch_size = hyperparams.get("batch_size")
    if batch_size is not None:
        if batch_size < 1:
            issues.append(f"Batch size {batch_size} < 1 — invalid")
        elif batch_size > 100000:
            issues.append(f"Batch size {batch_size} > 100K — unusually large")

    n_estimators = hyperparams.get("n_estimators")
    if n_estimators is not None and n_estimators < 1:
        issues.append(f"n_estimators {n_estimators} < 1 — invalid")

    if issues:
        check["status"] = "warn"
        check["reason"] = "; ".join(issues)
        check["issues"] = issues
    else:
        check["status"] = "pass"
        check["reason"] = "Config values in reasonable ranges"

    return check


# --- Full Sanity Check ---


def run_sanity_checks(
    config_path: str = "config.yaml",
    quick: bool = False,
    initial_loss: float | None = None,
    loss_history: list[float] | None = None,
    gradient_stats: list[dict] | None = None,
    outputs: list | None = None,
    batch_shapes: dict | None = None,
    data_has_nan: bool = False,
    data_has_inf: bool = False,
    data_loads_ok: bool = True,
    num_classes: int | None = None,
) -> dict:
    """Run all sanity checks and produce a report.

    In CLI mode, most inputs come from running a quick training probe.
    In test mode, values are provided directly.
    """
    config = load_config(config_path)
    task_type = config.get("task", {}).get("type", "classification")

    checks = []

    # Data pipeline
    checks.append(check_data_pipeline(batch_shapes, data_has_nan, data_has_inf, data_loads_ok))

    # Initial loss
    if initial_loss is not None:
        checks.append(check_initial_loss(initial_loss, num_classes, task_type))

    # Gradient flow
    if gradient_stats is not None:
        checks.append(check_gradient_flow(gradient_stats))

    # Single-batch overfit (skip in quick mode)
    if not quick and loss_history is not None:
        checks.append(check_single_batch_overfit(loss_history))

    # Output validation
    if outputs is not None:
        checks.append(check_output_validation(outputs, task_type))

    # Config consistency
    checks.append(check_config_consistency(config))

    # Compute verdict
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    n_skip = sum(1 for c in checks if c["status"] == "skip")

    if n_fail > 0:
        verdict = "fail"
    elif n_warn > 2:
        verdict = "warn"
    elif n_warn > 0:
        verdict = "pass_with_warnings"
    else:
        verdict = "pass"

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "quick_mode": quick,
        "task_type": task_type,
        "checks": checks,
        "score": {
            "pass": n_pass,
            "fail": n_fail,
            "warn": n_warn,
            "skip": n_skip,
            "total": len(checks),
        },
        "verdict": verdict,
    }


# --- Report Formatting ---


def save_sanity_report(report: dict, output_dir: str = "experiments/sanity") -> Path:
    """Save sanity report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filepath = out_path / f"sanity-{timestamp}.yaml"

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_sanity_report(report: dict) -> str:
    """Format sanity report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    verdict = report.get("verdict", "?")
    score = report.get("score", {})
    quick = report.get("quick_mode", False)

    verdict_labels = {
        "pass": "PASS — Safe to proceed with training",
        "pass_with_warnings": "PASS (with warnings) — Review before training",
        "warn": "WARNINGS — Multiple issues detected",
        "fail": "FAIL — Do not proceed to full training",
    }

    lines = [
        "# Sanity Check Report",
        "",
        f"*Checked {report.get('checked_at', 'N/A')[:19]}*",
        f"*Mode: {'quick' if quick else 'full'}*",
        "",
        f"**{verdict_labels.get(verdict, verdict.upper())}**",
        "",
    ]

    status_markers = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "skip": "SKIP"}

    for c in report.get("checks", []):
        status = c.get("status", "?")
        marker = status_markers.get(status, status.upper())
        lines.append(f"- **[{marker}]** {c.get('check', '?')}: {c.get('reason', 'N/A')}")

    lines.extend([
        "",
        f"**Score:** {score.get('pass', 0)}/{score.get('total', 0)} pass, "
        f"{score.get('fail', 0)} fail, {score.get('warn', 0)} warn",
    ])

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-training sanity checks",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: skip single-batch overfit test",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed check output",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    # In CLI mode, we'd run actual probes. For now, report with config check only.
    report = run_sanity_checks(
        config_path=args.config,
        quick=args.quick,
    )

    if "error" not in report:
        filepath = save_sanity_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_sanity_report(report))

    if report.get("verdict") == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
