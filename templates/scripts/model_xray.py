#!/usr/bin/env python3
"""Internal model diagnostics for the autoresearch pipeline.

Inspects model internals: gradient flow per layer, activation statistics,
dead neurons, weight distributions, decision path analysis. Answers
"what is the model doing internally?" rather than "what are its predictions?"

Usage:
    python scripts/model_xray.py exp-042
    python scripts/model_xray.py exp-042 --layer "encoder.layer.2"
    python scripts/model_xray.py --compare exp-042 exp-053
    python scripts/model_xray.py --json
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
DEAD_NEURON_THRESHOLD = 0.01  # Activation below this = dead
EXPLODING_GRADIENT_RATIO = 100  # Gradient > N * mean = exploding
NEAR_ZERO_WEIGHT = 0.001  # Weight below this = pruning candidate


# --- Neural Network Diagnostics ---


def diagnose_neural_layers(layer_stats: list[dict]) -> dict:
    """Analyze neural network layer statistics.

    Args:
        layer_stats: List of dicts with name, grad_mean, grad_max, act_mean,
                     act_std, dead_pct, weight_mean, weight_std.

    Returns:
        Diagnosis dict with per-layer analysis and detected issues.
    """
    if not layer_stats:
        return {"layers": [], "issues": [], "model_type": "neural"}

    issues = []
    analyzed = []

    # Compute global gradient mean for relative comparison
    grad_means = [abs(l.get("grad_mean", 0)) for l in layer_stats if l.get("grad_mean") is not None]
    global_grad_mean = np.mean(grad_means) if grad_means else 0

    for layer in layer_stats:
        name = layer.get("name", "?")
        analysis = {"name": name}

        # Gradient analysis
        grad_mean = abs(layer.get("grad_mean", 0))
        grad_max = abs(layer.get("grad_max", 0))
        analysis["grad_mean"] = grad_mean
        analysis["grad_max"] = grad_max

        if grad_mean == 0 and grad_max == 0:
            issues.append({"layer": name, "issue": "dead_gradient", "severity": "high",
                          "message": f"{name}: zero gradients — layer is not learning"})
        elif global_grad_mean > 0 and grad_mean < global_grad_mean / EXPLODING_GRADIENT_RATIO:
            ratio = global_grad_mean / grad_mean if grad_mean > 0 else float("inf")
            issues.append({"layer": name, "issue": "vanishing_gradient", "severity": "high",
                          "message": f"{name}: gradient {ratio:.0f}x weaker than average — possible vanishing gradient"})
        elif global_grad_mean > 0 and grad_max > EXPLODING_GRADIENT_RATIO * global_grad_mean:
            issues.append({"layer": name, "issue": "exploding_gradient", "severity": "critical",
                          "message": f"{name}: gradient max {grad_max:.2e} is {grad_max/global_grad_mean:.0f}x the average — exploding gradient"})

        # Activation analysis
        dead_pct = layer.get("dead_pct", 0)
        analysis["dead_pct"] = dead_pct
        if dead_pct > 20:
            issues.append({"layer": name, "issue": "dead_neurons", "severity": "high",
                          "message": f"{name}: {dead_pct:.0f}% dead neurons — consider batch norm or layer width reduction"})
        elif dead_pct > 5:
            issues.append({"layer": name, "issue": "dying_neurons", "severity": "medium",
                          "message": f"{name}: {dead_pct:.0f}% near-dead neurons"})

        # Weight analysis
        weight_std = layer.get("weight_std", 0)
        near_zero_pct = layer.get("near_zero_pct", 0)
        analysis["weight_std"] = weight_std
        analysis["near_zero_pct"] = near_zero_pct
        if near_zero_pct > 50:
            issues.append({"layer": name, "issue": "sparse_weights", "severity": "medium",
                          "message": f"{name}: {near_zero_pct:.0f}% near-zero weights — pruning candidate"})

        analyzed.append(analysis)

    return {"layers": analyzed, "issues": issues, "model_type": "neural"}


# --- Tree Model Diagnostics ---


def diagnose_tree_model(tree_stats: dict) -> dict:
    """Analyze tree-based model statistics.

    Args:
        tree_stats: Dict with n_trees, avg_depth, max_depth_allowed,
                    feature_split_freq, leaf_purity.

    Returns:
        Diagnosis dict.
    """
    issues = []

    n_trees = tree_stats.get("n_trees", 0)
    avg_depth = tree_stats.get("avg_depth", 0)
    max_depth = tree_stats.get("max_depth_allowed", 0)
    feature_splits = tree_stats.get("feature_split_freq", {})
    leaf_purity = tree_stats.get("leaf_purity", 0)

    # Depth utilization
    if max_depth > 0 and avg_depth > 0:
        utilization = avg_depth / max_depth
        if utilization < 0.5:
            issues.append({"issue": "underutilized_depth", "severity": "medium",
                          "message": f"Trees use only {utilization:.0%} of allowed depth ({avg_depth:.1f}/{max_depth}) — consider reducing max_depth"})
        elif utilization > 0.95:
            issues.append({"issue": "depth_saturated", "severity": "medium",
                          "message": f"Trees use {utilization:.0%} of allowed depth — consider increasing max_depth"})

    # Feature dominance
    if feature_splits:
        total_splits = sum(feature_splits.values())
        if total_splits > 0:
            top_feature = max(feature_splits, key=feature_splits.get)
            top_pct = feature_splits[top_feature] / total_splits
            if top_pct > 0.5:
                issues.append({"issue": "feature_dominance", "severity": "medium",
                              "message": f"Feature '{top_feature}' dominates {top_pct:.0%} of splits — check for leakage or engineering opportunity"})

    # Leaf purity
    if leaf_purity > 0.99:
        issues.append({"issue": "overfitting_risk", "severity": "medium",
                      "message": f"Leaf purity {leaf_purity:.4f} — model may be overfitting"})

    return {
        "model_type": "tree",
        "n_trees": n_trees,
        "avg_depth": avg_depth,
        "max_depth_allowed": max_depth,
        "depth_utilization": round(avg_depth / max_depth, 3) if max_depth > 0 else None,
        "feature_split_freq": feature_splits,
        "leaf_purity": leaf_purity,
        "issues": issues,
    }


# --- sklearn Diagnostics ---


def diagnose_sklearn_model(model_stats: dict) -> dict:
    """Analyze scikit-learn model statistics.

    Args:
        model_stats: Dict with model_type, coefficients, feature_importances.
    """
    issues = []
    model_type = model_stats.get("model_type", "unknown")

    coefficients = model_stats.get("coefficients", [])
    if coefficients:
        coef_arr = np.array(coefficients)
        max_coef = float(np.max(np.abs(coef_arr)))
        near_zero = float(np.mean(np.abs(coef_arr) < NEAR_ZERO_WEIGHT))

        if max_coef > 100:
            issues.append({"issue": "large_coefficients", "severity": "high",
                          "message": f"Max coefficient magnitude {max_coef:.1f} — consider regularization"})
        if near_zero > 0.5:
            issues.append({"issue": "sparse_coefficients", "severity": "medium",
                          "message": f"{near_zero:.0%} near-zero coefficients — feature selection may help"})

    importances = model_stats.get("feature_importances", [])
    if importances:
        imp_arr = np.array(importances)
        if len(imp_arr) > 0 and np.std(imp_arr) > 0:
            top_k = min(3, len(imp_arr))
            top_indices = np.argsort(imp_arr)[-top_k:]
            top_total = float(np.sum(imp_arr[top_indices]))
            if top_total > 0.8:
                issues.append({"issue": "importance_concentrated", "severity": "medium",
                              "message": f"Top {top_k} features account for {top_total:.0%} of importance"})

    return {
        "model_type": model_type,
        "n_coefficients": len(coefficients),
        "n_importances": len(importances),
        "issues": issues,
    }


# --- Full X-Ray Pipeline ---


def xray_model(
    exp_id: str | None = None,
    layer_stats: list[dict] | None = None,
    tree_stats: dict | None = None,
    sklearn_stats: dict | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Run model diagnostics."""
    config = load_config(config_path)
    model_type_hint = config.get("model", {}).get("type", "")

    diagnosis = None
    if layer_stats is not None:
        diagnosis = diagnose_neural_layers(layer_stats)
    elif tree_stats is not None:
        diagnosis = diagnose_tree_model(tree_stats)
    elif sklearn_stats is not None:
        diagnosis = diagnose_sklearn_model(sklearn_stats)
    else:
        diagnosis = {"model_type": "unknown", "issues": [],
                     "note": "No model stats provided. Run with model-specific stats for full diagnostics."}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": exp_id,
        "diagnosis": diagnosis,
        "n_issues": len(diagnosis.get("issues", [])),
    }


# --- Report Formatting ---


def save_xray_report(report: dict, output_dir: str = "experiments/xrays") -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("experiment_id", "unknown")
    filepath = out_path / f"{exp_id}-xray.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_xray_report(report: dict) -> str:
    if "error" in report:
        return f"ERROR: {report['error']}"

    diag = report.get("diagnosis", {})
    model_type = diag.get("model_type", "?")
    exp_id = report.get("experiment_id", "?")
    issues = diag.get("issues", [])

    lines = [f"# X-Ray: {exp_id} ({model_type})", "",
             f"*Generated {report.get('generated_at', 'N/A')[:19]}*", ""]

    # Neural layer table
    layers = diag.get("layers", [])
    if layers:
        lines.extend(["## Layer Analysis", "",
                      "| Layer | Grad Mean | Grad Max | Dead % | Weight Std |",
                      "|-------|-----------|----------|--------|------------|"])
        for l in layers:
            lines.append(f"| {l['name']} | {l.get('grad_mean', 0):.2e} | {l.get('grad_max', 0):.2e} | {l.get('dead_pct', 0):.0f}% | {l.get('weight_std', 0):.4f} |")
        lines.append("")

    # Tree stats
    if model_type == "tree":
        lines.extend(["## Tree Statistics", "",
                      f"- **Trees:** {diag.get('n_trees', '?')}",
                      f"- **Avg depth:** {diag.get('avg_depth', '?')}/{diag.get('max_depth_allowed', '?')}",
                      f"- **Leaf purity:** {diag.get('leaf_purity', '?')}", ""])

    # Issues
    if issues:
        lines.extend(["## Issues Detected", ""])
        for i in issues:
            sev = i.get("severity", "?").upper()
            lines.append(f"- **[{sev}]** {i.get('message', 'N/A')}")
    else:
        lines.extend(["## Issues Detected", "", "No issues found."])

    if diag.get("note"):
        lines.extend(["", f"*{diag['note']}*"])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal model diagnostics")
    parser.add_argument("exp_id", nargs="?", help="Experiment ID")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = xray_model(exp_id=args.exp_id, config_path=args.config, log_path=args.log)

    if "error" not in report:
        filepath = save_xray_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_xray_report(report))


if __name__ == "__main__":
    main()
