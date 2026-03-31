#!/usr/bin/env python3
"""Error analysis for ML experiments.

Goes beyond aggregate metrics to answer "where and why does this model
fail?" Clusters failure cases, identifies systematic failure modes, and
suggests targeted fixes as auto-queued hypotheses.

For classification: confusion matrix, most-confused pairs, per-class P/R.
For regression: high-residual analysis, feature-range bias detection.

Usage:
    python scripts/diagnose_errors.py                      # Best experiment
    python scripts/diagnose_errors.py --exp-id exp-042     # Specific experiment
    python scripts/diagnose_errors.py --auto-queue         # Queue fix hypotheses
    python scripts/diagnose_errors.py --top 5              # Top 5 failure modes
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments


def find_experiment(experiments: list[dict], exp_id: str | None, metric: str, lower_is_better: bool) -> dict | None:
    """Find experiment by ID or return best kept experiment."""
    if exp_id:
        for exp in experiments:
            if exp.get("experiment_id") == exp_id:
                return exp
        return None
    best = None
    best_val = float("inf") if lower_is_better else float("-inf")
    for exp in experiments:
        if exp.get("status") != "kept":
            continue
        val = exp.get("metrics", {}).get(metric)
        if val is None:
            continue
        if (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val = val
            best = exp
    return best


def compute_confusion_matrix(y_true: list, y_pred: list) -> dict:
    """Compute confusion matrix and derived metrics for classification.

    Returns dict with matrix, classes, per_class metrics, and most_confused pairs.
    """
    classes = sorted(set(y_true) | set(y_pred))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)

    matrix = [[0] * n for _ in range(n)]
    for true, pred in zip(y_true, y_pred):
        matrix[class_to_idx[true]][class_to_idx[pred]] += 1

    # Per-class precision, recall, F1
    per_class = {}
    for i, cls in enumerate(classes):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n)) - tp
        fn = sum(matrix[i][j] for j in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(matrix[i])
        per_class[str(cls)] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Most confused pairs (off-diagonal entries sorted by count)
    confused_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > 0:
                confused_pairs.append({
                    "true_class": str(classes[i]),
                    "predicted_class": str(classes[j]),
                    "count": matrix[i][j],
                    "pct_of_true": round(matrix[i][j] / max(sum(matrix[i]), 1) * 100, 1),
                })
    confused_pairs.sort(key=lambda x: -x["count"])

    return {
        "classes": [str(c) for c in classes],
        "matrix": matrix,
        "per_class": per_class,
        "most_confused": confused_pairs[:10],
        "total_errors": sum(1 for t, p in zip(y_true, y_pred) if t != p),
        "total_samples": len(y_true),
        "error_rate": round(sum(1 for t, p in zip(y_true, y_pred) if t != p) / max(len(y_true), 1), 4),
    }


def analyze_regression_errors(
    y_true: list[float],
    y_pred: list[float],
    features: list[dict] | None = None,
    top_n: int = 5,
) -> dict:
    """Analyze regression errors: high-residual samples, feature-range bias.

    Returns dict with residual stats, worst predictions, and feature-range analysis.
    """
    residuals = [abs(t - p) for t, p in zip(y_true, y_pred)]
    signed_residuals = [p - t for t, p in zip(y_true, y_pred)]

    arr = np.array(residuals)
    mean_error = float(np.mean(arr))
    median_error = float(np.median(arr))
    std_error = float(np.std(arr))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))

    # Worst predictions
    indexed = list(enumerate(residuals))
    indexed.sort(key=lambda x: -x[1])
    worst = []
    for idx, res in indexed[:top_n]:
        entry = {
            "index": idx,
            "true": round(y_true[idx], 4),
            "predicted": round(y_pred[idx], 4),
            "residual": round(res, 4),
        }
        if features and idx < len(features):
            entry["features"] = features[idx]
        worst.append(entry)

    result = {
        "mean_absolute_error": round(mean_error, 4),
        "median_absolute_error": round(median_error, 4),
        "std_error": round(std_error, 4),
        "p90_error": round(p90, 4),
        "p95_error": round(p95, 4),
        "worst_predictions": worst,
        "bias": round(float(np.mean(signed_residuals)), 4),
    }

    # Feature-range analysis (if features provided)
    if features and len(features) > 0:
        feature_analysis = analyze_feature_ranges(y_true, y_pred, residuals, features)
        result["feature_range_bias"] = feature_analysis

    return result


def analyze_feature_ranges(
    y_true: list[float],
    y_pred: list[float],
    residuals: list[float],
    features: list[dict],
) -> list[dict]:
    """Find feature ranges where the model performs systematically worse.

    For each numeric feature, split samples into quartiles and compare
    error rates across quartiles.
    """
    if not features:
        return []

    # Collect numeric features
    numeric_keys = set()
    for f in features:
        for k, v in f.items():
            if isinstance(v, (int, float)) and not math.isnan(v):
                numeric_keys.add(k)

    results = []
    for key in sorted(numeric_keys):
        values = []
        errors = []
        for i, f in enumerate(features):
            if key in f and isinstance(f[key], (int, float)) and i < len(residuals):
                values.append(f[key])
                errors.append(residuals[i])

        if len(values) < 8:
            continue

        arr_v = np.array(values)
        arr_e = np.array(errors)
        quartiles = np.percentile(arr_v, [25, 50, 75])

        bins = [
            ("Q1", arr_v <= quartiles[0]),
            ("Q2", (arr_v > quartiles[0]) & (arr_v <= quartiles[1])),
            ("Q3", (arr_v > quartiles[1]) & (arr_v <= quartiles[2])),
            ("Q4", arr_v > quartiles[2]),
        ]

        bin_errors = {}
        for name, mask in bins:
            if mask.sum() > 0:
                bin_errors[name] = round(float(np.mean(arr_e[mask])), 4)

        if not bin_errors:
            continue

        max_error_bin = max(bin_errors, key=bin_errors.get)
        min_error_bin = min(bin_errors, key=bin_errors.get)
        ratio = bin_errors[max_error_bin] / max(bin_errors[min_error_bin], 1e-8)

        if ratio > 2.0:
            results.append({
                "feature": key,
                "worst_quartile": max_error_bin,
                "best_quartile": min_error_bin,
                "error_ratio": round(ratio, 2),
                "bin_errors": bin_errors,
                "description": f"Model error is {ratio:.1f}x higher in {max_error_bin} ({key}) vs {min_error_bin}",
            })

    results.sort(key=lambda x: -x["error_ratio"])
    return results


def identify_failure_modes(
    confusion_data: dict | None = None,
    regression_data: dict | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Extract actionable failure modes from analysis results.

    Returns a list of failure mode dicts with id, description, affected_samples,
    suggested_fix, and auto_hypothesis.
    """
    modes = []
    fm_id = 1

    if confusion_data:
        # Failure mode from confused pairs
        for pair in confusion_data.get("most_confused", [])[:top_n]:
            tc = pair["true_class"]
            pc = pair["predicted_class"]
            count = pair["count"]
            pct = pair["pct_of_true"]
            total_errors = confusion_data.get("total_errors", 1)
            error_pct = round(count / max(total_errors, 1) * 100, 1)

            modes.append({
                "id": f"fm-{fm_id:03d}",
                "type": "class_confusion",
                "description": f"Model confuses '{tc}' and '{pc}' — {count} errors ({error_pct}% of all errors)",
                "affected_samples": count,
                "pct_of_errors": error_pct,
                "suggested_fix": f"Add distinguishing features for '{tc}' vs '{pc}', or increase training data for these classes",
                "auto_hypothesis": f"Add targeted features to distinguish class '{tc}' from '{pc}'",
            })
            fm_id += 1

        # Failure mode from low-recall classes
        for cls, stats in confusion_data.get("per_class", {}).items():
            if stats["recall"] < 0.5 and stats["support"] >= 5:
                modes.append({
                    "id": f"fm-{fm_id:03d}",
                    "type": "low_recall",
                    "description": f"Class '{cls}' has recall={stats['recall']:.2f} — model misses {stats['fn']} of {stats['support']} samples",
                    "affected_samples": stats["fn"],
                    "pct_of_errors": round(stats["fn"] / max(confusion_data.get("total_errors", 1), 1) * 100, 1),
                    "suggested_fix": f"Oversample class '{cls}' or add class-specific features",
                    "auto_hypothesis": f"Apply SMOTE or class weights to improve recall on class '{cls}'",
                })
                fm_id += 1

    if regression_data:
        # Failure mode from feature-range bias
        for fb in regression_data.get("feature_range_bias", [])[:top_n]:
            modes.append({
                "id": f"fm-{fm_id:03d}",
                "type": "feature_range_bias",
                "description": f"High error when {fb['feature']} in {fb['worst_quartile']} — {fb['error_ratio']:.1f}x worse than {fb['best_quartile']}",
                "affected_samples": 0,
                "suggested_fix": f"Add {fb['feature']} binning or cap outliers in {fb['worst_quartile']} range",
                "auto_hypothesis": f"Bin {fb['feature']} into quantiles instead of raw values",
            })
            fm_id += 1

        # Failure mode from systematic bias
        bias = regression_data.get("bias", 0)
        if abs(bias) > regression_data.get("std_error", 0) * 0.5:
            direction = "over-predicts" if bias > 0 else "under-predicts"
            modes.append({
                "id": f"fm-{fm_id:03d}",
                "type": "systematic_bias",
                "description": f"Model systematically {direction} by {abs(bias):.4f} on average",
                "affected_samples": regression_data.get("worst_predictions", [{}])[0].get("index", 0),
                "suggested_fix": f"Add bias correction or investigate data distribution skew",
                "auto_hypothesis": f"Add target variable transformation to correct {direction} bias",
            })
            fm_id += 1

    # Sort by affected_samples (most impactful first)
    modes.sort(key=lambda m: -m.get("affected_samples", 0))
    return modes[:top_n]


def generate_hypotheses_from_modes(failure_modes: list[dict]) -> list[dict]:
    """Convert failure modes into hypothesis queue entries.

    Returns list of hypothesis dicts ready for hypotheses.yaml.
    """
    hypotheses = []
    for mode in failure_modes:
        if not mode.get("auto_hypothesis"):
            continue
        hypotheses.append({
            "id": f"hyp-diag-{mode['id'].split('-')[-1]}",
            "description": mode["auto_hypothesis"],
            "source": "diagnose",
            "status": "queued",
            "priority": "high" if mode.get("affected_samples", 0) > 10 else "normal",
            "rationale": mode["description"],
            "failure_mode_id": mode["id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    return hypotheses


def save_diagnosis(diagnosis: dict, output_dir: str = "experiments/diagnoses") -> Path:
    """Save diagnosis report to YAML file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_id = diagnosis.get("experiment_id", "unknown")
    filepath = out_path / f"{exp_id}-diagnosis.yaml"
    with open(filepath, "w") as f:
        yaml.dump(diagnosis, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_diagnosis_report(diagnosis: dict) -> str:
    """Format diagnosis as human-readable markdown report."""
    if "error" in diagnosis:
        return f"ERROR: {diagnosis['error']}"

    exp_id = diagnosis["experiment_id"]
    task_type = diagnosis.get("task_type", "unknown")

    lines = [
        f"# Error Analysis: {exp_id}",
        "",
        f"*Task type: {task_type}*",
        "",
    ]

    # Classification-specific
    confusion = diagnosis.get("confusion_matrix")
    if confusion:
        lines.extend([
            "## Error Summary",
            "",
            f"- **Total samples:** {confusion['total_samples']}",
            f"- **Total errors:** {confusion['total_errors']}",
            f"- **Error rate:** {confusion['error_rate']:.2%}",
            "",
            "## Per-Class Performance",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|-------|-----------|--------|-----|---------|",
        ])
        for cls, stats in confusion.get("per_class", {}).items():
            lines.append(
                f"| {cls} | {stats['precision']:.3f} | {stats['recall']:.3f} "
                f"| {stats['f1']:.3f} | {stats['support']} |"
            )

        if confusion.get("most_confused"):
            lines.extend([
                "",
                "## Most Confused Pairs",
                "",
                "| True | Predicted | Count | % of True Class |",
                "|------|-----------|-------|-----------------|",
            ])
            for pair in confusion["most_confused"][:5]:
                lines.append(
                    f"| {pair['true_class']} | {pair['predicted_class']} "
                    f"| {pair['count']} | {pair['pct_of_true']}% |"
                )

    # Regression-specific
    regression = diagnosis.get("regression_analysis")
    if regression:
        lines.extend([
            "## Error Distribution",
            "",
            f"- **Mean absolute error:** {regression['mean_absolute_error']:.4f}",
            f"- **Median absolute error:** {regression['median_absolute_error']:.4f}",
            f"- **P90 error:** {regression['p90_error']:.4f}",
            f"- **P95 error:** {regression['p95_error']:.4f}",
            f"- **Systematic bias:** {regression['bias']:.4f}",
        ])

        if regression.get("feature_range_bias"):
            lines.extend(["", "## Feature-Range Bias", ""])
            for fb in regression["feature_range_bias"]:
                lines.append(f"- **{fb['feature']}:** {fb['description']}")

    # Failure modes
    modes = diagnosis.get("failure_modes", [])
    if modes:
        lines.extend([
            "",
            "## Failure Modes",
            "",
        ])
        for mode in modes:
            lines.append(f"### {mode['id']}: {mode['type']}")
            lines.append("")
            lines.append(f"- **Description:** {mode['description']}")
            lines.append(f"- **Affected samples:** {mode['affected_samples']}")
            lines.append(f"- **Suggested fix:** {mode['suggested_fix']}")
            if mode.get("auto_hypothesis"):
                lines.append(f"- **Auto-hypothesis:** {mode['auto_hypothesis']}")
            lines.append("")

    # Auto-queued hypotheses
    hypotheses = diagnosis.get("auto_hypotheses", [])
    if hypotheses:
        lines.extend([
            "## Auto-Queued Hypotheses",
            "",
        ])
        for hyp in hypotheses:
            priority = f" **(HIGH)**" if hyp.get("priority") == "high" else ""
            lines.append(f"- {hyp['id']}: {hyp['description']}{priority}")

    return "\n".join(lines)


def queue_hypotheses(hypotheses: list[dict], queue_path: str = "hypotheses.yaml") -> int:
    """Append hypotheses to the hypothesis queue file.

    Returns number of hypotheses added.
    """
    path = Path(queue_path)
    existing = []
    if path.exists() and path.stat().st_size > 0:
        with open(path) as f:
            data = yaml.safe_load(f)
            if isinstance(data, list):
                existing = data

    # Avoid duplicate IDs
    existing_ids = {h.get("id") for h in existing}
    new = [h for h in hypotheses if h["id"] not in existing_ids]

    if new:
        existing.extend(new)
        with open(path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    return len(new)


def diagnose_experiment(
    exp_id: str | None = None,
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
    predictions_path: str | None = None,
    auto_queue: bool = False,
    top_n: int = 5,
) -> dict:
    """Run error analysis on an experiment.

    This function operates on pre-computed predictions. If no predictions
    file is provided, it looks for experiments/predictions/exp-NNN-preds.yaml.

    Args:
        exp_id: Experiment ID (defaults to best experiment).
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        predictions_path: Path to predictions YAML file.
        auto_queue: Whether to auto-queue hypotheses.
        top_n: Number of failure modes to report.

    Returns:
        Diagnosis dict with analysis results and failure modes.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)
    target_exp = find_experiment(experiments, exp_id, primary_metric, lower_is_better)

    if not target_exp:
        return {"error": f"No experiment found{f' with ID {exp_id}' if exp_id else ''}", "experiment_id": exp_id}

    target_id = target_exp.get("experiment_id", "unknown")

    # Load predictions
    if not predictions_path:
        predictions_path = f"experiments/predictions/{target_id}-preds.yaml"

    preds_file = Path(predictions_path)
    if not preds_file.exists():
        return {
            "error": f"Predictions file not found: {predictions_path}. Run the model on the validation set first.",
            "experiment_id": target_id,
            "hint": "Generate predictions with: python train.py --predict-only --output experiments/predictions/",
        }

    with open(preds_file) as f:
        preds_data = yaml.safe_load(f) or {}

    y_true = preds_data.get("y_true", [])
    y_pred = preds_data.get("y_pred", [])
    features = preds_data.get("features", None)
    task_type = preds_data.get("task_type", "classification")

    if not y_true or not y_pred:
        return {"error": "Predictions file has no y_true/y_pred data", "experiment_id": target_id}

    diagnosis = {
        "experiment_id": target_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type,
        "n_samples": len(y_true),
        "primary_metric": primary_metric,
        "original_metrics": target_exp.get("metrics", {}),
    }

    if task_type == "classification":
        confusion = compute_confusion_matrix(y_true, y_pred)
        diagnosis["confusion_matrix"] = confusion
        failure_modes = identify_failure_modes(confusion_data=confusion, top_n=top_n)
    elif task_type == "regression":
        y_true_f = [float(v) for v in y_true]
        y_pred_f = [float(v) for v in y_pred]
        regression = analyze_regression_errors(y_true_f, y_pred_f, features, top_n=top_n)
        diagnosis["regression_analysis"] = regression
        failure_modes = identify_failure_modes(regression_data=regression, top_n=top_n)
    else:
        failure_modes = []

    diagnosis["failure_modes"] = failure_modes

    # Generate and optionally queue hypotheses
    hypotheses = generate_hypotheses_from_modes(failure_modes)
    diagnosis["auto_hypotheses"] = hypotheses

    if auto_queue and hypotheses:
        n_added = queue_hypotheses(hypotheses)
        diagnosis["hypotheses_queued"] = n_added
        print(f"Queued {n_added} hypotheses from failure modes", file=sys.stderr)

    return diagnosis


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Error analysis for ML experiments")
    parser.add_argument("--exp-id", default=None, help="Experiment ID (defaults to best)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl", help="Path to experiment log")
    parser.add_argument("--predictions", default=None, help="Path to predictions YAML file")
    parser.add_argument("--auto-queue", action="store_true", help="Auto-queue hypotheses from failure modes")
    parser.add_argument("--top", type=int, default=5, help="Number of failure modes to report")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    diagnosis = diagnose_experiment(
        exp_id=args.exp_id,
        config_path=args.config,
        log_path=args.log,
        predictions_path=args.predictions,
        auto_queue=args.auto_queue,
        top_n=args.top,
    )

    if "error" not in diagnosis:
        filepath = save_diagnosis(diagnosis)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(diagnosis, indent=2, default=str))
    else:
        print(format_diagnosis_report(diagnosis))


if __name__ == "__main__":
    main()
