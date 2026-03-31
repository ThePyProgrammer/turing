"""Evaluation harness for the {{PROJECT_NAME}} ML pipeline.

HIDDEN — MEASUREMENT APPARATUS.

This file is hidden from the autoresearch agent. The agent cannot
read, modify, or reference this file. This prevents metric gaming,
seed exploitation, and evaluation function reverse-engineering.

The platform runs this file automatically. The agent knows only:
- The primary metric name (from config.yaml)
- Whether higher or lower is better (from config.yaml)
- The metric value (from parsed run.log output)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    config: dict | None = None,
) -> dict:
    """Compute evaluation metrics from predictions vs ground truth.

    Customize this function for your specific ML task. Add or remove
    metrics as needed. The autoresearch agent reads these via format_metrics().

    Args:
        predictions: Model predictions (numpy array).
        ground_truth: Ground truth values (numpy array).
        config: Optional config dict for metric selection.

    Returns:
        Dict with metric names and values.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truth)} ground truth"
        )

    # Determine which metrics to compute from config
    eval_cfg = config.get("evaluation", {}) if config else {}
    metric_names = eval_cfg.get("metrics", ["accuracy", "f1_weighted"])

    results = {}

    for metric_name in metric_names:
        if metric_name == "accuracy":
            results["accuracy"] = round(float(accuracy_score(ground_truth, predictions)), 4)
        elif metric_name == "f1_weighted":
            results["f1_weighted"] = round(float(f1_score(ground_truth, predictions, average="weighted")), 4)
        elif metric_name == "f1_macro":
            results["f1_macro"] = round(float(f1_score(ground_truth, predictions, average="macro")), 4)
        elif metric_name == "f1_micro":
            results["f1_micro"] = round(float(f1_score(ground_truth, predictions, average="micro")), 4)
        elif metric_name == "precision":
            results["precision"] = round(float(precision_score(ground_truth, predictions, average="weighted")), 4)
        elif metric_name == "recall":
            results["recall"] = round(float(recall_score(ground_truth, predictions, average="weighted")), 4)
        elif metric_name == "mae":
            results["mae"] = round(float(mean_absolute_error(ground_truth, predictions)), 4)
        elif metric_name == "mse":
            results["mse"] = round(float(mean_squared_error(ground_truth, predictions)), 4)
        elif metric_name == "rmse":
            results["rmse"] = round(float(np.sqrt(mean_squared_error(ground_truth, predictions))), 4)

    return results


def evaluate_detailed(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    config: dict | None = None,
) -> dict:
    """Compute detailed evaluation metrics including per-class breakdown.

    Extends evaluate_model with per-class precision/recall/F1 and a
    confusion matrix. The agent uses this to understand WHERE the model
    fails, not just that it fails.

    Args:
        predictions: Model predictions (numpy array).
        ground_truth: Ground truth values (numpy array).
        config: Optional config dict for metric selection.

    Returns:
        Dict with 'aggregate' (same as evaluate_model), 'per_class'
        (dict of class -> {precision, recall, f1, support}), and
        'confusion_matrix' (dict representation).
    """
    aggregate = evaluate_model(predictions, ground_truth, config)

    # Per-class breakdown
    classes = sorted(set(ground_truth.tolist()))
    per_class = {}

    for cls in classes:
        cls_mask = ground_truth == cls
        n_support = int(cls_mask.sum())
        cls_preds = predictions[cls_mask]

        tp = int((cls_preds == cls).sum())
        precision_denom = int((predictions == cls).sum())
        cls_precision = round(tp / precision_denom, 4) if precision_denom > 0 else 0.0
        cls_recall = round(tp / n_support, 4) if n_support > 0 else 0.0

        if cls_precision + cls_recall > 0:
            cls_f1 = round(2 * cls_precision * cls_recall / (cls_precision + cls_recall), 4)
        else:
            cls_f1 = 0.0

        per_class[str(cls)] = {
            "precision": cls_precision,
            "recall": cls_recall,
            "f1": cls_f1,
            "support": n_support,
        }

    # Confusion matrix as dict
    confusion = {}
    for true_cls in classes:
        row = {}
        for pred_cls in classes:
            row[str(pred_cls)] = int(((ground_truth == true_cls) & (predictions == pred_cls)).sum())
        confusion[str(true_cls)] = row

    return {
        "aggregate": aggregate,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def format_metrics(metrics: dict) -> str:
    """Format metrics in a parseable delimited format.

    Output format (for the autoresearch agent to parse):
        ---
        metric_name:     value
        ...
        ---

    The agent reads metrics by grepping between --- delimiters.

    Args:
        metrics: Dict with metric names and values.

    Returns:
        Formatted string.
    """
    # Separate known metadata keys from actual metrics
    metadata_keys = {"model_type", "train_seconds"}
    metric_keys = [k for k in metrics if k not in metadata_keys]
    all_keys = metric_keys + [k for k in metadata_keys if k in metrics]

    lines = ["---"]
    for key in all_keys:
        padding = " " * max(1, 15 - len(key))
        lines.append(f"{key}:{padding}{metrics[key]}")
    lines.append("---")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate {{PROJECT_NAME}} model predictions"
    )
    parser.add_argument("predictions", help="Path to predictions JSONL file")
    parser.add_argument("ground_truth", help="Path to ground truth JSONL file")
    args = parser.parse_args()

    with open(args.predictions) as f:
        preds = [json.loads(line) for line in f if line.strip()]
    with open(args.ground_truth) as f:
        truth = [json.loads(line) for line in f if line.strip()]

    pred_values = np.array([p.get("prediction") for p in preds])
    truth_values = np.array([t.get("label") for t in truth])

    result = evaluate_model(pred_values, truth_values)
    print(format_metrics(result))
