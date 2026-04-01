#!/usr/bin/env python3
"""Automated ensemble construction for the autoresearch pipeline.

Builds ensembles from the top-K experiments automatically. Tries voting,
weighted voting, stacking, and blending. Often yields 1-3% improvement
from models already trained — zero additional training cost.

Usage:
    python scripts/build_ensemble.py
    python scripts/build_ensemble.py --top-k 5
    python scripts/build_ensemble.py --methods voting,stacking
    python scripts/build_ensemble.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_TOP_K = 5
DEFAULT_METHODS = ["voting", "weighted_voting", "stacking", "blending"]
BLEND_HOLDOUT_RATIO = 0.3
MIN_DIVERSITY_THRESHOLD = 0.95  # Correlation above this = too similar


# --- Model Selection ---


def select_top_k(
    experiments: list[dict],
    primary_metric: str,
    k: int,
    lower_is_better: bool = False,
) -> list[dict]:
    """Select top-K experiments by primary metric.

    Args:
        experiments: All experiments.
        primary_metric: Metric to rank by.
        k: Number of top experiments.
        lower_is_better: Whether lower metric is better.

    Returns:
        Top-K experiments sorted by metric.
    """
    kept = [e for e in experiments if e.get("status") == "kept"]
    if not kept:
        kept = [e for e in experiments if primary_metric in e.get("metrics", {})]

    with_metric = [
        e for e in kept
        if isinstance(e.get("metrics", {}).get(primary_metric), (int, float))
    ]

    with_metric.sort(
        key=lambda e: e["metrics"][primary_metric],
        reverse=not lower_is_better,
    )

    return with_metric[:k]


def compute_prediction_correlation(predictions: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise correlation matrix of model predictions.

    Args:
        predictions: List of prediction arrays, one per model.

    Returns:
        NxN correlation matrix.
    """
    n = len(predictions)
    if n < 2:
        return np.eye(n)

    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if len(predictions[i]) == len(predictions[j]) and len(predictions[i]) > 0:
                c = np.corrcoef(predictions[i].ravel(), predictions[j].ravel())[0, 1]
                if np.isnan(c):
                    c = 0.0
                corr[i, j] = c
                corr[j, i] = c

    return corr


def filter_diverse_models(
    experiments: list[dict],
    predictions: list[np.ndarray] | None,
    threshold: float = MIN_DIVERSITY_THRESHOLD,
) -> tuple[list[dict], list[int]]:
    """Filter out models with highly correlated predictions.

    Args:
        experiments: Candidate experiments.
        predictions: Prediction arrays (same order as experiments).
        threshold: Max correlation to keep both models.

    Returns:
        (filtered_experiments, kept_indices)
    """
    if predictions is None or len(predictions) < 2:
        return experiments, list(range(len(experiments)))

    corr = compute_prediction_correlation(predictions)
    n = len(experiments)
    kept = [True] * n

    for i in range(n):
        if not kept[i]:
            continue
        for j in range(i + 1, n):
            if not kept[j]:
                continue
            if abs(corr[i, j]) > threshold:
                # Drop the worse model
                metric_i = next(iter(experiments[i].get("metrics", {}).values()), 0)
                metric_j = next(iter(experiments[j].get("metrics", {}).values()), 0)
                if metric_j >= metric_i:
                    kept[i] = False
                else:
                    kept[j] = False

    indices = [i for i in range(n) if kept[i]]
    filtered = [experiments[i] for i in indices]
    return filtered, indices


# --- Ensemble Methods ---


def voting_ensemble(
    predictions: list[np.ndarray],
    task_type: str = "classification",
) -> np.ndarray:
    """Uniform voting ensemble.

    Classification: majority vote.
    Regression: simple mean.
    """
    if not predictions:
        return np.array([])

    stacked = np.stack(predictions)

    if task_type == "classification":
        # Majority vote (assumes integer class labels)
        from scipy import stats as scipy_stats
        result, _ = scipy_stats.mode(stacked, axis=0, keepdims=False)
        return result.ravel()
    else:
        return np.mean(stacked, axis=0)


def weighted_voting_ensemble(
    predictions: list[np.ndarray],
    weights: list[float],
    task_type: str = "classification",
) -> np.ndarray:
    """Weighted voting/averaging ensemble.

    Classification: weighted majority vote.
    Regression: weighted mean.
    """
    if not predictions or not weights:
        return np.array([])

    w = np.array(weights)
    w = w / w.sum()  # Normalize

    if task_type == "classification":
        # Weighted vote: accumulate votes per class
        stacked = np.stack(predictions)
        n_samples = stacked.shape[1] if stacked.ndim > 1 else len(stacked[0])
        result = np.zeros(n_samples)
        classes = np.unique(stacked)
        for idx in range(n_samples):
            class_votes = {}
            for m, pred in enumerate(predictions):
                val = pred[idx] if idx < len(pred) else 0
                class_votes[val] = class_votes.get(val, 0) + w[m]
            result[idx] = max(class_votes, key=class_votes.get)
        return result
    else:
        stacked = np.stack(predictions)
        return np.average(stacked, axis=0, weights=w)


def stacking_ensemble(
    predictions: list[np.ndarray],
    labels: np.ndarray,
    task_type: str = "classification",
    n_folds: int = 5,
) -> dict:
    """Stacking ensemble with cross-validated meta-learner.

    Trains a logistic regression (classification) or ridge regression
    on out-of-fold predictions from base models.

    Returns:
        Dict with meta_predictions, meta_weights, meta_model_type.
    """
    if not predictions or len(labels) == 0:
        return {"meta_predictions": np.array([]), "meta_weights": [], "meta_model_type": "none"}

    # Build meta-features: NxM matrix (N samples, M models)
    X_meta = np.column_stack(predictions)
    y = labels

    n_samples = len(y)
    if n_samples < n_folds:
        n_folds = max(2, n_samples)

    fold_size = n_samples // n_folds
    oof_predictions = np.zeros(n_samples)
    meta_weights = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else n_samples

        val_idx = list(range(start, end))
        train_idx = [i for i in range(n_samples) if i not in val_idx]

        X_train, X_val = X_meta[train_idx], X_meta[val_idx]
        y_train = y[train_idx]

        if task_type == "classification":
            # Simple logistic regression via closed-form softmax regression
            # Using sklearn-like approach but manual for minimal deps
            weights = _fit_linear_meta(X_train, y_train, regularize=True)
        else:
            weights = _fit_linear_meta(X_train, y_train, regularize=True)

        oof_predictions[val_idx] = X_val @ weights
        meta_weights.append(weights)

    # Average weights across folds
    avg_weights = np.mean(meta_weights, axis=0)

    if task_type == "classification":
        oof_predictions = np.round(oof_predictions).astype(int)

    return {
        "meta_predictions": oof_predictions,
        "meta_weights": avg_weights.tolist(),
        "meta_model_type": "ridge" if task_type == "regression" else "logistic",
    }


def blending_ensemble(
    predictions: list[np.ndarray],
    labels: np.ndarray,
    task_type: str = "classification",
    holdout_ratio: float = BLEND_HOLDOUT_RATIO,
) -> dict:
    """Blending ensemble using holdout set for meta-learner.

    Simpler than stacking (no cross-validation), but less data-efficient.

    Returns:
        Dict with meta_predictions, meta_weights, holdout_size.
    """
    if not predictions or len(labels) == 0:
        return {"meta_predictions": np.array([]), "meta_weights": [], "holdout_size": 0}

    X_meta = np.column_stack(predictions)
    y = labels

    n_samples = len(y)
    split = int(n_samples * (1 - holdout_ratio))
    if split < 2 or n_samples - split < 2:
        return {"meta_predictions": np.array([]), "meta_weights": [], "holdout_size": 0}

    X_train, X_val = X_meta[:split], X_meta[split:]
    y_train, y_val = y[:split], y[split:]

    weights = _fit_linear_meta(X_train, y_train, regularize=True)
    blend_predictions = X_val @ weights

    if task_type == "classification":
        blend_predictions = np.round(blend_predictions).astype(int)

    return {
        "meta_predictions": blend_predictions,
        "meta_weights": weights.tolist(),
        "holdout_size": n_samples - split,
        "holdout_labels": y_val,
    }


def _fit_linear_meta(X: np.ndarray, y: np.ndarray, regularize: bool = True) -> np.ndarray:
    """Fit a linear meta-learner (ridge regression).

    Returns weight vector of shape (n_models,).
    """
    n_features = X.shape[1]
    alpha = 1.0 if regularize else 0.0

    # Ridge: w = (X^T X + alpha I)^-1 X^T y
    XtX = X.T @ X + alpha * np.eye(n_features)
    Xty = X.T @ y

    try:
        weights = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        weights = np.ones(n_features) / n_features

    return weights


# --- Evaluation ---


def evaluate_ensemble(
    predictions: np.ndarray,
    labels: np.ndarray,
    task_type: str = "classification",
) -> dict:
    """Evaluate ensemble predictions against ground truth.

    Returns dict with accuracy (classification) or mse/rmse (regression).
    """
    if len(predictions) == 0 or len(labels) == 0:
        return {}

    min_len = min(len(predictions), len(labels))
    predictions = predictions[:min_len]
    labels = labels[:min_len]

    if task_type == "classification":
        correct = np.sum(predictions == labels)
        return {
            "accuracy": round(float(correct / min_len), 6),
            "n_samples": min_len,
        }
    else:
        mse = float(np.mean((predictions - labels) ** 2))
        return {
            "mse": round(mse, 6),
            "rmse": round(float(np.sqrt(mse)), 6),
            "n_samples": min_len,
        }


# --- Full Ensemble Pipeline ---


def build_ensemble(
    top_k: int = DEFAULT_TOP_K,
    methods: list[str] | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    predictions_dir: str = "experiments/predictions",
) -> dict:
    """Build and evaluate ensembles from top-K experiments.

    Args:
        top_k: Number of top models to consider.
        methods: Ensemble methods to try.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        predictions_dir: Directory containing saved predictions.

    Returns:
        Complete ensemble report.
    """
    if methods is None:
        methods = DEFAULT_METHODS

    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)
    task_type = config.get("task", {}).get("type", "classification")

    experiments = load_experiments(log_path)
    candidates = select_top_k(experiments, primary_metric, top_k, lower_is_better)

    if not candidates:
        return {"error": f"No experiments with {primary_metric} found in {log_path}"}

    if len(candidates) < 2:
        return {"error": "Need at least 2 experiments for ensemble building"}

    # Load predictions if available
    predictions = _load_predictions(candidates, predictions_dir)
    labels = _load_labels(predictions_dir)

    # Diversity analysis
    diversity = {}
    if predictions:
        corr_matrix = compute_prediction_correlation(predictions)
        diversity = {
            "correlation_matrix": corr_matrix.tolist(),
            "mean_correlation": float(np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])) if len(corr_matrix) > 1 else 0.0,
            "model_ids": [e.get("experiment_id", "?") for e in candidates],
        }

    # Best single model baseline
    best_single = candidates[0]
    best_metric = best_single.get("metrics", {}).get(primary_metric, 0)

    # Try each ensemble method
    results = []
    results.append({
        "method": "best_single",
        "metric_value": best_metric,
        "delta": 0.0,
        "experiment_id": best_single.get("experiment_id"),
    })

    if predictions and labels is not None:
        weights = [
            e.get("metrics", {}).get(primary_metric, 0)
            for e in candidates[:len(predictions)]
        ]

        for method in methods:
            result = _try_method(
                method, predictions, labels, weights, task_type, primary_metric,
            )
            if result:
                result["delta"] = round(result.get("metric_value", 0) - best_metric, 6)
                results.append(result)

    # Find best ensemble
    if lower_is_better:
        best_result = min(results, key=lambda r: r.get("metric_value", float("inf")))
    else:
        best_result = max(results, key=lambda r: r.get("metric_value", float("-inf")))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": primary_metric,
        "task_type": task_type,
        "n_candidates": len(candidates),
        "base_models": [
            {
                "experiment_id": e.get("experiment_id"),
                "model_type": e.get("config", {}).get("model_type", "?"),
                primary_metric: e.get("metrics", {}).get(primary_metric),
            }
            for e in candidates
        ],
        "results": results,
        "best_method": best_result.get("method"),
        "best_metric": best_result.get("metric_value"),
        "improvement": best_result.get("delta", 0),
        "diversity": diversity,
    }

    return report


def _try_method(
    method: str,
    predictions: list[np.ndarray],
    labels: np.ndarray,
    weights: list[float],
    task_type: str,
    primary_metric: str,
) -> dict | None:
    """Try a single ensemble method and return results."""
    try:
        if method == "voting":
            preds = voting_ensemble(predictions, task_type)
            eval_result = evaluate_ensemble(preds, labels, task_type)
            metric_val = eval_result.get("accuracy", eval_result.get("rmse", 0))
            return {"method": "voting", "metric_value": metric_val, "details": eval_result}

        elif method == "weighted_voting":
            preds = weighted_voting_ensemble(predictions, weights, task_type)
            eval_result = evaluate_ensemble(preds, labels, task_type)
            metric_val = eval_result.get("accuracy", eval_result.get("rmse", 0))
            return {"method": "weighted_voting", "metric_value": metric_val, "details": eval_result, "weights": [round(w / sum(weights), 4) for w in weights]}

        elif method == "stacking":
            result = stacking_ensemble(predictions, labels, task_type)
            if len(result["meta_predictions"]) > 0:
                eval_result = evaluate_ensemble(result["meta_predictions"], labels, task_type)
                metric_val = eval_result.get("accuracy", eval_result.get("rmse", 0))
                return {"method": "stacking", "metric_value": metric_val, "details": eval_result, "meta_weights": result["meta_weights"]}

        elif method == "blending":
            result = blending_ensemble(predictions, labels, task_type)
            if len(result["meta_predictions"]) > 0 and result.get("holdout_labels") is not None:
                eval_result = evaluate_ensemble(result["meta_predictions"], result["holdout_labels"], task_type)
                metric_val = eval_result.get("accuracy", eval_result.get("rmse", 0))
                return {"method": "blending", "metric_value": metric_val, "details": eval_result, "holdout_size": result["holdout_size"]}

    except Exception:
        pass

    return None


def _load_predictions(
    experiments: list[dict],
    predictions_dir: str,
) -> list[np.ndarray]:
    """Load saved predictions for experiments."""
    preds_path = Path(predictions_dir)
    predictions = []

    for exp in experiments:
        exp_id = exp.get("experiment_id", "")
        pred_file = preds_path / f"{exp_id}-predictions.npy"
        if pred_file.exists():
            predictions.append(np.load(pred_file))
        else:
            # Try CSV fallback
            csv_file = preds_path / f"{exp_id}-predictions.csv"
            if csv_file.exists():
                predictions.append(np.loadtxt(csv_file, delimiter=","))

    return predictions


def _load_labels(predictions_dir: str) -> np.ndarray | None:
    """Load ground truth labels."""
    preds_path = Path(predictions_dir)
    labels_file = preds_path / "labels.npy"
    if labels_file.exists():
        return np.load(labels_file)
    csv_file = preds_path / "labels.csv"
    if csv_file.exists():
        return np.loadtxt(csv_file, delimiter=",")
    return None


# --- Report Formatting ---


def save_ensemble_report(report: dict, output_dir: str = "experiments/ensembles") -> Path:
    """Save ensemble report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filepath = out_path / f"ensemble-{timestamp}.yaml"

    # Convert numpy types for YAML serialization
    clean = json.loads(json.dumps(report, default=str))
    with open(filepath, "w") as f:
        yaml.dump(clean, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_ensemble_report(report: dict) -> str:
    """Format ensemble report as human-readable markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    lines = [
        "# Ensemble Results",
        "",
        f"*Generated {report.get('generated_at', 'N/A')[:19]}*",
        "",
        f"**Task type:** {report.get('task_type', '?')}",
        f"**Primary metric:** {report.get('primary_metric', '?')}",
        f"**Base models:** {report.get('n_candidates', 0)}",
        "",
    ]

    # Base models
    lines.extend(["## Base Models", ""])
    base = report.get("base_models", [])
    if base:
        metric = report.get("primary_metric", "metric")
        lines.append(f"| Experiment | Model Type | {metric} |")
        lines.append("|------------|------------|--------|")
        for m in base:
            val = m.get(metric, "N/A")
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            lines.append(f"| {m.get('experiment_id', '?')} | {m.get('model_type', '?')} | {val_str} |")
        lines.append("")

    # Results table
    results = report.get("results", [])
    if results:
        metric = report.get("primary_metric", "metric")
        lines.extend(["## Ensemble Comparison", ""])
        lines.append(f"| Method | {metric} | Delta vs Best Single |")
        lines.append("|--------|--------|---------------------|")
        best_method = report.get("best_method")
        for r in results:
            val = r.get("metric_value", "N/A")
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            delta = r.get("delta", 0)
            delta_str = f"{delta:+.4f}" if isinstance(delta, (int, float)) else "—"
            marker = " BEST" if r.get("method") == best_method and r.get("method") != "best_single" else ""
            lines.append(f"| {r.get('method', '?')} | {val_str} | {delta_str} |{marker}")
        lines.append("")

    # Improvement summary
    improvement = report.get("improvement", 0)
    best = report.get("best_method", "best_single")
    if best != "best_single" and improvement > 0:
        lines.extend([
            "## Summary",
            "",
            f"**Best ensemble ({best}) improves over best single model by {improvement:+.4f}**",
            "",
        ])
    elif best == "best_single":
        lines.extend([
            "## Summary",
            "",
            "No ensemble method improved over the best single model.",
            "Consider training more diverse models before ensembling.",
            "",
        ])

    # Diversity
    diversity = report.get("diversity", {})
    if diversity.get("mean_correlation") is not None:
        lines.extend([
            "## Diversity Analysis",
            "",
            f"**Mean prediction correlation:** {diversity['mean_correlation']:.3f}",
        ])
        if diversity["mean_correlation"] > 0.9:
            lines.append("*High correlation — models are very similar. Diversity would help.*")
        elif diversity["mean_correlation"] < 0.5:
            lines.append("*Good diversity — models complement each other well.*")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automated ensemble construction",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of top models to include (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--methods", default=",".join(DEFAULT_METHODS),
        help=f"Ensemble methods to try (default: {','.join(DEFAULT_METHODS)})",
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
        "--predictions-dir", default="experiments/predictions",
        help="Directory containing saved predictions",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    report = build_ensemble(
        top_k=args.top_k,
        methods=methods,
        config_path=args.config,
        log_path=args.log,
        predictions_dir=args.predictions_dir,
    )

    if "error" not in report:
        filepath = save_ensemble_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_ensemble_report(report))


if __name__ == "__main__":
    main()
