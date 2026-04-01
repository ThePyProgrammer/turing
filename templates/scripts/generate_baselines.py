#!/usr/bin/env python3
"""Automatic baseline generation for the autoresearch pipeline.

Auto-generates trivial baselines (majority, mean, random, linear, k-NN)
so every experiment has a "is this better than dumb?" reference point.

Usage:
    python scripts/generate_baselines.py
    python scripts/generate_baselines.py --methods all
    python scripts/generate_baselines.py --methods simple
    python scripts/generate_baselines.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"

# Baseline method groups
SIMPLE_METHODS = ["random", "majority_or_mean"]
LINEAR_METHODS = ["linear"]
ALL_METHODS = ["random", "majority_or_mean", "stratified_or_median", "linear", "knn"]


# --- Baseline Methods ---


def random_baseline(y: np.ndarray, task_type: str = "classification") -> np.ndarray:
    """Random predictions."""
    n = len(y)
    if task_type == "classification":
        classes = np.unique(y)
        return np.random.choice(classes, size=n)
    else:
        return np.random.uniform(np.min(y), np.max(y), size=n)


def majority_or_mean_baseline(y: np.ndarray, task_type: str = "classification") -> np.ndarray:
    """Majority class (classification) or mean (regression)."""
    n = len(y)
    if task_type == "classification":
        from scipy import stats as scipy_stats
        mode_result = scipy_stats.mode(y, keepdims=False)
        majority = mode_result.mode
        return np.full(n, majority)
    else:
        return np.full(n, np.mean(y))


def stratified_or_median_baseline(y: np.ndarray, task_type: str = "classification") -> np.ndarray:
    """Stratified random (classification) or median (regression)."""
    n = len(y)
    if task_type == "classification":
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return np.random.choice(classes, size=n, p=probs)
    else:
        return np.full(n, np.median(y))


def linear_baseline(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = "classification",
) -> dict:
    """Linear model baseline (LogisticRegression / Ridge)."""
    from sklearn.linear_model import LogisticRegression, Ridge

    n_samples = X.shape[0]
    split = int(n_samples * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if task_type == "classification":
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
    else:
        model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "predictions": predictions,
        "labels": y_test,
        "model_name": "LogisticRegression" if task_type == "classification" else "Ridge",
    }


def knn_baseline(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str = "classification",
    n_neighbors: int = 5,
) -> dict:
    """k-NN baseline."""
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    n_samples = X.shape[0]
    split = int(n_samples * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    k = min(n_neighbors, len(X_train))
    if task_type == "classification":
        model = KNeighborsClassifier(n_neighbors=k)
    else:
        model = KNeighborsRegressor(n_neighbors=k)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "predictions": predictions,
        "labels": y_test,
        "model_name": f"k-NN (k={k})",
    }


# --- Evaluation ---


def evaluate_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    task_type: str = "classification",
    primary_metric: str = "accuracy",
) -> dict:
    """Evaluate baseline predictions."""
    min_len = min(len(predictions), len(labels))
    predictions = predictions[:min_len]
    labels = labels[:min_len]

    if task_type == "classification":
        accuracy = float(np.mean(predictions == labels))
        return {"accuracy": round(accuracy, 6), "n_samples": min_len}
    else:
        mse = float(np.mean((predictions - labels) ** 2))
        rmse = float(np.sqrt(mse))
        return {"mse": round(mse, 6), "rmse": round(rmse, 6), "n_samples": min_len}


# --- Full Pipeline ---


def generate_baselines(
    methods: str = "all",
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    data_path: str | None = None,
) -> dict:
    """Generate baseline results.

    Args:
        methods: Method group (all, simple, linear).
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        data_path: Path to data (optional, for linear/knn).

    Returns:
        Baseline report dict.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    task_type = config.get("task", {}).get("type", "classification")

    experiments = load_experiments(log_path)

    # Find current best for comparison
    kept = [e for e in experiments if e.get("status") == "kept"]
    current_best_value = None
    if kept:
        best = max(kept, key=lambda e: e.get("metrics", {}).get(primary_metric, 0))
        current_best_value = best.get("metrics", {}).get(primary_metric)

    # Select methods
    if methods == "simple":
        method_list = SIMPLE_METHODS
    elif methods == "linear":
        method_list = LINEAR_METHODS
    else:
        method_list = ALL_METHODS

    # For methods that need data, check if data is available
    has_data = data_path is not None and Path(data_path).exists()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type,
        "primary_metric": primary_metric,
        "methods_requested": methods,
        "baselines": [],
        "current_best": current_best_value,
        "data_available": has_data,
    }

    # Generate synthetic labels for demo if no data
    # In real use, evaluate.py would provide these
    if not has_data:
        report["note"] = "No data loaded — baseline plan generated. Run with --data to compute actual scores."
        for method in method_list:
            report["baselines"].append({
                "method": _method_display_name(method, task_type),
                "metric_value": None,
                "notes": "Requires data",
            })
        return report

    # Load data
    try:
        data = np.load(data_path, allow_pickle=True)
        X = data.get("X", data.get("features"))
        y = data.get("y", data.get("labels", data.get("target")))
        if X is None or y is None:
            return {"error": f"Data file {data_path} missing X/y arrays"}
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    # Run baselines
    for method in method_list:
        result = _run_baseline(method, X, y, task_type, primary_metric)
        report["baselines"].append(result)

    # Add current best for comparison
    if current_best_value is not None:
        report["baselines"].append({
            "method": "Current best",
            "metric_value": current_best_value,
            "notes": "",
        })

    # Compute improvement over linear baseline
    linear_result = next((b for b in report["baselines"] if "linear" in b.get("method", "").lower()), None)
    if linear_result and linear_result.get("metric_value") and current_best_value:
        improvement = current_best_value - linear_result["metric_value"]
        report["improvement_over_linear"] = round(improvement, 6)

    return report


def _method_display_name(method: str, task_type: str) -> str:
    """Human-readable method name."""
    names = {
        "random": "Random",
        "majority_or_mean": "Majority class" if task_type == "classification" else "Mean predictor",
        "stratified_or_median": "Stratified random" if task_type == "classification" else "Median predictor",
        "linear": "Logistic Regression" if task_type == "classification" else "Ridge Regression",
        "knn": "k-NN (k=5)",
    }
    return names.get(method, method)


def _run_baseline(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    primary_metric: str,
) -> dict:
    """Run a single baseline method."""
    try:
        if method == "random":
            preds = random_baseline(y, task_type)
            eval_result = evaluate_predictions(preds, y, task_type, primary_metric)
            return {
                "method": "Random",
                "metric_value": eval_result.get(primary_metric, eval_result.get("accuracy", eval_result.get("rmse"))),
                "notes": "Floor — below this = bug",
            }

        elif method == "majority_or_mean":
            preds = majority_or_mean_baseline(y, task_type)
            eval_result = evaluate_predictions(preds, y, task_type, primary_metric)
            name = "Majority class" if task_type == "classification" else "Mean predictor"
            return {
                "method": name,
                "metric_value": eval_result.get(primary_metric, eval_result.get("accuracy", eval_result.get("rmse"))),
                "notes": "Naive floor",
            }

        elif method == "stratified_or_median":
            preds = stratified_or_median_baseline(y, task_type)
            eval_result = evaluate_predictions(preds, y, task_type, primary_metric)
            name = "Stratified random" if task_type == "classification" else "Median predictor"
            return {
                "method": name,
                "metric_value": eval_result.get(primary_metric, eval_result.get("accuracy", eval_result.get("rmse"))),
                "notes": "",
            }

        elif method == "linear":
            result = linear_baseline(X, y, task_type)
            eval_result = evaluate_predictions(result["predictions"], result["labels"], task_type, primary_metric)
            return {
                "method": result["model_name"],
                "metric_value": eval_result.get(primary_metric, eval_result.get("accuracy", eval_result.get("rmse"))),
                "notes": "Linear ceiling",
            }

        elif method == "knn":
            result = knn_baseline(X, y, task_type)
            eval_result = evaluate_predictions(result["predictions"], result["labels"], task_type, primary_metric)
            return {
                "method": result["model_name"],
                "metric_value": eval_result.get(primary_metric, eval_result.get("accuracy", eval_result.get("rmse"))),
                "notes": "Non-parametric reference",
            }

    except Exception as e:
        return {"method": method, "metric_value": None, "notes": f"Error: {e}"}

    return {"method": method, "metric_value": None, "notes": "Unknown method"}


# --- Report Formatting ---


def save_baseline_report(report: dict, output_dir: str = "experiments/baselines") -> Path:
    """Save baseline report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = out_path / f"baselines-{date}.yaml"

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_baseline_report(report: dict) -> str:
    """Format baseline report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    task = report.get("task_type", "?")
    metric = report.get("primary_metric", "metric")

    lines = [
        f"# Baselines for {task} ({metric})",
        "",
        f"*Generated {report.get('generated_at', 'N/A')[:19]}*",
        "",
    ]

    baselines = report.get("baselines", [])
    if baselines:
        lines.append(f"| Method | {metric} | Notes |")
        lines.append("|--------|--------|-------|")
        for b in baselines:
            val = b.get("metric_value")
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val or "N/A")
            lines.append(f"| {b.get('method', '?')} | {val_str} | {b.get('notes', '')} |")
        lines.append("")

    improvement = report.get("improvement_over_linear")
    if improvement is not None:
        lines.append(f"**Your model beats the linear baseline by {improvement:+.4f} ({improvement / report.get('current_best', 1) * 100:.1f}%)**")
        lines.append("")

    if report.get("note"):
        lines.append(f"*{report['note']}*")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automatic baseline generation",
    )
    parser.add_argument(
        "--methods", choices=["all", "simple", "linear"], default="all",
        help="Baseline method group (default: all)",
    )
    parser.add_argument(
        "--data",
        help="Path to data file (.npz with X and y arrays)",
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
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    report = generate_baselines(
        methods=args.methods,
        config_path=args.config,
        log_path=args.log,
        data_path=args.data,
    )

    if "error" not in report:
        filepath = save_baseline_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_baseline_report(report))


if __name__ == "__main__":
    main()
