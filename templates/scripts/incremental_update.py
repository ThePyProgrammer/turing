#!/usr/bin/env python3
"""Incremental model update for the autoresearch pipeline.

Updates the existing best model with new data without retraining from
scratch. For tree models: add boosting rounds. For neural nets: fine-tune
with replay buffer. For sklearn: partial_fit or warm_start.

Tracks accuracy on both old and new data to detect catastrophic forgetting.

Usage:
    python scripts/incremental_update.py exp-089 --new-data data/new_batch.csv
    python scripts/incremental_update.py exp-089 --new-data data/new.csv --replay-ratio 0.1
    python scripts/incremental_update.py exp-089 --new-data data/new.csv --tolerance 0.005
    python scripts/incremental_update.py --json
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
DEFAULT_REPLAY_RATIO = 0.1
DEFAULT_FORGETTING_TOLERANCE = 0.005
DEFAULT_NEW_ROUNDS = 50

# Model type detection
TREE_MODELS = {"xgboost", "lightgbm", "catboost", "gradient_boosting", "gbm"}
NEURAL_MODELS = {"mlp", "neural_network", "nn", "pytorch", "tensorflow", "keras", "transformer"}
SKLEARN_PARTIAL_FIT = {
    "sgd", "passive_aggressive", "perceptron", "multinomial_nb",
    "bernoulli_nb", "minibatch_kmeans",
}
SKLEARN_WARM_START = {
    "random_forest", "gradient_boosting", "bagging", "adaboost",
}


# --- Model Type Detection ---


def detect_model_type(experiment: dict) -> str:
    """Detect model type category from an experiment.

    Returns one of: 'tree', 'neural', 'sklearn_partial', 'sklearn_warm', 'unknown'.
    """
    config = experiment.get("config", {})
    model_type = config.get("model_type", "").lower()

    if any(t in model_type for t in TREE_MODELS):
        return "tree"
    if any(t in model_type for t in NEURAL_MODELS):
        return "neural"
    if any(t in model_type for t in SKLEARN_PARTIAL_FIT):
        return "sklearn_partial"
    if any(t in model_type for t in SKLEARN_WARM_START):
        return "sklearn_warm"

    # Check hyperparams for hints
    hyperparams = config.get("hyperparams", {})
    if "n_estimators" in hyperparams and ("max_depth" in hyperparams or "num_leaves" in hyperparams):
        return "tree"
    if "hidden_size" in hyperparams or "layers" in hyperparams:
        return "neural"

    return "unknown"


# --- Update Strategies ---


def plan_tree_update(
    experiment: dict,
    new_data_size: int,
    new_rounds: int = DEFAULT_NEW_ROUNDS,
) -> dict:
    """Plan incremental update for tree-based models.

    XGBoost/LightGBM support continued boosting with new data.
    """
    config = experiment.get("config", {})
    hyperparams = config.get("hyperparams", {})
    current_rounds = hyperparams.get("n_estimators", hyperparams.get("num_boost_round", 100))

    return {
        "strategy": "continued_boosting",
        "model_type": "tree",
        "current_rounds": current_rounds,
        "additional_rounds": new_rounds,
        "total_rounds": current_rounds + new_rounds,
        "new_data_size": new_data_size,
        "method": "xgb_model parameter for continued training",
        "instructions": [
            f"Load model from {experiment.get('experiment_id', 'exp-???')}",
            f"Set xgb_model/init_model to loaded model",
            f"Train for {new_rounds} additional rounds on new data",
            "Evaluate on old validation set + new data",
        ],
    }


def plan_neural_update(
    experiment: dict,
    new_data_size: int,
    replay_ratio: float = DEFAULT_REPLAY_RATIO,
) -> dict:
    """Plan incremental update for neural network models.

    Fine-tune on new data + replay buffer from old data.
    """
    config = experiment.get("config", {})
    hyperparams = config.get("hyperparams", {})
    original_lr = hyperparams.get("learning_rate", hyperparams.get("lr", 0.001))

    replay_size = int(new_data_size * replay_ratio)

    return {
        "strategy": "fine_tune_with_replay",
        "model_type": "neural",
        "original_lr": original_lr,
        "fine_tune_lr": original_lr * 0.1,
        "new_data_size": new_data_size,
        "replay_size": replay_size,
        "replay_ratio": replay_ratio,
        "total_training_size": new_data_size + replay_size,
        "method": "Load weights, reduce LR by 10x, mix new data with replay buffer",
        "instructions": [
            f"Load model weights from {experiment.get('experiment_id', 'exp-???')}",
            f"Reduce learning rate to {original_lr * 0.1}",
            f"Sample {replay_size} examples from old training data (replay buffer)",
            f"Train on {new_data_size} new + {replay_size} replay samples",
            "Evaluate on old validation set + new data",
        ],
    }


def plan_sklearn_update(
    experiment: dict,
    new_data_size: int,
    model_category: str,
) -> dict:
    """Plan incremental update for scikit-learn models."""
    if model_category == "sklearn_partial":
        return {
            "strategy": "partial_fit",
            "model_type": "sklearn",
            "new_data_size": new_data_size,
            "method": "Call partial_fit() with new data batch",
            "instructions": [
                f"Load model from {experiment.get('experiment_id', 'exp-???')}",
                "Call model.partial_fit(X_new, y_new)",
                "Evaluate on old validation set + new data",
            ],
        }
    else:
        return {
            "strategy": "warm_start_retrain",
            "model_type": "sklearn",
            "new_data_size": new_data_size,
            "method": "Set warm_start=True, retrain on combined old+new data",
            "instructions": [
                f"Load model from {experiment.get('experiment_id', 'exp-???')}",
                "Set warm_start=True",
                "Fit on combined old + new data",
                "Evaluate on old validation set + new data",
            ],
        }


def plan_update(
    experiment: dict,
    new_data_size: int,
    replay_ratio: float = DEFAULT_REPLAY_RATIO,
    new_rounds: int = DEFAULT_NEW_ROUNDS,
) -> dict:
    """Plan the update strategy based on model type.

    Args:
        experiment: The experiment to update.
        new_data_size: Number of new data samples.
        replay_ratio: Fraction of old data to replay (neural nets).
        new_rounds: Additional boosting rounds (tree models).

    Returns:
        Update plan with strategy, instructions, and parameters.
    """
    model_category = detect_model_type(experiment)

    if model_category == "tree":
        return plan_tree_update(experiment, new_data_size, new_rounds)
    elif model_category == "neural":
        return plan_neural_update(experiment, new_data_size, replay_ratio)
    elif model_category in ("sklearn_partial", "sklearn_warm"):
        return plan_sklearn_update(experiment, new_data_size, model_category)
    else:
        return {
            "strategy": "unknown",
            "model_type": "unknown",
            "error": "Cannot determine model type for incremental update",
            "suggestion": "Add model_type to config (e.g., 'xgboost', 'lightgbm', 'mlp')",
        }


# --- Forgetting Detection ---


def check_forgetting(
    old_metrics: dict[str, float],
    new_metrics: dict[str, float],
    primary_metric: str,
    tolerance: float = DEFAULT_FORGETTING_TOLERANCE,
    lower_is_better: bool = False,
) -> dict:
    """Check for catastrophic forgetting after incremental update.

    Compares old data metrics before and after update.

    Args:
        old_metrics: Metrics on old validation data BEFORE update.
        new_metrics: Metrics on old validation data AFTER update.
        primary_metric: Primary metric name.
        tolerance: Maximum allowed degradation.
        lower_is_better: Whether lower metric is better.

    Returns:
        Forgetting check result with verdict and details.
    """
    old_val = old_metrics.get(primary_metric)
    new_val = new_metrics.get(primary_metric)

    if old_val is None or new_val is None:
        return {
            "verdict": "UNKNOWN",
            "reason": f"Missing {primary_metric} in old or new metrics",
            "old_value": old_val,
            "new_value": new_val,
        }

    if lower_is_better:
        degradation = new_val - old_val  # Positive means worse
    else:
        degradation = old_val - new_val  # Positive means worse

    if degradation <= 0:
        verdict = "PASS"
        reason = "No degradation on old data"
    elif degradation <= tolerance:
        verdict = "PASS"
        reason = f"Degradation {degradation:.4f} within tolerance {tolerance}"
    elif degradation <= tolerance * 2:
        verdict = "WARNING"
        reason = f"Degradation {degradation:.4f} exceeds tolerance {tolerance} but within 2x"
    else:
        verdict = "FAIL"
        reason = f"Catastrophic forgetting: degradation {degradation:.4f} >> tolerance {tolerance}"

    return {
        "verdict": verdict,
        "reason": reason,
        "primary_metric": primary_metric,
        "old_value": round(float(old_val), 6),
        "new_value": round(float(new_val), 6),
        "degradation": round(float(degradation), 6),
        "tolerance": tolerance,
        "within_tolerance": degradation <= tolerance,
    }


# --- Update Report ---


def build_update_report(
    experiment: dict,
    plan: dict,
    old_data_metrics_before: dict[str, float] | None = None,
    old_data_metrics_after: dict[str, float] | None = None,
    new_data_metrics: dict[str, float] | None = None,
    combined_metrics: dict[str, float] | None = None,
    primary_metric: str = "accuracy",
    tolerance: float = DEFAULT_FORGETTING_TOLERANCE,
    lower_is_better: bool = False,
    update_time_seconds: float | None = None,
    full_retrain_time_seconds: float | None = None,
) -> dict:
    """Build a complete update report.

    Args:
        experiment: Original experiment.
        plan: Update plan from plan_update().
        old_data_metrics_before: Metrics on old data before update.
        old_data_metrics_after: Metrics on old data after update.
        new_data_metrics: Metrics on new data after update.
        combined_metrics: Metrics on combined old+new data after update.
        primary_metric: Primary metric name.
        tolerance: Forgetting tolerance.
        lower_is_better: Whether lower metric is better.
        update_time_seconds: Time for incremental update.
        full_retrain_time_seconds: Estimated time for full retrain.

    Returns:
        Complete update report.
    """
    exp_id = experiment.get("experiment_id", "unknown")

    # Forgetting check
    forgetting = None
    if old_data_metrics_before and old_data_metrics_after:
        forgetting = check_forgetting(
            old_data_metrics_before, old_data_metrics_after,
            primary_metric, tolerance, lower_is_better,
        )

    # Metric comparison table
    metric_table = []
    if old_data_metrics_before and old_data_metrics_after:
        before_val = old_data_metrics_before.get(primary_metric)
        after_val = old_data_metrics_after.get(primary_metric)
        if before_val is not None and after_val is not None:
            delta = after_val - before_val
            metric_table.append({
                "dataset": "Old data",
                "before": round(float(before_val), 4),
                "after": round(float(after_val), 4),
                "delta": round(float(delta), 4),
            })

    if new_data_metrics:
        new_val = new_data_metrics.get(primary_metric)
        if new_val is not None:
            metric_table.append({
                "dataset": "New data",
                "before": None,
                "after": round(float(new_val), 4),
                "delta": None,
            })

    if combined_metrics and old_data_metrics_before:
        combined_val = combined_metrics.get(primary_metric)
        before_val = old_data_metrics_before.get(primary_metric)
        if combined_val is not None and before_val is not None:
            delta = combined_val - before_val
            metric_table.append({
                "dataset": "Combined",
                "before": round(float(before_val), 4),
                "after": round(float(combined_val), 4),
                "delta": round(float(delta), 4),
            })

    # Speedup
    speedup = None
    if update_time_seconds and full_retrain_time_seconds and full_retrain_time_seconds > 0:
        speedup = round(full_retrain_time_seconds / update_time_seconds, 1)

    return {
        "experiment_id": exp_id,
        "parent_experiment": exp_id,
        "family": "update",
        "plan": plan,
        "metric_table": metric_table,
        "forgetting_check": forgetting,
        "update_time_seconds": update_time_seconds,
        "full_retrain_time_seconds": full_retrain_time_seconds,
        "speedup": speedup,
        "verdict": forgetting["verdict"] if forgetting else "PENDING",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# --- Full Pipeline ---


def incremental_update(
    exp_id: str,
    new_data_path: str | None = None,
    new_data_size: int | None = None,
    replay_ratio: float = DEFAULT_REPLAY_RATIO,
    new_rounds: int = DEFAULT_NEW_ROUNDS,
    tolerance: float = DEFAULT_FORGETTING_TOLERANCE,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Plan and report an incremental model update.

    Args:
        exp_id: Experiment ID to update.
        new_data_path: Path to new data file.
        new_data_size: Number of new samples (auto-detected from file if available).
        replay_ratio: Fraction of old data for replay buffer.
        new_rounds: Additional boosting rounds for tree models.
        tolerance: Forgetting tolerance.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.

    Returns:
        Update report with plan, instructions, and placeholder for results.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)
    experiment = None
    for exp in experiments:
        if exp.get("experiment_id") == exp_id:
            experiment = exp
            break

    if experiment is None:
        return {"error": f"Experiment {exp_id} not found in log"}

    # Determine new data size
    if new_data_size is None and new_data_path:
        new_data_size = _count_data_samples(new_data_path)
    if new_data_size is None:
        new_data_size = 0

    if new_data_size == 0 and new_data_path is None:
        return {"error": "No new data provided. Use --new-data <path> or --new-data-size <N>"}

    plan = plan_update(experiment, new_data_size, replay_ratio, new_rounds)

    if "error" in plan:
        return {
            "experiment_id": exp_id,
            "error": plan["error"],
            "suggestion": plan.get("suggestion"),
        }

    # Build report (actual metrics filled in after execution)
    report = build_update_report(
        experiment=experiment,
        plan=plan,
        primary_metric=primary_metric,
        tolerance=tolerance,
        lower_is_better=lower_is_better,
    )

    return report


def _count_data_samples(path: str) -> int:
    """Count samples in a data file (CSV/JSONL)."""
    p = Path(path)
    if not p.exists():
        return 0
    try:
        with open(p) as f:
            count = sum(1 for _ in f)
        # Subtract header for CSV
        if p.suffix == ".csv" and count > 0:
            count -= 1
        return max(count, 0)
    except (OSError, UnicodeDecodeError):
        return 0


# --- Report Formatting ---


def save_update_report(report: dict, output_dir: str = "experiments/updates") -> Path:
    """Save update report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("experiment_id", "unknown")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filepath = out_path / f"{exp_id}-update-{ts}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_update_report(report: dict) -> str:
    """Format update report as readable markdown."""
    if "error" in report:
        lines = [f"ERROR: {report['error']}"]
        if "suggestion" in report:
            lines.append(f"\n{report['suggestion']}")
        return "\n".join(lines)

    lines = ["# Incremental Update Report", ""]
    lines.append(f"**Experiment:** {report.get('experiment_id', 'N/A')}")

    plan = report.get("plan", {})
    lines.append(f"**Strategy:** {plan.get('strategy', 'N/A')}")
    lines.append(f"**Model type:** {plan.get('model_type', 'N/A')}")
    lines.append("")

    # Instructions
    instructions = plan.get("instructions", [])
    if instructions:
        lines.append("**Steps:**")
        for step in instructions:
            lines.append(f"1. {step}")
        lines.append("")

    # Metric table
    metric_table = report.get("metric_table", [])
    if metric_table:
        lines.append("| Dataset | Before | After | Delta |")
        lines.append("|---------|--------|-------|-------|")
        for row in metric_table:
            before = f"{row['before']:.4f}" if row["before"] is not None else "—"
            after = f"{row['after']:.4f}" if row["after"] is not None else "—"
            delta = f"{row['delta']:+.4f}" if row["delta"] is not None else "(first)"
            lines.append(f"| {row['dataset']} | {before} | {after} | {delta} |")
        lines.append("")

    # Forgetting check
    forgetting = report.get("forgetting_check")
    if forgetting:
        lines.append(f"**Forgetting check:** {forgetting['verdict']}")
        lines.append(f"**Reason:** {forgetting['reason']}")
        lines.append("")

    # Speedup
    speedup = report.get("speedup")
    if speedup:
        update_time = report.get("update_time_seconds", 0)
        retrain_time = report.get("full_retrain_time_seconds", 0)
        lines.append(f"**Update time:** {update_time:.0f}s (vs {retrain_time:.0f}s full retrain, {speedup}x faster)")
    elif report.get("verdict") == "PENDING":
        lines.append("**Status:** Plan generated — run the update to get metrics")

    lines.append("")
    lines.append(f"*Generated: {report.get('generated_at', 'N/A')}*")
    return "\n".join(lines)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Incremental model update — add new data without full retraining"
    )
    parser.add_argument("exp_id", nargs="?", help="Experiment ID to update")
    parser.add_argument("--new-data", help="Path to new data file")
    parser.add_argument("--new-data-size", type=int, help="Number of new samples")
    parser.add_argument("--replay-ratio", type=float, default=DEFAULT_REPLAY_RATIO,
                        help="Replay buffer ratio for neural nets")
    parser.add_argument("--new-rounds", type=int, default=DEFAULT_NEW_ROUNDS,
                        help="Additional boosting rounds for tree models")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_FORGETTING_TOLERANCE,
                        help="Max allowed metric degradation on old data")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    if not args.exp_id:
        parser.error("Please provide an experiment ID")

    report = incremental_update(
        exp_id=args.exp_id,
        new_data_path=args.new_data,
        new_data_size=args.new_data_size,
        replay_ratio=args.replay_ratio,
        new_rounds=args.new_rounds,
        tolerance=args.tolerance,
        config_path=args.config,
        log_path=args.log,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_update_report(report))

    if "error" not in report:
        saved = save_update_report(report)
        if not args.json:
            print(f"\nSaved: {saved}")


if __name__ == "__main__":
    main()
