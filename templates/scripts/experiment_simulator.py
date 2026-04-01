#!/usr/bin/env python3
"""Experiment outcome simulator for the autoresearch pipeline.

Predicts experiment outcomes from prior data using a surrogate model.
Pre-filters experiment configs to save budget — only run the ones
predicted to beat the current best.

Usage:
    python scripts/experiment_simulator.py --configs configs.yaml
    python scripts/experiment_simulator.py --top-k 5
    python scripts/experiment_simulator.py --threshold 0.001
    python scripts/experiment_simulator.py --json
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
DEFAULT_TOP_K = 5
DEFAULT_IMPROVEMENT_THRESHOLD = 0.0
NOVELTY_PENALTY_FACTOR = 0.1
MIN_HISTORY_FOR_SURROGATE = 5


# --- Feature Extraction ---


def extract_config_features(config: dict) -> dict[str, float]:
    """Extract numeric features from an experiment config.

    Flattens nested config into a flat dict of numeric values.
    """
    features = {}
    _flatten(config, features, prefix="")
    return features


def _flatten(obj: dict, out: dict, prefix: str) -> None:
    """Recursively flatten a dict, keeping only numeric values."""
    for key, val in obj.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[full_key] = float(val)
        elif isinstance(val, dict):
            _flatten(val, out, full_key)


def experiment_to_features(exp: dict) -> dict[str, float]:
    """Extract feature vector from an experiment log entry."""
    features = {}

    # Extract from config
    config = exp.get("config", {})
    if isinstance(config, dict):
        hyperparams = config.get("hyperparams", config.get("model", {}).get("hyperparams", {}))
        if isinstance(hyperparams, dict):
            for k, v in hyperparams.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    features[k] = float(v)

    # Also check top-level numeric fields
    for key in ("learning_rate", "lr", "max_depth", "depth", "n_estimators",
                "epochs", "batch_size", "hidden_size", "dropout"):
        val = exp.get(key) or config.get(key)
        if isinstance(val, (int, float)):
            features[key] = float(val)

    return features


# --- Surrogate Model ---


def build_surrogate(
    experiments: list[dict],
    primary_metric: str,
) -> dict:
    """Build a simple surrogate model from experiment history.

    Uses a weighted k-NN approach: for a new config, predict the metric
    as a weighted average of the k nearest experiments in config space.

    Returns:
        Surrogate model dict with training data and feature info.
    """
    data_points = []
    for exp in experiments:
        metric = exp.get("metrics", {}).get(primary_metric)
        if metric is None:
            continue
        features = experiment_to_features(exp)
        if features:
            data_points.append({"features": features, "metric": metric})

    if len(data_points) < MIN_HISTORY_FOR_SURROGATE:
        return {
            "status": "insufficient",
            "n_points": len(data_points),
            "min_required": MIN_HISTORY_FOR_SURROGATE,
        }

    # Collect all feature names
    all_features = set()
    for dp in data_points:
        all_features.update(dp["features"].keys())

    return {
        "status": "ready",
        "data_points": data_points,
        "feature_names": sorted(all_features),
        "n_points": len(data_points),
    }


def predict_with_surrogate(
    surrogate: dict,
    config_features: dict[str, float],
    k: int = 3,
) -> dict:
    """Predict metric for a config using weighted k-NN surrogate.

    Args:
        surrogate: Built surrogate model.
        config_features: Feature dict for the config to predict.
        k: Number of nearest neighbors.

    Returns:
        Prediction dict with predicted metric and uncertainty.
    """
    if surrogate.get("status") != "ready":
        return {"error": "Surrogate not ready", "predicted": None, "uncertainty": None}

    data_points = surrogate["data_points"]
    feature_names = surrogate["feature_names"]

    # Compute distances
    distances = []
    for dp in data_points:
        dist = _config_distance(config_features, dp["features"], feature_names)
        distances.append((dist, dp["metric"]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    if not neighbors:
        return {"error": "No neighbors found", "predicted": None, "uncertainty": None}

    # Weighted average (inverse distance weighting)
    metrics = [m for _, m in neighbors]
    dists = [d for d, _ in neighbors]

    if all(d == 0 for d in dists):
        predicted = np.mean(metrics)
        uncertainty = 0.0
    else:
        weights = [1.0 / (d + 1e-6) for d in dists]
        total_weight = sum(weights)
        predicted = sum(w * m for w, m in zip(weights, metrics)) / total_weight
        uncertainty = float(np.std(metrics))

    # Novelty penalty: discount if far from training distribution
    min_dist = dists[0] if dists else 0
    avg_dist = np.mean([d for d, _ in distances]) if distances else 1
    novelty = min_dist / avg_dist if avg_dist > 0 else 0
    novelty_penalty = novelty * NOVELTY_PENALTY_FACTOR

    return {
        "predicted": round(float(predicted - novelty_penalty), 6),
        "uncertainty": round(float(uncertainty), 6),
        "novelty_score": round(float(novelty), 4),
        "n_neighbors": len(neighbors),
        "nearest_distance": round(float(min_dist), 4),
    }


def _config_distance(
    config_a: dict[str, float],
    config_b: dict[str, float],
    feature_names: list[str],
) -> float:
    """Compute normalized distance between two configs."""
    total = 0.0
    n = 0
    for feat in feature_names:
        a = config_a.get(feat)
        b = config_b.get(feat)
        if a is not None and b is not None:
            # Normalize by max(|a|, |b|, 1) to handle different scales
            scale = max(abs(a), abs(b), 1.0)
            total += ((a - b) / scale) ** 2
            n += 1

    if n == 0:
        return float("inf")
    return float(np.sqrt(total / n))


# --- Simulation Pipeline ---


def simulate_experiments(
    proposed_configs: list[dict],
    experiments: list[dict],
    primary_metric: str,
    top_k: int = DEFAULT_TOP_K,
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
    lower_is_better: bool = False,
) -> dict:
    """Simulate proposed experiments and rank by predicted outcome.

    Args:
        proposed_configs: List of experiment configs to simulate.
        experiments: Historical experiment data.
        primary_metric: Metric to predict.
        top_k: Number of top configs to recommend running.
        improvement_threshold: Minimum predicted improvement over current best.
        lower_is_better: Whether lower metric is better.

    Returns:
        Simulation report with ranked configs and budget savings.
    """
    if not proposed_configs:
        return {"error": "No proposed configs to simulate"}

    surrogate = build_surrogate(experiments, primary_metric)
    if surrogate.get("status") != "ready":
        return {
            "error": f"Insufficient experiment history ({surrogate.get('n_points', 0)} experiments, "
                     f"need {MIN_HISTORY_FOR_SURROGATE})",
            "suggestion": "Run more experiments first to build a reliable surrogate model.",
        }

    # Get current best
    best_metrics = [
        exp.get("metrics", {}).get(primary_metric)
        for exp in experiments
        if exp.get("metrics", {}).get(primary_metric) is not None
    ]
    if lower_is_better:
        current_best = min(best_metrics) if best_metrics else float("inf")
    else:
        current_best = max(best_metrics) if best_metrics else 0

    # Predict each config
    predictions = []
    for i, config in enumerate(proposed_configs):
        features = extract_config_features(config)
        pred = predict_with_surrogate(surrogate, features)
        predicted = pred.get("predicted")
        uncertainty = pred.get("uncertainty", 0)

        if predicted is not None:
            if lower_is_better:
                improvement = current_best - predicted
            else:
                improvement = predicted - current_best

            # Classify uncertainty
            if uncertainty < 0.005:
                unc_level = "LOW"
            elif uncertainty < 0.015:
                unc_level = "MED"
            else:
                unc_level = "HIGH"

            verdict = "RUN" if improvement > improvement_threshold else "SKIP"

            predictions.append({
                "rank": 0,  # filled later
                "config_index": i,
                "config_summary": _summarize_config(config),
                "predicted_metric": predicted,
                "uncertainty": uncertainty,
                "uncertainty_level": unc_level,
                "improvement": round(improvement, 6),
                "verdict": verdict,
                "novelty_score": pred.get("novelty_score", 0),
            })

    # Sort by predicted metric
    predictions.sort(
        key=lambda p: p["predicted_metric"],
        reverse=not lower_is_better,
    )

    # Assign ranks
    for i, p in enumerate(predictions):
        p["rank"] = i + 1

    # Apply top-k
    run_configs = [p for p in predictions if p["verdict"] == "RUN"][:top_k]
    skip_configs = [p for p in predictions if p not in run_configs]

    # Mark skipped
    for p in skip_configs:
        p["verdict"] = "SKIP"

    total = len(predictions)
    n_run = len(run_configs)
    n_skip = total - n_run
    savings = round(n_skip / total * 100, 1) if total > 0 else 0

    return {
        "current_best": current_best,
        "primary_metric": primary_metric,
        "total_proposed": total,
        "run_count": n_run,
        "skip_count": n_skip,
        "budget_savings_pct": savings,
        "predictions": predictions,
        "surrogate_info": {
            "n_training_points": surrogate["n_points"],
            "n_features": len(surrogate["feature_names"]),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _summarize_config(config: dict, max_items: int = 4) -> str:
    """Create a short summary of a config dict."""
    features = extract_config_features(config)
    items = list(features.items())[:max_items]
    parts = [f"{k}={v}" for k, v in items]
    if len(features) > max_items:
        parts.append("...")
    return ", ".join(parts) if parts else "(empty config)"


# --- Report Formatting ---


def save_simulation_report(report: dict, output_dir: str = "experiments/simulations") -> Path:
    """Save simulation report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filepath = out_path / f"simulation-{ts}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_simulation_report(report: dict) -> str:
    """Format simulation report as readable markdown."""
    if "error" in report:
        lines = [f"ERROR: {report['error']}"]
        if "suggestion" in report:
            lines.append(f"\n{report['suggestion']}")
        return "\n".join(lines)

    lines = ["# Experiment Simulation", ""]
    lines.append(f"**Current best:** {report.get('current_best', 'N/A')}")
    lines.append(f"**Proposed configs:** {report.get('total_proposed', 0)}")
    lines.append(f"**Recommended to run:** {report.get('run_count', 0)}")
    lines.append(f"**Budget savings:** {report.get('budget_savings_pct', 0)}%")
    lines.append("")

    predictions = report.get("predictions", [])
    if predictions:
        lines.append("| Rank | Config Summary | Predicted | Uncertainty | Verdict |")
        lines.append("|------|---------------|-----------|-------------|---------|")
        for p in predictions:
            unc = f"{p['predicted_metric']:.4f} \u00b1 {p['uncertainty']:.4f}"
            lines.append(
                f"| {p['rank']} | {p['config_summary'][:40]} | {p['predicted_metric']:.4f} "
                f"| {p['uncertainty_level']} | {p['verdict']} {'✓' if p['verdict'] == 'RUN' else '✗'} |"
            )

    lines.append("")
    rec_run = report.get("run_count", 0)
    rec_skip = report.get("skip_count", 0)
    lines.append(
        f"**Recommendation:** Run top {rec_run}, skip {rec_skip}. "
        f"Estimated budget savings: {report.get('budget_savings_pct', 0)}%."
    )

    lines.append("")
    lines.append(f"*Generated: {report.get('generated_at', 'N/A')}*")
    return "\n".join(lines)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Experiment outcome simulator — predict results before running"
    )
    parser.add_argument("--configs", help="YAML file with proposed experiment configs")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Number of top configs to recommend")
    parser.add_argument("--threshold", type=float, default=DEFAULT_IMPROVEMENT_THRESHOLD,
                        help="Minimum predicted improvement to recommend running")
    parser.add_argument("--config", default="config.yaml", help="Path to project config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(args.log)

    # Load proposed configs
    proposed = []
    if args.configs:
        with open(args.configs) as f:
            data = yaml.safe_load(f)
        if isinstance(data, list):
            proposed = data
        elif isinstance(data, dict) and "configs" in data:
            proposed = data["configs"]
        else:
            proposed = [data]

    if not proposed:
        print("No proposed configs provided. Use --configs <file.yaml>")
        sys.exit(1)

    report = simulate_experiments(
        proposed_configs=proposed,
        experiments=experiments,
        primary_metric=primary_metric,
        top_k=args.top_k,
        improvement_threshold=args.threshold,
        lower_is_better=lower_is_better,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_simulation_report(report))

    if "error" not in report:
        saved = save_simulation_report(report)
        if not args.json:
            print(f"\nSaved: {saved}")


if __name__ == "__main__":
    main()
