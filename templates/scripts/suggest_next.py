#!/usr/bin/env python3
"""Bayesian-guided hypothesis suggestion for the autoresearch pipeline.

Reads experiment history from log.jsonl, builds a surrogate model
(Random Forest) over the hyperparameter space, and suggests the
configurations most likely to improve the primary metric.

This is the data-driven complement to human taste: the human selects
which room to search, this script suggests which coins in that room
are most likely to be biased toward heads.

Usage:
    python scripts/suggest_next.py [--log experiments/log.jsonl] [--config config.yaml] [--top 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_experiments


def extract_features_and_targets(
    experiments: list[dict],
    metric_name: str,
) -> tuple[list[dict], list[float], list[str]]:
    """Extract hyperparameter features and metric targets from experiments.

    Returns (feature_dicts, metric_values, experiment_ids).
    Only includes experiments with valid metric values.
    """
    features = []
    targets = []
    ids = []

    for exp in experiments:
        metric_val = exp.get("metrics", {}).get(metric_name)
        if metric_val is None or not isinstance(metric_val, (int, float)):
            continue

        # Extract hyperparameters as features
        config = exp.get("config", {})
        hyperparams = config.get("hyperparams", {})
        if not hyperparams:
            # Try to find hyperparams at top level of config
            hyperparams = {k: v for k, v in config.items()
                         if k not in ("model_type",) and isinstance(v, (int, float))}

        if hyperparams:
            features.append(hyperparams)
            targets.append(float(metric_val))
            ids.append(exp.get("experiment_id", "?"))

    return features, targets, ids


def features_to_matrix(feature_dicts: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Convert list of feature dicts to a numpy matrix.

    Handles missing keys by filling with NaN.
    Returns (matrix, column_names).
    """
    if not feature_dicts:
        return np.array([]).reshape(0, 0), []

    # Collect all keys
    all_keys = sorted(set(k for d in feature_dicts for k in d if isinstance(d[k], (int, float))))

    if not all_keys:
        return np.array([]).reshape(0, 0), []

    matrix = np.full((len(feature_dicts), len(all_keys)), np.nan)
    for i, d in enumerate(feature_dicts):
        for j, key in enumerate(all_keys):
            val = d.get(key)
            if isinstance(val, (int, float)):
                matrix[i, j] = val

    return matrix, all_keys


def suggest_configurations(
    experiments: list[dict],
    metric_name: str,
    lower_is_better: bool = False,
    n_suggestions: int = 3,
    sweep_config_path: str | None = None,
) -> list[dict]:
    """Suggest promising configurations using a Random Forest surrogate.

    If a sweep config exists, generates candidates from the untried region.
    Otherwise, generates perturbations of the best-known configuration.

    Returns list of suggestion dicts with predicted_metric and config.
    """
    features, targets, ids = extract_features_and_targets(experiments, metric_name)

    if len(features) < 3:
        return [{
            "reason": "insufficient_data",
            "detail": f"Need at least 3 experiments with hyperparameters, have {len(features)}",
            "suggestion": "Run more experiments before requesting data-driven suggestions",
        }]

    X, col_names = features_to_matrix(features)
    y = np.array(targets)

    if X.shape[1] == 0:
        return [{
            "reason": "no_numeric_hyperparameters",
            "detail": "No numeric hyperparameters found in experiment configs",
            "suggestion": "Ensure config includes numeric hyperparams like n_estimators, max_depth, learning_rate",
        }]

    # Handle NaN in features
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_clean = imputer.fit_transform(X)

    # Fit surrogate model
    from sklearn.ensemble import RandomForestRegressor
    surrogate = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    surrogate.fit(X_clean, y)

    # Generate candidates
    candidates = _generate_candidates(X_clean, col_names, sweep_config_path, n_candidates=200)

    if len(candidates) == 0:
        return [{
            "reason": "no_candidates",
            "detail": "Could not generate candidate configurations",
            "suggestion": "Check sweep_config.yaml or experiment hyperparameter ranges",
        }]

    # Predict with surrogate
    preds = surrogate.predict(candidates)

    # Also get uncertainty (std across trees)
    tree_preds = np.array([tree.predict(candidates) for tree in surrogate.estimators_])
    uncertainties = np.std(tree_preds, axis=0)

    # Acquisition function: UCB (Upper Confidence Bound)
    # For higher-is-better: score = predicted + kappa * uncertainty
    # For lower-is-better: score = -predicted + kappa * uncertainty
    kappa = 1.5  # exploration-exploitation tradeoff
    if lower_is_better:
        scores = -preds + kappa * uncertainties
    else:
        scores = preds + kappa * uncertainties

    # Select top-N by acquisition score
    top_indices = np.argsort(scores)[-n_suggestions:][::-1]

    suggestions = []
    for idx in top_indices:
        config = {col_names[j]: round(float(candidates[idx, j]), 6) for j in range(len(col_names))}
        suggestions.append({
            "config": config,
            "predicted_metric": round(float(preds[idx]), 6),
            "uncertainty": round(float(uncertainties[idx]), 6),
            "acquisition_score": round(float(scores[idx]), 6),
        })

    return suggestions


def _generate_candidates(
    X: np.ndarray,
    col_names: list[str],
    sweep_config_path: str | None,
    n_candidates: int = 200,
) -> np.ndarray:
    """Generate candidate configurations for the surrogate to evaluate.

    Uses sweep config ranges if available, otherwise perturbs existing data.
    """
    if sweep_config_path and Path(sweep_config_path).exists():
        return _candidates_from_sweep(sweep_config_path, col_names, n_candidates)

    return _candidates_from_perturbation(X, n_candidates)


def _candidates_from_sweep(
    sweep_config_path: str,
    col_names: list[str],
    n_candidates: int,
) -> np.ndarray:
    """Generate random candidates from sweep parameter ranges."""
    with open(sweep_config_path) as f:
        sweep_config = yaml.safe_load(f)

    sweep_params = sweep_config.get("sweep", {})
    if not sweep_params:
        return np.array([]).reshape(0, len(col_names))

    # Map sweep param names to column indices
    candidates = np.random.RandomState(42).uniform(size=(n_candidates, len(col_names)))

    for j, col in enumerate(col_names):
        # Try to find matching sweep param
        matching_key = None
        for key in sweep_params:
            if key.endswith(col):
                matching_key = key
                break

        if matching_key and isinstance(sweep_params[matching_key], list):
            values = [v for v in sweep_params[matching_key] if isinstance(v, (int, float))]
            if values:
                lo, hi = min(values), max(values)
                # Expand range slightly for exploration
                margin = (hi - lo) * 0.2 if hi != lo else abs(lo) * 0.5
                candidates[:, j] = np.random.RandomState(42 + j).uniform(
                    lo - margin, hi + margin, size=n_candidates,
                )

    return candidates


def _candidates_from_perturbation(
    X: np.ndarray,
    n_candidates: int,
) -> np.ndarray:
    """Generate candidates by perturbing existing observations."""
    rng = np.random.RandomState(42)

    # Compute column ranges
    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = np.abs(col_min[col_range == 0]) * 0.5 + 1e-6

    candidates = np.zeros((n_candidates, X.shape[1]))
    for i in range(n_candidates):
        # Pick a random existing point and perturb it
        base_idx = rng.randint(0, X.shape[0])
        perturbation = rng.normal(0, 0.3, size=X.shape[1]) * col_range
        candidates[i] = X[base_idx] + perturbation

    return candidates


def format_suggestions(suggestions: list[dict], metric_name: str) -> str:
    """Format suggestions for display."""
    if not suggestions:
        return "No suggestions available."

    if "reason" in suggestions[0]:
        return f"Cannot suggest: {suggestions[0]['detail']}\n{suggestions[0].get('suggestion', '')}"

    lines = [f"Top {len(suggestions)} suggested configurations (by expected {metric_name}):", ""]

    for i, s in enumerate(suggestions, 1):
        config_str = ", ".join(f"{k}={v}" for k, v in s["config"].items())
        lines.append(f"  {i}. {config_str}")
        lines.append(f"     Predicted {metric_name}: {s['predicted_metric']:.4f} (uncertainty: {s['uncertainty']:.4f})")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Suggest next experiment configuration")
    parser.add_argument("--log", default="experiments/log.jsonl")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sweep", default="sweep_config.yaml", help="Sweep config for candidate ranges")
    parser.add_argument("--top", type=int, default=3, help="Number of suggestions")
    args = parser.parse_args()

    # Load config
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    eval_cfg = config.get("evaluation", {})
    metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(args.log)
    suggestions = suggest_configurations(
        experiments, metric, lower_is_better, args.top,
        sweep_config_path=args.sweep if Path(args.sweep).exists() else None,
    )

    print(format_suggestions(suggestions, metric))


if __name__ == "__main__":
    main()
