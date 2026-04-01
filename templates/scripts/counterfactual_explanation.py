#!/usr/bin/env python3
"""Input-level counterfactual explanations for the autoresearch pipeline.

For a given prediction, finds the smallest input change that would flip
the outcome. "This sample was classified as X — what's the minimum change
to make it Y?" Useful for debugging predictions and regulatory explanations.

Usage:
    python scripts/counterfactual_explanation.py exp-042 --sample 1247
    python scripts/counterfactual_explanation.py exp-042 --sample 1247 --target 0
    python scripts/counterfactual_explanation.py exp-042 --batch-misclassified
    python scripts/counterfactual_explanation.py --json
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
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_DISTANCE_METRIC = "normalized_l2"


# --- Feature Perturbation ---


def greedy_perturbation(
    sample: dict[str, float],
    predict_fn,
    target_class: int | str,
    feature_names: list[str],
    feature_ranges: dict[str, tuple[float, float]],
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    categorical_features: list[str] | None = None,
) -> dict:
    """Find counterfactual by greedily changing one feature at a time.

    Args:
        sample: Original sample as {feature_name: value}.
        predict_fn: Function(sample_dict) -> (predicted_class, confidence).
        target_class: Desired target class.
        feature_names: Ordered list of feature names.
        feature_ranges: {feature: (min, max)} from training data.
        max_iterations: Maximum perturbation attempts.
        categorical_features: Features that are categorical (discrete changes).

    Returns:
        Counterfactual result dict.
    """
    if categorical_features is None:
        categorical_features = []

    current = dict(sample)
    original_pred, original_conf = predict_fn(sample)

    if str(original_pred) == str(target_class):
        return {
            "status": "already_target",
            "message": f"Sample is already predicted as {target_class}",
        }

    best_cf = None
    best_distance = float("inf")
    changes = []

    for iteration in range(max_iterations):
        improved = False

        for feat in feature_names:
            if feat in categorical_features:
                candidates = _categorical_candidates(feat, current[feat], feature_ranges.get(feat))
            else:
                candidates = _numeric_candidates(
                    current[feat],
                    feature_ranges.get(feat, (0, 1)),
                    n_steps=5,
                )

            for candidate_val in candidates:
                trial = dict(current)
                trial[feat] = candidate_val
                pred, conf = predict_fn(trial)

                if str(pred) == str(target_class):
                    dist = _compute_distance(sample, trial, feature_ranges)
                    if dist < best_distance:
                        best_distance = dist
                        best_cf = dict(trial)
                        changes = _compute_changes(sample, trial, feature_names)
                        improved = True

        if best_cf is not None and not improved:
            break

    if best_cf is None:
        return {
            "status": "not_found",
            "message": f"Could not find counterfactual within {max_iterations} iterations",
            "original_prediction": original_pred,
            "original_confidence": float(original_conf),
        }

    cf_pred, cf_conf = predict_fn(best_cf)

    return {
        "status": "found",
        "original_prediction": original_pred,
        "original_confidence": float(original_conf),
        "counterfactual_prediction": cf_pred,
        "counterfactual_confidence": float(cf_conf),
        "distance": round(float(best_distance), 4),
        "n_changes": len(changes),
        "changes": changes,
        "counterfactual_sample": best_cf,
    }


def _numeric_candidates(current: float, value_range: tuple[float, float], n_steps: int = 5) -> list[float]:
    """Generate candidate values for a numeric feature."""
    low, high = value_range
    step = (high - low) / max(n_steps, 1)
    candidates = []
    for i in range(n_steps + 1):
        val = low + i * step
        if val != current:
            candidates.append(val)
    return candidates


def _categorical_candidates(
    feature: str,
    current_value,
    value_range: tuple | list | None,
) -> list:
    """Generate candidate values for a categorical feature."""
    if value_range is None:
        return []
    if isinstance(value_range, (tuple, list)):
        return [v for v in value_range if v != current_value]
    return []


def _compute_distance(
    original: dict[str, float],
    counterfactual: dict[str, float],
    feature_ranges: dict[str, tuple[float, float]],
) -> float:
    """Compute normalized L2 distance between original and counterfactual."""
    total = 0.0
    for feat in original:
        orig_val = original[feat]
        cf_val = counterfactual.get(feat, orig_val)
        low, high = feature_ranges.get(feat, (0, 1))
        span = high - low if high != low else 1
        normalized_diff = (cf_val - orig_val) / span
        total += normalized_diff ** 2
    return float(np.sqrt(total))


def _compute_changes(
    original: dict[str, float],
    counterfactual: dict[str, float],
    feature_names: list[str],
) -> list[dict]:
    """Compute the list of changed features."""
    changes = []
    for feat in feature_names:
        orig = original.get(feat)
        cf = counterfactual.get(feat)
        if orig != cf:
            change = {
                "feature": feat,
                "original": orig,
                "counterfactual": cf,
            }
            if isinstance(orig, (int, float)) and isinstance(cf, (int, float)):
                change["delta"] = round(cf - orig, 6)
            else:
                change["delta"] = "category_change"
            changes.append(change)
    return changes


# --- Prototype-Based Search ---


def prototype_counterfactual(
    sample: dict[str, float],
    training_data: list[dict[str, float]],
    training_labels: list,
    target_class: int | str,
    feature_names: list[str],
    feature_ranges: dict[str, tuple[float, float]],
) -> dict:
    """Find the nearest training sample from the target class.

    Args:
        sample: Original sample.
        training_data: List of training samples as dicts.
        training_labels: Corresponding labels.
        target_class: Desired target class.
        feature_names: Feature names.
        feature_ranges: {feature: (min, max)}.

    Returns:
        Nearest prototype counterfactual result.
    """
    target_indices = [i for i, label in enumerate(training_labels) if str(label) == str(target_class)]

    if not target_indices:
        return {
            "status": "not_found",
            "message": f"No training samples found for class {target_class}",
        }

    best_dist = float("inf")
    best_idx = -1

    for idx in target_indices:
        dist = _compute_distance(sample, training_data[idx], feature_ranges)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    if best_idx < 0:
        return {"status": "not_found", "message": "No valid prototype found"}

    prototype = training_data[best_idx]
    changes = _compute_changes(sample, prototype, feature_names)

    return {
        "status": "found",
        "method": "prototype",
        "prototype_index": best_idx,
        "distance": round(float(best_dist), 4),
        "n_changes": len(changes),
        "changes": changes,
        "counterfactual_sample": prototype,
    }


# --- Full Pipeline ---


def counterfactual_analysis(
    exp_id: str,
    sample_index: int | None = None,
    sample_data: dict[str, float] | None = None,
    target_class: int | str | None = None,
    predict_fn=None,
    training_data: list[dict] | None = None,
    training_labels: list | None = None,
    feature_names: list[str] | None = None,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
    categorical_features: list[str] | None = None,
    batch_misclassified: bool = False,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Run counterfactual analysis.

    Args:
        exp_id: Experiment ID to analyze.
        sample_index: Index of the sample to explain.
        sample_data: Direct sample data (alternative to index).
        target_class: Desired counterfactual class.
        predict_fn: Prediction function (sample_dict) -> (class, confidence).
        training_data: Training data for prototype search.
        training_labels: Training labels for prototype search.
        feature_names: Feature names.
        feature_ranges: Feature value ranges.
        categorical_features: Categorical feature names.
        batch_misclassified: If True, generate for all misclassified samples.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.

    Returns:
        Counterfactual analysis report.
    """
    config = load_config(config_path)

    if sample_data is None and sample_index is None and not batch_misclassified:
        return {"error": "Provide --sample <index> or --batch-misclassified"}

    if predict_fn is None:
        return {
            "error": "No prediction function available. "
                     "Load the model from the experiment first.",
            "suggestion": f"Run `/turing:counterfactual {exp_id} --sample <index>` "
                          "from the experiment directory with train.py available.",
        }

    if feature_names is None:
        return {"error": "Feature names not available. Provide feature_names."}

    if feature_ranges is None:
        feature_ranges = {}

    results = []

    if batch_misclassified and training_data and training_labels:
        for i, (data, label) in enumerate(zip(training_data, training_labels)):
            pred, conf = predict_fn(data)
            if str(pred) != str(label):
                cf = greedy_perturbation(
                    data, predict_fn, label, feature_names,
                    feature_ranges, categorical_features=categorical_features,
                )
                cf["sample_index"] = i
                cf["true_label"] = label
                results.append(cf)
    elif sample_data is not None:
        if target_class is None:
            pred, _ = predict_fn(sample_data)
            # Flip to opposite for binary
            target_class = 0 if pred == 1 else 1

        # Try greedy perturbation
        cf_greedy = greedy_perturbation(
            sample_data, predict_fn, target_class, feature_names,
            feature_ranges, categorical_features=categorical_features,
        )

        # Try prototype-based if training data available
        cf_proto = None
        if training_data and training_labels:
            cf_proto = prototype_counterfactual(
                sample_data, training_data, training_labels,
                target_class, feature_names, feature_ranges,
            )

        results = {
            "greedy": cf_greedy,
            "prototype": cf_proto,
            "best": _select_best([cf_greedy, cf_proto]),
        }

    return {
        "experiment_id": exp_id,
        "sample_index": sample_index,
        "target_class": target_class,
        "results": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _select_best(candidates: list[dict | None]) -> dict | None:
    """Select the counterfactual with smallest distance."""
    valid = [c for c in candidates if c and c.get("status") == "found"]
    if not valid:
        return None
    return min(valid, key=lambda c: c.get("distance", float("inf")))


# --- Report Formatting ---


def save_counterfactual_report(report: dict, output_dir: str = "experiments/counterfactuals") -> Path:
    """Save counterfactual report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("experiment_id", "unknown")
    sample = report.get("sample_index", "batch")
    filepath = out_path / f"{exp_id}-cf-{sample}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_counterfactual_report(report: dict) -> str:
    """Format counterfactual report as readable markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    lines = ["# Counterfactual Explanation", ""]
    lines.append(f"**Experiment:** {report.get('experiment_id', 'N/A')}")
    lines.append(f"**Sample:** {report.get('sample_index', 'N/A')}")
    lines.append(f"**Target class:** {report.get('target_class', 'N/A')}")
    lines.append("")

    results = report.get("results", {})

    if isinstance(results, dict):
        best = results.get("best")
        if best and best.get("status") == "found":
            lines.append(f"**Method:** {best.get('method', 'greedy')}")
            lines.append(f"**Distance:** {best.get('distance', 'N/A')}")
            lines.append(f"**Changes needed:** {best.get('n_changes', 0)}")
            lines.append("")

            changes = best.get("changes", [])
            if changes:
                lines.append("| Feature | Original | Counterfactual | Change |")
                lines.append("|---------|----------|----------------|--------|")
                for c in changes:
                    delta = c.get("delta", "")
                    if isinstance(delta, (int, float)):
                        delta_str = f"{delta:+.4f}" if isinstance(delta, float) else f"{delta:+d}"
                    else:
                        delta_str = str(delta)
                    lines.append(
                        f"| {c['feature']} | {c['original']} | {c['counterfactual']} | {delta_str} |"
                    )
        else:
            lines.append("No counterfactual found within search budget.")

        # Show method comparison
        greedy = results.get("greedy", {})
        proto = results.get("prototype", {})
        if greedy.get("status") == "found" or (proto and proto.get("status") == "found"):
            lines.append("")
            lines.append("**Method comparison:**")
            if greedy.get("status") == "found":
                lines.append(f"- Greedy: distance={greedy.get('distance')}, changes={greedy.get('n_changes')}")
            if proto and proto.get("status") == "found":
                lines.append(f"- Prototype: distance={proto.get('distance')}, changes={proto.get('n_changes')}")

    elif isinstance(results, list):
        lines.append(f"**Batch results:** {len(results)} misclassified samples analyzed")
        found = sum(1 for r in results if r.get("status") == "found")
        lines.append(f"**Counterfactuals found:** {found}/{len(results)}")

    lines.append("")
    lines.append(f"*Generated: {report.get('generated_at', 'N/A')}*")
    return "\n".join(lines)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual explanations — find minimum input changes to flip predictions"
    )
    parser.add_argument("exp_id", nargs="?", help="Experiment ID")
    parser.add_argument("--sample", type=int, help="Sample index to explain")
    parser.add_argument("--target", help="Target class for counterfactual")
    parser.add_argument("--batch-misclassified", action="store_true",
                        help="Generate counterfactuals for all misclassified samples")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    if not args.exp_id:
        parser.error("Please provide an experiment ID")

    report = counterfactual_analysis(
        exp_id=args.exp_id,
        sample_index=args.sample,
        target_class=args.target,
        batch_misclassified=args.batch_misclassified,
        config_path=args.config,
        log_path=args.log,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_counterfactual_report(report))

    if "error" not in report:
        saved = save_counterfactual_report(report)
        if not args.json:
            print(f"\nSaved: {saved}")


if __name__ == "__main__":
    main()
