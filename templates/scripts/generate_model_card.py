#!/usr/bin/env python3
"""Model card generator for the autoresearch pipeline.

Produces a standardized model card markdown document inspired by
Hugging Face model cards and Google's Model Cards for Model Reporting.
Consolidates information from the experiment log, model contract,
and project config into a single documentation artifact.

Usage:
    python scripts/generate_model_card.py [--config config.yaml] [--log experiments/log.jsonl] [--output MODEL_CARD.md]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments


def load_best_experiment(
    log_path: str,
    metric: str,
    lower_is_better: bool,
) -> dict | None:
    """Find the current best kept experiment.

    Scans all experiments in the log and returns the one with
    the best value for the given metric among 'kept' entries.

    Args:
        log_path: Path to the experiments JSONL file.
        metric: Primary metric name to optimize.
        lower_is_better: True for metrics like MAE/MSE, False for accuracy/F1.

    Returns:
        Best experiment dict, or None if no kept entries exist.
    """
    experiments = load_experiments(log_path)
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


def load_model_contract(contract_path: str) -> dict:
    """Read model_contract.md and extract structured fields.

    Parses the contract markdown to pull out the contract version
    and bundle format description.

    Args:
        contract_path: Path to model_contract.md.

    Returns:
        Dict with 'version', 'bundle_format', and 'raw' keys.
        Returns defaults if the file does not exist.
    """
    path = Path(contract_path)
    if not path.exists():
        return {"version": "1", "bundle_format": "joblib bundle", "raw": ""}

    text = path.read_text(encoding="utf-8")

    # Extract version
    version = "1"
    version_match = re.search(r"Version:\s*(\S+)", text)
    if version_match:
        version = version_match.group(1)

    # Extract bundle format from the heading or description
    bundle_format = "joblib bundle"
    if "joblib" in text.lower():
        bundle_format = "joblib bundle (model + featurizer + config)"
    elif "onnx" in text.lower():
        bundle_format = "ONNX"
    elif "pickle" in text.lower():
        bundle_format = "pickle"

    return {"version": version, "bundle_format": bundle_format, "raw": text}


def load_registry_status(registry_path: str = "experiments/registry.yaml") -> dict | None:
    """Load registry status for the best model."""
    path = Path(registry_path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and data.get("models"):
            return data
    except (Exception,):
        pass
    return None


def compute_fairness_metrics(
    predictions: list | None = None,
    labels: list | None = None,
    protected_attribute: list | None = None,
    group_names: list[str] | None = None,
) -> dict | None:
    """Compute demographic parity and equal opportunity metrics.

    Args:
        predictions: Model predictions.
        labels: True labels.
        protected_attribute: Group membership for each sample.
        group_names: Names of groups.

    Returns:
        Fairness metrics dict or None if insufficient data.
    """
    if predictions is None or protected_attribute is None:
        return None
    if len(predictions) != len(protected_attribute):
        return None
    if len(predictions) == 0:
        return None

    import numpy as np

    preds = np.array(predictions)
    groups = np.array(protected_attribute)
    unique_groups = sorted(set(groups))

    if group_names is None:
        group_names = [str(g) for g in unique_groups]

    # Demographic parity: P(Y_hat=1 | G=g) for each group
    group_positive_rates = {}
    for g, name in zip(unique_groups, group_names):
        mask = groups == g
        if mask.sum() == 0:
            continue
        rate = float(preds[mask].mean()) if preds[mask].size > 0 else 0
        group_positive_rates[name] = round(rate, 4)

    # Demographic parity difference
    rates = list(group_positive_rates.values())
    dp_diff = round(max(rates) - min(rates), 4) if len(rates) >= 2 else 0

    result = {
        "group_positive_rates": group_positive_rates,
        "demographic_parity_difference": dp_diff,
        "n_groups": len(unique_groups),
    }

    # Equal opportunity (if labels available): P(Y_hat=1 | Y=1, G=g)
    if labels is not None and len(labels) == len(predictions):
        labs = np.array(labels)
        group_tpr = {}
        for g, name in zip(unique_groups, group_names):
            mask = (groups == g) & (labs == 1)
            if mask.sum() == 0:
                continue
            tpr = float(preds[mask].mean()) if preds[mask].size > 0 else 0
            group_tpr[name] = round(tpr, 4)

        result["group_true_positive_rates"] = group_tpr
        tpr_vals = list(group_tpr.values())
        result["equal_opportunity_difference"] = round(max(tpr_vals) - min(tpr_vals), 4) if len(tpr_vals) >= 2 else 0

    return result


def generate_card(
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
    contract_path: str = "model_contract.md",
    output_path: str | None = None,
    include_fairness: bool = False,
    fairness_data: dict | None = None,
    registry_path: str = "experiments/registry.yaml",
) -> str:
    """Produce a model card markdown document.

    Combines information from the project config, experiment log,
    model contract, registry, and optional fairness data.

    Args:
        config_path: Path to config.yaml.
        log_path: Path to experiments/log.jsonl.
        contract_path: Path to model_contract.md.
        output_path: If given, write the card to this file.
        include_fairness: If True, add fairness section.
        fairness_data: Pre-computed fairness data {predictions, labels, protected_attribute}.
        registry_path: Path to registry YAML.

    Returns:
        The model card as a markdown string.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)
    direction = "lower" if lower_is_better else "higher"

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    experiments = load_experiments(log_path)
    best = load_best_experiment(log_path, metric, lower_is_better)
    contract = load_model_contract(contract_path)

    # Derive project name from config or directory
    project_name = config.get("project_name", Path(".").resolve().name)

    # Compute experiment stats
    total_experiments = len(experiments)
    kept_experiments = sum(1 for e in experiments if e.get("status") == "kept")

    # Determine convergence status
    convergence_cfg = config.get("convergence", {})
    patience = convergence_cfg.get("patience", 3)
    converged = "not converged"
    if kept_experiments > 0:
        # Check if last N kept experiments showed no improvement
        kept_exps = [e for e in experiments if e.get("status") == "kept"]
        if len(kept_exps) >= patience:
            recent_vals = [
                e.get("metrics", {}).get(metric)
                for e in kept_exps[-patience:]
                if e.get("metrics", {}).get(metric) is not None
            ]
            if len(recent_vals) == patience:
                if lower_is_better:
                    converged = "converged" if min(recent_vals) == recent_vals[0] else "not converged"
                else:
                    converged = "converged" if max(recent_vals) == recent_vals[0] else "not converged"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Model Card: {project_name}",
        "",
        f"Generated: {now}",
        "",
    ]

    # --- Model Details ---
    lines.extend([
        "## Model Details",
        "",
    ])
    if best:
        best_config = best.get("config", {})
        model_type = best_config.get("model_type", model_cfg.get("type", "unknown"))
        framework = _infer_framework(model_type)
        training_time = best.get("metrics", {}).get("training_time", "N/A")
        lines.extend([
            f"- **Model type:** {model_type}",
            f"- **Task:** {config.get('task_description', eval_cfg.get('primary_metric', 'N/A'))}",
            f"- **Primary metric:** {metric} ({direction} is better)",
            f"- **Framework:** {framework}",
            f"- **Training time:** {training_time}s" if training_time != "N/A" else f"- **Training time:** {training_time}",
            f"- **Artifact:** models/best/model.joblib",
        ])
    else:
        lines.extend([
            f"- **Model type:** {model_cfg.get('type', 'unknown')}",
            f"- **Task:** {config.get('task_description', 'N/A')}",
            f"- **Primary metric:** {metric} ({direction} is better)",
            "- **Framework:** N/A",
            "- **Training time:** N/A",
            "- **Artifact:** N/A (no trained model yet)",
        ])

    # --- Training Data ---
    lines.extend([
        "",
        "## Training Data",
        "",
        f"- **Source:** {data_cfg.get('source', 'N/A')}",
    ])
    split_ratios = data_cfg.get("split_ratios", {})
    if split_ratios:
        splits_str = ", ".join(f"{k}: {v}" for k, v in split_ratios.items())
        lines.append(f"- **Split ratios:** {splits_str}")
    # Samples from experiment metadata if available
    if best and best.get("metrics", {}).get("n_samples"):
        lines.append(f"- **Samples:** {best['metrics']['n_samples']}")

    # --- Performance ---
    lines.extend([
        "",
        "## Performance",
        "",
    ])
    if best:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for m, v in best.get("metrics", {}).items():
            if m in ("training_time", "n_samples"):
                continue  # Skip non-metric fields
            if isinstance(v, float):
                lines.append(f"| {m} | {v:.4f} |")
            else:
                lines.append(f"| {m} | {v} |")

        # Per-class performance
        per_class = best.get("metrics", {}).get("per_class")
        if per_class:
            lines.extend([
                "",
                "### Per-Class Performance",
                "",
            ])
            if isinstance(per_class, dict):
                headers = list(next(iter(per_class.values())).keys()) if per_class else []
                if headers:
                    lines.append("| Class | " + " | ".join(headers) + " |")
                    lines.append("|-------|" + "|".join(["-------"] * len(headers)) + "|")
                    for cls, vals in per_class.items():
                        row = " | ".join(
                            f"{vals.get(h, 'N/A'):.4f}" if isinstance(vals.get(h), float) else str(vals.get(h, "N/A"))
                            for h in headers
                        )
                        lines.append(f"| {cls} | {row} |")
    else:
        lines.append("No experiments completed yet.")

    # --- Seed Study ---
    if best:
        seed_study_path = Path("experiments/seed_studies") / f"{best.get('experiment_id', 'unknown')}-seeds.yaml"
        if seed_study_path.exists():
            with open(seed_study_path) as f:
                seed_study = yaml.safe_load(f) or {}
            if seed_study and "mean" in seed_study:
                sensitive = seed_study.get("seed_sensitive", False)
                status = "SEED-SENSITIVE" if sensitive else "STABLE"
                lines.extend([
                    "",
                    "### Seed Study",
                    "",
                    f"- **Status:** {status}",
                    f"- **{metric}:** {seed_study['mean']:.4f} +/- {seed_study.get('std', 0):.4f}",
                ])
                if "ci_95" in seed_study:
                    ci = seed_study["ci_95"]
                    lines.append(f"- **95% CI:** [{ci[0]:.4f}, {ci[1]:.4f}]")
                lines.append(f"- **CV:** {seed_study.get('cv_percent', 0):.2f}%")
                lines.append(f"- **Seeds tested:** {len(seed_study.get('seeds_run', []))}")
                if sensitive:
                    lines.append("- *Result varies significantly across seeds. Report distribution, not point estimate.*")

    # --- Training History ---
    lines.extend([
        "",
        "## Training History",
        "",
        f"- **Total experiments:** {total_experiments}",
        f"- **Experiments kept:** {kept_experiments}",
        f"- **Best experiment:** {best.get('experiment_id', 'N/A') if best else 'N/A'}",
        f"- **Convergence:** {converged}",
    ])

    # --- Limitations ---
    lines.extend([
        "",
        "## Limitations",
        "",
        f"- This model was trained on {data_cfg.get('source', 'the provided dataset')} and may not generalize to other distributions",
        "- Performance metrics are from the validation set; test set performance may differ",
    ])
    # Check for overfitting gap
    if best:
        train_metric = best.get("metrics", {}).get(f"train_{metric}")
        val_metric = best.get("metrics", {}).get(metric)
        if train_metric is not None and val_metric is not None:
            gap = abs(train_metric - val_metric)
            if gap > 0.05:
                lines.append(f"- Train/val gap of {gap:.3f} suggests possible overfitting")

    # --- Intended Use ---
    task_desc = config.get("task_description", "N/A")
    lines.extend([
        "",
        "## Intended Use",
        "",
        f"- {task_desc}",
        "- Not intended for: <placeholder for user to fill>",
    ])

    # --- Registry Status ---
    registry_data = load_registry_status(registry_path)
    if registry_data and best:
        exp_id = best.get("experiment_id", "")
        for model in registry_data.get("models", []):
            if model.get("exp_id") == exp_id:
                lines.extend([
                    "",
                    "## Registry Status",
                    "",
                    f"- **Stage:** {model.get('stage', 'unregistered')}",
                    f"- **Version:** {model.get('version', 'N/A')}",
                    f"- **Registered:** {model.get('registered_at', 'N/A')[:10]}",
                    f"- **Gates passed:** {', '.join(model.get('gates_passed', [])) or 'none'}",
                ])
                break

    # --- Fairness ---
    if include_fairness:
        lines.extend([
            "",
            "## Fairness Analysis",
            "",
        ])
        if fairness_data:
            fairness = compute_fairness_metrics(
                predictions=fairness_data.get("predictions"),
                labels=fairness_data.get("labels"),
                protected_attribute=fairness_data.get("protected_attribute"),
                group_names=fairness_data.get("group_names"),
            )
            if fairness:
                lines.append("### Demographic Parity")
                lines.append("")
                for group, rate in fairness.get("group_positive_rates", {}).items():
                    lines.append(f"- **{group}:** {rate:.4f}")
                lines.append(f"- **Parity difference:** {fairness['demographic_parity_difference']:.4f}")

                if "group_true_positive_rates" in fairness:
                    lines.append("")
                    lines.append("### Equal Opportunity")
                    lines.append("")
                    for group, tpr in fairness["group_true_positive_rates"].items():
                        lines.append(f"- **{group}:** {tpr:.4f}")
                    lines.append(f"- **Opportunity difference:** {fairness['equal_opportunity_difference']:.4f}")
            else:
                lines.append("- Fairness analysis requested but insufficient data provided")
        else:
            lines.append("- Fairness analysis requested but no protected attribute data available")
            lines.append("- Provide `--fairness-data` with predictions, labels, and protected attributes")

    # --- Ethical Considerations ---
    lines.extend([
        "",
        "## Ethical Considerations",
        "",
        "- <placeholder -- users should document bias, fairness, and impact>",
    ])

    # --- Artifact Contract ---
    lines.extend([
        "",
        "## Artifact Contract",
        "",
        f"- **Contract version:** {contract['version']}",
        f"- **Bundle format:** {contract['bundle_format']}",
        "- See `model_contract.md` for full consumer documentation",
    ])

    card = "\n".join(lines) + "\n"

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(card, encoding="utf-8")

    return card


def _infer_framework(model_type: str) -> str:
    """Infer the ML framework from the model type string."""
    model_lower = model_type.lower()
    if "xgboost" in model_lower or "xgb" in model_lower:
        return "xgboost"
    if "lightgbm" in model_lower or "lgb" in model_lower:
        return "lightgbm"
    if "catboost" in model_lower:
        return "catboost"
    if any(t in model_lower for t in ("random_forest", "randomforest", "logistic", "svm", "svc", "mlp", "knn")):
        return "sklearn"
    return model_lower


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate a standardized model card")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl", help="Path to experiment log")
    parser.add_argument("--contract", default="model_contract.md", help="Path to model contract")
    parser.add_argument("--output", default=None, help="Output path (default: print to stdout)")
    parser.add_argument("--include", default=None, help="Include extra sections (e.g., 'fairness')")
    parser.add_argument("--registry", default="experiments/registry.yaml", help="Path to model registry")
    args = parser.parse_args()

    include_fairness = args.include and "fairness" in args.include

    card = generate_card(
        args.config, args.log, args.contract, args.output,
        include_fairness=include_fairness,
        registry_path=args.registry,
    )
    if args.output:
        print(f"Model card written to {args.output}")
    else:
        print(card)


if __name__ == "__main__":
    main()
