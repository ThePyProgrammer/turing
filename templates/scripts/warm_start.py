#!/usr/bin/env python3
"""Warm-start from prior model for the autoresearch pipeline.

Takes a trained checkpoint and uses it as initialization for a different
configuration. Automates the "start from here but change X" pattern for
tree models (continue boosting), neural networks (load weights, freeze
layers), and scikit-learn (warm_start=True).

Usage:
    python scripts/warm_start.py exp-042
    python scripts/warm_start.py exp-042 --freeze-layers encoder
    python scripts/warm_start.py exp-042 --unfreeze-after 5
    python scripts/warm_start.py exp-042 --lr-factor 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_CHECKPOINT_DIR = "experiments/checkpoints"
DEFAULT_LR_FACTOR = 0.1  # Reduce LR by 10x for fine-tuning


# --- Model Type Detection ---


TREE_MODELS = {"xgboost", "lightgbm", "catboost", "gradient_boosting", "gbm"}
NEURAL_MODELS = {"mlp", "neural_network", "nn", "pytorch", "tensorflow", "keras", "transformer"}
SKLEARN_WARM_STARTABLE = {
    "random_forest", "gradient_boosting", "mlp",
    "sgd", "passive_aggressive", "perceptron",
    "bagging", "adaboost",
}


def detect_model_type(experiment: dict) -> str:
    """Detect the model type category from an experiment.

    Returns one of: 'tree', 'neural', 'sklearn', 'unknown'.
    """
    config = experiment.get("config", {})
    model_type = config.get("model_type", "").lower()

    if any(t in model_type for t in TREE_MODELS):
        return "tree"
    if any(t in model_type for t in NEURAL_MODELS):
        return "neural"
    if any(t in model_type for t in SKLEARN_WARM_STARTABLE):
        return "sklearn"

    # Check hyperparams for hints
    hyperparams = config.get("hyperparams", {})
    if "n_estimators" in hyperparams and ("max_depth" in hyperparams or "num_leaves" in hyperparams):
        return "tree"
    if "hidden_size" in hyperparams or "layers" in hyperparams:
        return "neural"

    return "unknown"


# --- Warm-Start Strategy ---


def plan_warm_start(
    experiment: dict,
    freeze_layers: list[str] | None = None,
    unfreeze_after: int | None = None,
    lr_factor: float = DEFAULT_LR_FACTOR,
) -> dict:
    """Plan the warm-start strategy for an experiment.

    Args:
        experiment: Source experiment to warm-start from.
        freeze_layers: Layer names to freeze (neural only).
        unfreeze_after: Unfreeze all layers after N epochs (neural only).
        lr_factor: Learning rate reduction factor for fine-tuning.

    Returns:
        Warm-start plan dict with strategy, config changes, and instructions.
    """
    model_category = detect_model_type(experiment)
    exp_id = experiment.get("experiment_id", "unknown")
    config = experiment.get("config", {})
    hyperparams = config.get("hyperparams", {})

    plan = {
        "source_experiment": exp_id,
        "model_category": model_category,
        "model_type": config.get("model_type", "unknown"),
        "strategy": None,
        "config_changes": {},
        "instructions": [],
        "warnings": [],
    }

    if model_category == "tree":
        plan.update(_plan_tree_warm_start(config, hyperparams))
    elif model_category == "neural":
        plan.update(_plan_neural_warm_start(
            config, hyperparams, freeze_layers, unfreeze_after, lr_factor,
        ))
    elif model_category == "sklearn":
        plan.update(_plan_sklearn_warm_start(config, hyperparams))
    else:
        plan["strategy"] = "unsupported"
        plan["warnings"].append(
            f"Model type '{config.get('model_type', '?')}' does not support warm-starting. "
            "Consider manually loading the checkpoint."
        )

    return plan


def _plan_tree_warm_start(config: dict, hyperparams: dict) -> dict:
    """Plan warm-start for tree-based models."""
    model_type = config.get("model_type", "").lower()
    current_estimators = hyperparams.get("n_estimators", 100)

    changes = {}
    instructions = []

    if "xgboost" in model_type:
        changes["xgb_model"] = "checkpoint_path"  # Load existing model
        changes["n_estimators"] = current_estimators + 100  # Continue boosting
        instructions.append(f"Load XGBoost model from checkpoint")
        instructions.append(f"Continue boosting: {current_estimators} → {current_estimators + 100} estimators")
        strategy = "continue_boosting"

    elif "lightgbm" in model_type:
        changes["init_model"] = "checkpoint_path"
        changes["n_estimators"] = current_estimators + 100
        instructions.append(f"Load LightGBM model as init_model")
        instructions.append(f"Continue training with additional estimators")
        strategy = "continue_boosting"

    else:
        changes["warm_start"] = True
        changes["n_estimators"] = current_estimators + 100
        instructions.append(f"Set warm_start=True for incremental learning")
        strategy = "warm_start_param"

    return {
        "strategy": strategy,
        "config_changes": changes,
        "instructions": instructions,
    }


def _plan_neural_warm_start(
    config: dict,
    hyperparams: dict,
    freeze_layers: list[str] | None,
    unfreeze_after: int | None,
    lr_factor: float,
) -> dict:
    """Plan warm-start for neural network models."""
    changes = {}
    instructions = []
    warnings = []

    # Load weights
    changes["load_checkpoint"] = True
    changes["checkpoint_source"] = config.get("model_type", "?")
    instructions.append("Load weights from source experiment checkpoint")

    # Layer freezing
    if freeze_layers:
        changes["freeze_layers"] = freeze_layers
        instructions.append(f"Freeze layers: {', '.join(freeze_layers)}")

        if unfreeze_after:
            changes["unfreeze_after_epochs"] = unfreeze_after
            instructions.append(f"Gradual unfreezing: unfreeze all after epoch {unfreeze_after}")

    # Learning rate adjustment
    current_lr = hyperparams.get("learning_rate", hyperparams.get("lr", 0.001))
    new_lr = current_lr * lr_factor
    changes["learning_rate"] = new_lr
    instructions.append(f"Reduce learning rate: {current_lr} → {new_lr} ({lr_factor}x)")

    # Reset optimizer
    changes["reset_optimizer"] = True
    instructions.append("Reset optimizer state (fresh momentum/adaptive learning rates)")

    if not freeze_layers:
        warnings.append(
            "No layers frozen — all weights will be updated. "
            "Consider freezing early layers for more stable fine-tuning."
        )

    return {
        "strategy": "load_weights",
        "config_changes": changes,
        "instructions": instructions,
        "warnings": warnings,
    }


def _plan_sklearn_warm_start(config: dict, hyperparams: dict) -> dict:
    """Plan warm-start for scikit-learn models."""
    return {
        "strategy": "warm_start_param",
        "config_changes": {
            "warm_start": True,
            "n_estimators": hyperparams.get("n_estimators", 100) + 50,
        },
        "instructions": [
            "Set warm_start=True on the estimator",
            "Increase n_estimators for additional rounds",
            "Call fit() with the original training data — model continues from prior state",
        ],
    }


# --- Checkpoint Discovery ---


def find_checkpoint(
    exp_id: str,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> dict | None:
    """Find the checkpoint for a given experiment.

    Returns dict with path, format, and size if found.
    """
    ckpt_path = Path(checkpoint_dir)

    # Check for experiment-specific checkpoint directory
    exp_dir = ckpt_path / exp_id
    if exp_dir.exists() and exp_dir.is_dir():
        files = list(exp_dir.rglob("*"))
        model_files = [f for f in files if f.is_file()]
        if model_files:
            total_size = sum(f.stat().st_size for f in model_files)
            return {
                "path": str(exp_dir),
                "format": _detect_checkpoint_format(model_files),
                "n_files": len(model_files),
                "size_bytes": total_size,
                "size_mb": round(total_size / (1024 * 1024), 2),
            }

    # Check for single file checkpoints
    for ext in (".joblib", ".pkl", ".pt", ".pth", ".h5", ".xgb", ".lgb", ".cbm"):
        candidate = ckpt_path / f"{exp_id}{ext}"
        if candidate.exists():
            return {
                "path": str(candidate),
                "format": ext.lstrip("."),
                "n_files": 1,
                "size_bytes": candidate.stat().st_size,
                "size_mb": round(candidate.stat().st_size / (1024 * 1024), 2),
            }

    return None


def _detect_checkpoint_format(files: list[Path]) -> str:
    """Detect the format of checkpoint files."""
    extensions = {f.suffix.lower() for f in files}
    if ".pt" in extensions or ".pth" in extensions:
        return "pytorch"
    if ".h5" in extensions:
        return "keras"
    if ".joblib" in extensions:
        return "joblib"
    if ".pkl" in extensions:
        return "pickle"
    if ".xgb" in extensions:
        return "xgboost"
    if ".lgb" in extensions:
        return "lightgbm"
    return "unknown"


# --- Full Warm-Start Pipeline ---


def warm_start(
    exp_id: str,
    freeze_layers: list[str] | None = None,
    unfreeze_after: int | None = None,
    lr_factor: float = DEFAULT_LR_FACTOR,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> dict:
    """Plan and prepare a warm-start from a prior experiment.

    Args:
        exp_id: Source experiment ID.
        freeze_layers: Layers to freeze (neural only).
        unfreeze_after: Unfreeze after N epochs (neural only).
        lr_factor: Learning rate reduction factor.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        checkpoint_dir: Checkpoint directory.

    Returns:
        Complete warm-start report.
    """
    experiments = load_experiments(log_path)

    source = None
    for exp in experiments:
        if exp.get("experiment_id") == exp_id:
            source = exp
            break

    if not source:
        return {"error": f"Experiment {exp_id} not found in {log_path}"}

    # Find checkpoint
    checkpoint = find_checkpoint(exp_id, checkpoint_dir)

    # Plan warm-start
    plan = plan_warm_start(source, freeze_layers, unfreeze_after, lr_factor)

    # Source experiment info
    source_metrics = source.get("metrics", {})

    report = {
        "source_experiment": exp_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_metrics": source_metrics,
        "checkpoint": checkpoint,
        "plan": plan,
    }

    if not checkpoint:
        report["warning"] = (
            f"No checkpoint found for {exp_id} in {checkpoint_dir}. "
            "The warm-start plan is ready but requires a saved checkpoint to execute."
        )

    return report


# --- Report Formatting ---


def save_warm_start_report(report: dict, output_dir: str = "experiments/warm_starts") -> Path:
    """Save warm-start report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exp_id = report.get("source_experiment", "unknown")
    filepath = out_path / f"warm-{exp_id}.yaml"

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_warm_start_report(report: dict) -> str:
    """Format warm-start report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    plan = report.get("plan", {})
    exp_id = report.get("source_experiment", "?")

    lines = [
        f"# Warm-Start Plan: {exp_id}",
        "",
        f"*Generated {report.get('generated_at', 'N/A')[:19]}*",
        "",
        f"**Model:** {plan.get('model_type', '?')} ({plan.get('model_category', '?')})",
        f"**Strategy:** {plan.get('strategy', '?')}",
        "",
    ]

    # Source metrics
    metrics = report.get("source_metrics", {})
    if metrics:
        lines.extend(["## Source Experiment Metrics", ""])
        for k, v in metrics.items():
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"- **{k}:** {v_str}")
        lines.append("")

    # Checkpoint info
    checkpoint = report.get("checkpoint")
    if checkpoint:
        lines.extend([
            "## Checkpoint",
            "",
            f"- **Path:** {checkpoint['path']}",
            f"- **Format:** {checkpoint['format']}",
            f"- **Size:** {checkpoint.get('size_mb', 0):.1f} MB",
            "",
        ])
    elif report.get("warning"):
        lines.extend(["## Checkpoint", "", f"WARNING: {report['warning']}", ""])

    # Instructions
    instructions = plan.get("instructions", [])
    if instructions:
        lines.extend(["## Steps", ""])
        for i, inst in enumerate(instructions, 1):
            lines.append(f"{i}. {inst}")
        lines.append("")

    # Config changes
    changes = plan.get("config_changes", {})
    if changes:
        lines.extend(["## Config Changes", ""])
        for k, v in changes.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    # Warnings
    warnings = plan.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", ""])
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Warm-start from prior model checkpoint",
    )
    parser.add_argument(
        "exp_id",
        help="Source experiment ID (e.g., exp-042)",
    )
    parser.add_argument(
        "--freeze-layers", nargs="+",
        help="Layer names to freeze (neural networks only)",
    )
    parser.add_argument(
        "--unfreeze-after", type=int,
        help="Unfreeze all layers after N epochs (gradual unfreezing)",
    )
    parser.add_argument(
        "--lr-factor", type=float, default=DEFAULT_LR_FACTOR,
        help=f"Learning rate reduction factor (default: {DEFAULT_LR_FACTOR})",
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
        "--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    report = warm_start(
        exp_id=args.exp_id,
        freeze_layers=args.freeze_layers,
        unfreeze_after=args.unfreeze_after,
        lr_factor=args.lr_factor,
        config_path=args.config,
        log_path=args.log,
        checkpoint_dir=args.checkpoint_dir,
    )

    if "error" not in report:
        filepath = save_warm_start_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_warm_start_report(report))


if __name__ == "__main__":
    main()
