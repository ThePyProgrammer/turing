#!/usr/bin/env python3
"""Model compression via distillation for the autoresearch pipeline.

Takes a large accurate model (teacher) and plans/evaluates a smaller
model (student) that matches its predictions. Measures the accuracy/size/
latency tradeoff to bridge "best research model" and "production-ready model."

Usage:
    python scripts/model_distiller.py exp-042
    python scripts/model_distiller.py exp-042 --compression 4
    python scripts/model_distiller.py exp-042 --method soft-labels
    python scripts/model_distiller.py exp-042 --target-latency 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_COMPRESSION = 4  # 4x compression
DISTILLATION_METHODS = ["soft_labels", "feature_matching", "dataset_distillation"]


# --- Student Architecture Selection ---


def select_student_architecture(
    teacher_config: dict,
    compression: float,
) -> dict:
    """Auto-select student architecture based on teacher and compression target.

    Args:
        teacher_config: Teacher model config.
        compression: Compression ratio (e.g., 4 = 4x smaller).

    Returns:
        Student config dict with model type and hyperparameters.
    """
    model_type = teacher_config.get("model_type", "").lower()
    hyperparams = teacher_config.get("hyperparams", {})

    student = {
        "model_type": model_type,
        "hyperparams": {},
        "compression_strategy": "",
    }

    if _is_tree_model(model_type):
        student.update(_select_tree_student(hyperparams, compression))
    elif _is_neural_model(model_type):
        student.update(_select_neural_student(hyperparams, compression))
    elif _is_sklearn_model(model_type):
        student.update(_select_sklearn_student(model_type, hyperparams, compression))
    else:
        # Generic: reduce all numeric hyperparams by compression ratio
        student["compression_strategy"] = "generic_reduction"
        student["hyperparams"] = {
            k: max(1, int(v / compression)) if isinstance(v, int) else v
            for k, v in hyperparams.items()
        }

    return student


def _is_tree_model(model_type: str) -> bool:
    return any(t in model_type for t in ("xgboost", "lightgbm", "catboost", "gbm", "gradient_boosting"))


def _is_neural_model(model_type: str) -> bool:
    return any(t in model_type for t in ("mlp", "neural", "nn", "pytorch", "tensorflow", "transformer"))


def _is_sklearn_model(model_type: str) -> bool:
    return any(t in model_type for t in ("random_forest", "svm", "knn", "logistic", "ridge"))


def _select_tree_student(hyperparams: dict, compression: float) -> dict:
    """Select student for tree-based models: fewer estimators, shallower."""
    n_estimators = hyperparams.get("n_estimators", 100)
    max_depth = hyperparams.get("max_depth", 6)

    return {
        "compression_strategy": "reduce_trees",
        "hyperparams": {
            "n_estimators": max(1, int(n_estimators / compression)),
            "max_depth": max(1, int(max_depth / math.sqrt(compression))),
            "learning_rate": hyperparams.get("learning_rate", 0.1),
        },
    }


def _select_neural_student(hyperparams: dict, compression: float) -> dict:
    """Select student for neural models: fewer layers, narrower."""
    hidden_size = hyperparams.get("hidden_size", 256)
    n_layers = hyperparams.get("n_layers", hyperparams.get("layers", 4))

    return {
        "compression_strategy": "reduce_architecture",
        "hyperparams": {
            "hidden_size": max(8, int(hidden_size / math.sqrt(compression))),
            "n_layers": max(1, int(n_layers / math.sqrt(compression))),
            "learning_rate": hyperparams.get("learning_rate", 0.001),
        },
    }


def _select_sklearn_student(model_type: str, hyperparams: dict, compression: float) -> dict:
    """Select student for sklearn models: simpler model family."""
    # Map complex models to simpler alternatives
    student_map = {
        "random_forest": "decision_tree",
        "svm": "logistic_regression",
        "knn": "logistic_regression",
    }

    student_type = student_map.get(model_type, model_type)

    student_params = {}
    if student_type == "decision_tree":
        student_params["max_depth"] = max(1, int(hyperparams.get("max_depth", 10) / compression))
    elif student_type == "logistic_regression":
        student_params["C"] = hyperparams.get("C", 1.0)

    return {
        "model_type": student_type,
        "compression_strategy": "simpler_family",
        "hyperparams": student_params,
    }


# --- Distillation Configuration ---


def plan_distillation(
    teacher_exp: dict,
    compression: float = DEFAULT_COMPRESSION,
    method: str = "soft_labels",
    target_latency: float | None = None,
) -> dict:
    """Plan a distillation run.

    Args:
        teacher_exp: Teacher experiment dict.
        compression: Compression ratio.
        method: Distillation method.
        target_latency: Optional target latency in ms.

    Returns:
        Distillation plan dict.
    """
    teacher_id = teacher_exp.get("experiment_id", "unknown")
    teacher_config = teacher_exp.get("config", {})
    teacher_metrics = teacher_exp.get("metrics", {})

    # Select student architecture
    student = select_student_architecture(teacher_config, compression)

    # Estimate student size
    teacher_size = teacher_metrics.get("model_size_bytes", teacher_metrics.get("n_params", 0))
    estimated_student_size = teacher_size / compression if teacher_size else None

    # Estimate student latency
    teacher_latency = teacher_metrics.get("latency_ms", teacher_metrics.get("inference_ms", 0))
    estimated_student_latency = teacher_latency / math.sqrt(compression) if teacher_latency else None

    # If target latency specified, adjust compression
    if target_latency and teacher_latency and teacher_latency > 0:
        needed_speedup = teacher_latency / target_latency
        adjusted_compression = needed_speedup ** 2  # Latency scales with sqrt(compression)
        if adjusted_compression > compression:
            compression = adjusted_compression
            student = select_student_architecture(teacher_config, compression)

    plan = {
        "teacher_id": teacher_id,
        "teacher_metrics": teacher_metrics,
        "teacher_config": teacher_config,
        "compression": round(compression, 2),
        "method": method,
        "student": student,
        "estimates": {
            "student_size_bytes": int(estimated_student_size) if estimated_student_size else None,
            "student_latency_ms": round(estimated_student_latency, 2) if estimated_student_latency else None,
            "size_reduction": f"{(1 - 1/compression) * 100:.0f}%" if compression > 0 else "N/A",
        },
        "distillation_config": _build_distillation_config(method),
    }

    return plan


def _build_distillation_config(method: str) -> dict:
    """Build distillation-specific configuration."""
    if method == "soft_labels":
        return {
            "temperature": 3.0,
            "alpha": 0.7,  # Weight of soft labels vs hard labels
            "description": "Train student on teacher's probability outputs with temperature scaling",
        }
    elif method == "feature_matching":
        return {
            "match_layers": "last_hidden",
            "loss": "mse",
            "description": "Align student's intermediate representations with teacher's",
        }
    elif method == "dataset_distillation":
        return {
            "synthetic_samples": 1000,
            "description": "Train student on teacher-labeled synthetic data",
        }
    return {"description": "Unknown method"}


# --- Verdict ---


def compute_distillation_verdict(
    teacher_metrics: dict,
    student_metrics: dict,
    primary_metric: str,
    compression: float,
) -> dict:
    """Compute verdict on distillation quality.

    Args:
        teacher_metrics: Teacher model metrics.
        student_metrics: Student model metrics.
        primary_metric: Name of primary metric.
        compression: Achieved compression ratio.

    Returns:
        Verdict dict.
    """
    teacher_val = teacher_metrics.get(primary_metric, 0)
    student_val = student_metrics.get(primary_metric, 0)

    if teacher_val == 0:
        return {"verdict": "no_baseline", "reason": "Teacher has no metric to compare against"}

    delta = student_val - teacher_val
    relative_loss = abs(delta) / abs(teacher_val) if teacher_val != 0 else 0

    if relative_loss < 0.01:  # < 1% accuracy loss
        verdict = "excellent"
        reason = f"{relative_loss:.1%} accuracy loss for {compression:.0f}x compression. Excellent tradeoff."
    elif relative_loss < 0.03:  # < 3% loss
        verdict = "acceptable"
        reason = f"{relative_loss:.1%} accuracy loss for {compression:.0f}x compression. Acceptable for production."
    elif relative_loss < 0.05:  # < 5% loss
        verdict = "marginal"
        reason = f"{relative_loss:.1%} accuracy loss for {compression:.0f}x compression. Consider lower compression."
    else:
        verdict = "too_much_loss"
        reason = f"{relative_loss:.1%} accuracy loss for {compression:.0f}x compression. Try a less aggressive compression."

    return {
        "verdict": verdict,
        "delta": round(delta, 6),
        "relative_loss": round(relative_loss, 6),
        "compression": compression,
        "reason": reason,
    }


# --- Full Pipeline ---


def distill_model(
    teacher_exp_id: str,
    compression: float = DEFAULT_COMPRESSION,
    method: str = "soft_labels",
    target_latency: float | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Plan and report a model distillation.

    Args:
        teacher_exp_id: Teacher experiment ID.
        compression: Compression ratio.
        method: Distillation method.
        target_latency: Target inference latency in ms.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.

    Returns:
        Complete distillation report.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")

    experiments = load_experiments(log_path)

    teacher = None
    for exp in experiments:
        if exp.get("experiment_id") == teacher_exp_id:
            teacher = exp
            break

    if not teacher:
        return {"error": f"Teacher experiment {teacher_exp_id} not found in {log_path}"}

    plan = plan_distillation(teacher, compression, method, target_latency)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": primary_metric,
        "plan": plan,
    }

    return report


# --- Report Formatting ---


def save_distillation_report(report: dict, output_dir: str = "experiments/distillations") -> Path:
    """Save distillation report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    teacher_id = report.get("plan", {}).get("teacher_id", "unknown")
    filepath = out_path / f"distill-{teacher_id}.yaml"

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_distillation_report(report: dict) -> str:
    """Format distillation report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    plan = report.get("plan", {})
    teacher_id = plan.get("teacher_id", "?")
    compression = plan.get("compression", 0)
    method = plan.get("method", "?")
    student = plan.get("student", {})
    estimates = plan.get("estimates", {})
    dist_cfg = plan.get("distillation_config", {})

    lines = [
        f"# Distillation Plan: {teacher_id}",
        "",
        f"*Generated {report.get('generated_at', 'N/A')[:19]}*",
        "",
        f"**Compression:** {compression:.0f}x",
        f"**Method:** {method}",
        f"**Strategy:** {student.get('compression_strategy', '?')}",
        "",
    ]

    # Teacher info
    teacher_metrics = plan.get("teacher_metrics", {})
    if teacher_metrics:
        lines.extend(["## Teacher Model", ""])
        for k, v in teacher_metrics.items():
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"- **{k}:** {v_str}")
        lines.append("")

    # Student architecture
    lines.extend(["## Student Architecture", ""])
    lines.append(f"- **Model type:** {student.get('model_type', '?')}")
    for k, v in student.get("hyperparams", {}).items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")

    # Estimates
    if any(v is not None for v in estimates.values()):
        lines.extend(["## Estimates", ""])
        if estimates.get("size_reduction"):
            lines.append(f"- **Size reduction:** {estimates['size_reduction']}")
        if estimates.get("student_latency_ms"):
            lines.append(f"- **Estimated latency:** {estimates['student_latency_ms']:.1f} ms")
        lines.append("")

    # Distillation config
    lines.extend([
        "## Distillation Config",
        "",
        f"*{dist_cfg.get('description', method)}*",
        "",
    ])
    for k, v in dist_cfg.items():
        if k != "description":
            lines.append(f"- **{k}:** {v}")

    # Verdict (if student metrics available)
    verdict = report.get("verdict")
    if verdict:
        labels = {
            "excellent": "EXCELLENT",
            "acceptable": "ACCEPTABLE",
            "marginal": "MARGINAL",
            "too_much_loss": "TOO MUCH LOSS",
        }
        lines.extend([
            "",
            "## Verdict",
            "",
            f"**{labels.get(verdict.get('verdict', ''), verdict.get('verdict', '?'))}**",
            "",
            verdict.get("reason", ""),
        ])

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Model compression via distillation",
    )
    parser.add_argument(
        "teacher_exp_id",
        help="Teacher experiment ID (e.g., exp-042)",
    )
    parser.add_argument(
        "--compression", type=float, default=DEFAULT_COMPRESSION,
        help=f"Compression ratio (default: {DEFAULT_COMPRESSION}x)",
    )
    parser.add_argument(
        "--method", choices=DISTILLATION_METHODS, default="soft_labels",
        help="Distillation method (default: soft_labels)",
    )
    parser.add_argument(
        "--target-latency", type=float,
        help="Target inference latency in ms (auto-adjusts compression)",
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

    report = distill_model(
        teacher_exp_id=args.teacher_exp_id,
        compression=args.compression,
        method=args.method,
        target_latency=args.target_latency,
        config_path=args.config,
        log_path=args.log,
    )

    if "error" not in report:
        filepath = save_distillation_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_distillation_report(report))


if __name__ == "__main__":
    main()
