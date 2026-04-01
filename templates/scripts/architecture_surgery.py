#!/usr/bin/env python3
"""Architecture modification for the autoresearch pipeline.

Programmatic architecture changes: add/remove layers, widen/narrow,
swap activation functions, inject skip connections, change normalization.
Produces a modified config and instructions for the modified experiment.

Usage:
    python scripts/architecture_surgery.py exp-042 --op widen 2
    python scripts/architecture_surgery.py exp-042 --op add-layer
    python scripts/architecture_surgery.py exp-042 --op swap-activation relu gelu
    python scripts/architecture_surgery.py --json
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
OPERATIONS = ["add-layer", "remove-layer", "widen", "narrow", "swap-activation",
              "add-skip", "add-norm", "deepen", "swap-objective"]


def plan_operation(
    operation: str,
    config: dict,
    hyperparams: dict,
    model_type: str,
    args: list[str] | None = None,
) -> dict:
    """Plan an architecture modification.

    Returns a plan dict with new config, parameter count change, and instructions.
    """
    args = args or []
    plan = {
        "operation": operation,
        "model_type": model_type,
        "original_config": hyperparams.copy(),
        "new_config": hyperparams.copy(),
        "instructions": [],
        "param_change": None,
    }

    is_tree = any(t in model_type.lower() for t in ("xgboost", "lightgbm", "forest", "gbm", "catboost"))
    is_neural = any(t in model_type.lower() for t in ("mlp", "nn", "pytorch", "tensorflow", "transformer"))

    if operation == "widen":
        factor = float(args[0]) if args else 2.0
        if is_neural:
            hs = hyperparams.get("hidden_size", 256)
            new_hs = int(hs * factor)
            plan["new_config"]["hidden_size"] = new_hs
            plan["instructions"].append(f"Multiply hidden dimensions: {hs} → {new_hs} ({factor}x)")
            plan["param_change"] = f"+{(factor**2 - 1)*100:.0f}% parameters (quadratic in width)"
        elif is_tree:
            n = hyperparams.get("n_estimators", 100)
            new_n = int(n * factor)
            plan["new_config"]["n_estimators"] = new_n
            plan["instructions"].append(f"Increase estimators: {n} → {new_n}")
            plan["param_change"] = f"+{(factor - 1)*100:.0f}% trees"
        else:
            plan["instructions"].append(f"Widen by {factor}x — adjust model-specific width parameters")

    elif operation == "narrow":
        factor = float(args[0]) if args else 0.5
        if is_neural:
            hs = hyperparams.get("hidden_size", 256)
            new_hs = max(8, int(hs * factor))
            plan["new_config"]["hidden_size"] = new_hs
            plan["instructions"].append(f"Reduce hidden dimensions: {hs} → {new_hs} ({factor}x)")
        elif is_tree:
            n = hyperparams.get("n_estimators", 100)
            new_n = max(1, int(n * factor))
            plan["new_config"]["n_estimators"] = new_n
            plan["instructions"].append(f"Reduce estimators: {n} → {new_n}")

    elif operation == "add-layer":
        if is_neural:
            n_layers = hyperparams.get("n_layers", hyperparams.get("layers", 3))
            plan["new_config"]["n_layers"] = n_layers + 1
            plan["instructions"].extend([
                f"Add layer: {n_layers} → {n_layers + 1}",
                "New layer initialized with default weights",
                "Auto warm-start: existing layers loaded from source",
            ])
            plan["param_change"] = f"+1 layer ({n_layers} → {n_layers + 1})"
        else:
            plan["instructions"].append("add-layer not applicable for non-neural models")

    elif operation == "remove-layer":
        if is_neural:
            n_layers = hyperparams.get("n_layers", hyperparams.get("layers", 3))
            if n_layers > 1:
                plan["new_config"]["n_layers"] = n_layers - 1
                plan["instructions"].append(f"Remove layer: {n_layers} → {n_layers - 1}")
            else:
                plan["instructions"].append("Cannot remove — only 1 layer remaining")
        else:
            plan["instructions"].append("remove-layer not applicable for non-neural models")

    elif operation == "deepen":
        if is_tree:
            depth = hyperparams.get("max_depth", 6)
            new_depth = depth + 2
            plan["new_config"]["max_depth"] = new_depth
            plan["instructions"].append(f"Increase max depth: {depth} → {new_depth}")
        elif is_neural:
            n_layers = hyperparams.get("n_layers", 3)
            plan["new_config"]["n_layers"] = n_layers + 2
            plan["instructions"].append(f"Add 2 layers: {n_layers} → {n_layers + 2}")

    elif operation == "swap-activation":
        if len(args) >= 2:
            from_act, to_act = args[0], args[1]
        else:
            from_act, to_act = "relu", "gelu"
        plan["new_config"]["activation"] = to_act
        plan["instructions"].append(f"Swap activation: {from_act} → {to_act}")

    elif operation == "add-skip":
        plan["new_config"]["skip_connections"] = True
        plan["instructions"].append("Inject residual/skip connections between layers")

    elif operation == "add-norm":
        norm_type = args[0] if args else "batch_norm"
        plan["new_config"]["normalization"] = norm_type
        plan["instructions"].append(f"Add {norm_type} after each layer")

    elif operation == "swap-objective":
        if len(args) >= 2:
            from_obj, to_obj = args[0], args[1]
        else:
            from_obj, to_obj = hyperparams.get("objective", "logloss"), "focal"
        plan["new_config"]["objective"] = to_obj
        plan["instructions"].append(f"Swap objective: {from_obj} → {to_obj}")

    else:
        plan["instructions"].append(f"Unknown operation: {operation}")

    return plan


def surgery_report(
    exp_id: str,
    operation: str,
    op_args: list[str] | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Generate a surgery report."""
    experiments = load_experiments(log_path)
    exp = next((e for e in experiments if e.get("experiment_id") == exp_id), None)

    if not exp:
        return {"error": f"Experiment {exp_id} not found"}

    config = exp.get("config", {})
    model_type = config.get("model_type", "unknown")
    hyperparams = config.get("hyperparams", {})

    plan = plan_operation(operation, config, hyperparams, model_type, op_args)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": exp_id,
        "plan": plan,
        "warm_start_from": exp_id,
    }


def save_surgery_report(report: dict, output_dir: str = "experiments/surgery") -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("experiment_id", "unknown")
    op = report.get("plan", {}).get("operation", "unknown")
    fp = out / f"{exp_id}-{op}.yaml"
    with open(fp, "w") as f: yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return fp


def format_surgery_report(report: dict) -> str:
    if "error" in report: return f"ERROR: {report['error']}"

    plan = report.get("plan", {})
    exp_id = report.get("experiment_id", "?")
    op = plan.get("operation", "?")

    lines = [f"# Surgery: {op} ({exp_id})", "",
             f"*Generated {report.get('generated_at', 'N/A')[:19]}*",
             f"**Model:** {plan.get('model_type', '?')}", ""]

    lines.extend(["## Instructions", ""])
    for i, inst in enumerate(plan.get("instructions", []), 1):
        lines.append(f"{i}. {inst}")
    lines.append("")

    if plan.get("param_change"):
        lines.append(f"**Parameter change:** {plan['param_change']}")
        lines.append("")

    orig = plan.get("original_config", {})
    new = plan.get("new_config", {})
    changed = {k: (orig.get(k), new[k]) for k in new if orig.get(k) != new.get(k)}
    if changed:
        lines.extend(["## Config Changes", ""])
        for k, (old, new_v) in changed.items():
            lines.append(f"- `{k}`: {old} → {new_v}")
        lines.append("")

    lines.append(f"**Warm-start from:** {report.get('warm_start_from', '?')}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture modification")
    parser.add_argument("exp_id")
    parser.add_argument("--op", required=True, help="Operation name")
    parser.add_argument("op_args", nargs="*", help="Operation arguments")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = surgery_report(args.exp_id, args.op, args.op_args, args.config, args.log)
    if "error" not in report:
        fp = save_surgery_report(report); print(f"Saved to {fp}", file=sys.stderr)
    print(json.dumps(report, indent=2, default=str) if args.json else format_surgery_report(report))

if __name__ == "__main__": main()
