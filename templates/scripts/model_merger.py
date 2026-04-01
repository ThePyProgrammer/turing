#!/usr/bin/env python3
"""Model merging for the autoresearch pipeline.

Average or merge weights from multiple fine-tuned checkpoints into a
single model (model soups, TIES, DARE, greedy soup). Often beats any
individual model with zero additional training cost and no latency overhead.

Usage:
    python scripts/model_merger.py exp-042 exp-053 exp-067
    python scripts/model_merger.py exp-042 exp-053 --method greedy
    python scripts/model_merger.py --json
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
MERGE_METHODS = ["uniform", "greedy", "ties", "dare"]


def check_compatibility(experiments: list[dict]) -> dict:
    """Check that all models share the same architecture."""
    model_types = {e.get("config", {}).get("model_type", "?") for e in experiments}
    compatible = len(model_types) == 1
    return {
        "compatible": compatible,
        "model_types": list(model_types),
        "n_models": len(experiments),
        "reason": "All models share same architecture" if compatible else f"Mixed architectures: {model_types}",
    }


def plan_uniform_merge(
    experiments: list[dict],
    primary_metric: str,
) -> dict:
    """Plan uniform weight averaging (model soup)."""
    metrics = [e.get("metrics", {}).get(primary_metric, 0) for e in experiments]
    return {
        "method": "uniform",
        "description": "Simple average of all model weights",
        "n_models": len(experiments),
        "individual_metrics": [{"exp_id": e.get("experiment_id"), primary_metric: m} for e, m in zip(experiments, metrics)],
        "weights": [round(1.0 / len(experiments), 4)] * len(experiments),
    }


def plan_greedy_merge(
    experiments: list[dict],
    primary_metric: str,
    merge_results: list[dict] | None = None,
) -> dict:
    """Plan greedy soup — iteratively add models only if they improve the merge."""
    sorted_exps = sorted(experiments, key=lambda e: e.get("metrics", {}).get(primary_metric, 0), reverse=True)
    included = [sorted_exps[0].get("experiment_id")]
    excluded = []

    if merge_results:
        # Use actual results to determine inclusion
        for r in merge_results[1:]:
            if r.get("improved", True):
                included.append(r.get("exp_id"))
            else:
                excluded.append(r.get("exp_id"))
    else:
        # Plan: include all by default, actual filtering done at execution
        included = [e.get("experiment_id") for e in sorted_exps]

    return {
        "method": "greedy",
        "description": "Iteratively add models only if they improve the merged result",
        "included": included,
        "excluded": excluded,
        "n_included": len(included),
        "n_excluded": len(excluded),
    }


def plan_ties_merge(experiments: list[dict]) -> dict:
    """Plan TIES merging (Trim, Elect sign, disjoint Merge)."""
    return {
        "method": "ties",
        "description": "Trim redundant params, elect sign consensus, disjoint merge",
        "n_models": len(experiments),
        "steps": [
            "1. Compute task vectors (delta from base) for each model",
            "2. Trim: zero out smallest magnitude deltas",
            "3. Elect: resolve sign conflicts by majority vote",
            "4. Merge: average the surviving, sign-consistent deltas",
        ],
    }


def plan_dare_merge(experiments: list[dict]) -> dict:
    """Plan DARE merging (Drop And REscale)."""
    return {
        "method": "dare",
        "description": "Randomly drop parameters and rescale survivors to reduce interference",
        "n_models": len(experiments),
        "drop_rate": 0.5,
        "steps": [
            "1. Compute task vectors for each model",
            "2. Randomly drop 50% of parameters per model",
            "3. Rescale surviving parameters by 1/(1-drop_rate)",
            "4. Average the rescaled task vectors",
        ],
    }


def compare_merge_methods(
    method_results: dict[str, dict] | None = None,
    experiments: list[dict] | None = None,
    primary_metric: str = "accuracy",
) -> dict:
    """Compare merge method results."""
    if not experiments:
        return {"error": "No experiments provided"}

    # Best single model
    best_single = max(experiments, key=lambda e: e.get("metrics", {}).get(primary_metric, 0))
    best_metric = best_single.get("metrics", {}).get(primary_metric, 0)

    results = [{
        "method": "best_single",
        "metric_value": best_metric,
        "delta": 0.0,
        "experiment_id": best_single.get("experiment_id"),
    }]

    if method_results:
        for method_name, data in method_results.items():
            metric = data.get("metric_value", data.get(primary_metric, 0))
            results.append({
                "method": method_name,
                "metric_value": metric,
                "delta": round(metric - best_metric, 6),
            })

    best_merge = max(results, key=lambda r: r.get("metric_value", 0))

    return {
        "results": results,
        "best_method": best_merge.get("method"),
        "best_metric": best_merge.get("metric_value"),
        "improvement": best_merge.get("delta", 0),
    }


def merge_analysis(
    exp_ids: list[str] | None = None,
    method_results: dict[str, dict] | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Run merge analysis."""
    config = load_config(config_path)
    primary_metric = config.get("evaluation", {}).get("primary_metric", "accuracy")
    experiments = load_experiments(log_path)

    if exp_ids:
        selected = [e for e in experiments if e.get("experiment_id") in exp_ids]
    else:
        # Default: top 3 kept experiments
        kept = sorted(
            [e for e in experiments if e.get("status") == "kept"],
            key=lambda e: e.get("metrics", {}).get(primary_metric, 0), reverse=True,
        )
        selected = kept[:3]

    if len(selected) < 2:
        return {"error": "Need at least 2 experiments for model merging"}

    compat = check_compatibility(selected)

    plans = {
        "uniform": plan_uniform_merge(selected, primary_metric),
        "greedy": plan_greedy_merge(selected, primary_metric),
        "ties": plan_ties_merge(selected),
        "dare": plan_dare_merge(selected),
    }

    comparison = compare_merge_methods(method_results, selected, primary_metric) if method_results else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": primary_metric,
        "compatibility": compat,
        "base_models": [{"exp_id": e.get("experiment_id"),
                        "model_type": e.get("config", {}).get("model_type"),
                        primary_metric: e.get("metrics", {}).get(primary_metric)}
                       for e in selected],
        "plans": plans,
        "comparison": comparison,
    }


def save_merge_report(report: dict, output_dir: str = "experiments/merges") -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fp = out / f"merge-{ts}.yaml"
    with open(fp, "w") as f: yaml.dump(json.loads(json.dumps(report, default=str)), f, default_flow_style=False, sort_keys=False)
    return fp


def format_merge_report(report: dict) -> str:
    if "error" in report: return f"ERROR: {report['error']}"

    metric = report.get("primary_metric", "metric")
    lines = ["# Model Merge Analysis", "",
             f"*Generated {report.get('generated_at', 'N/A')[:19]}*", ""]

    # Compatibility
    compat = report.get("compatibility", {})
    lines.append(f"**Compatibility:** {'✓' if compat.get('compatible') else '✗'} {compat.get('reason', '')}")
    lines.append("")

    # Base models
    lines.extend(["## Base Models", "",
                  f"| Experiment | Model Type | {metric} |",
                  "|------------|------------|--------|"])
    for m in report.get("base_models", []):
        val = m.get(metric, "N/A")
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        lines.append(f"| {m.get('exp_id', '?')} | {m.get('model_type', '?')} | {val_str} |")
    lines.append("")

    # Methods
    plans = report.get("plans", {})
    if plans:
        lines.extend(["## Available Methods", ""])
        for name, plan in plans.items():
            lines.append(f"- **{name}:** {plan.get('description', '')}")
        lines.append("")

    # Comparison (if results available)
    comparison = report.get("comparison")
    if comparison:
        lines.extend(["## Results", "",
                      f"| Method | {metric} | Δ vs Best Single |",
                      "|--------|--------|------------------|"])
        for r in comparison.get("results", []):
            val = f"{r.get('metric_value', 0):.4f}"
            delta = f"{r.get('delta', 0):+.4f}" if r.get("delta") is not None else "—"
            marker = " ← BEST" if r["method"] == comparison.get("best_method") and r["method"] != "best_single" else ""
            lines.append(f"| {r['method']} | {val} | {delta} |{marker}")
        lines.append("")
        imp = comparison.get("improvement", 0)
        if imp > 0:
            lines.append(f"**{comparison['best_method']} improves by {imp:+.4f} over best single model — zero latency cost.**")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model merging")
    parser.add_argument("exp_ids", nargs="*", help="Experiment IDs to merge")
    parser.add_argument("--method", choices=MERGE_METHODS, help="Specific merge method")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = merge_analysis(exp_ids=args.exp_ids or None, config_path=args.config, log_path=args.log)
    if "error" not in report:
        fp = save_merge_report(report); print(f"Saved to {fp}", file=sys.stderr)
    print(json.dumps(report, indent=2, default=str) if args.json else format_merge_report(report))

if __name__ == "__main__": main()
