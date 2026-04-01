#!/usr/bin/env python3
"""Weight pruning for the autoresearch pipeline.

Structured and unstructured weight pruning. Measures accuracy at different
sparsity levels, finds the knee point, and plans pruned model production.

Usage:
    python scripts/model_pruning.py exp-042
    python scripts/model_pruning.py exp-042 --sparsity 0.5,0.75,0.9
    python scripts/model_pruning.py exp-042 --method magnitude
    python scripts/model_pruning.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_SPARSITY_LEVELS = [0.0, 0.50, 0.75, 0.90, 0.95]
PRUNING_METHODS = ["magnitude", "structured", "lottery"]


def plan_sparsity_sweep(
    sparsity_levels: list[float] | None = None,
) -> list[dict]:
    if sparsity_levels is None:
        sparsity_levels = DEFAULT_SPARSITY_LEVELS
    return [{"sparsity": s, "description": f"{s*100:.0f}% weights removed"} for s in sparsity_levels]


def compute_pruning_plan(
    model_type: str,
    hyperparams: dict,
    method: str,
    sparsity: float,
) -> dict:
    plan = {"method": method, "sparsity": sparsity, "config_changes": {}}
    if "xgboost" in model_type.lower() or "lightgbm" in model_type.lower() or "forest" in model_type.lower():
        n_est = hyperparams.get("n_estimators", 100)
        plan["config_changes"]["n_estimators"] = max(1, int(n_est * (1 - sparsity)))
        plan["strategy"] = "reduce_estimators"
    elif method == "magnitude":
        plan["strategy"] = "zero_small_weights"
        plan["description"] = f"Zero out smallest {sparsity*100:.0f}% of weights by absolute value"
    elif method == "structured":
        plan["strategy"] = "remove_neurons"
        plan["description"] = f"Remove {sparsity*100:.0f}% of neurons/filters by importance"
    elif method == "lottery":
        plan["strategy"] = "iterative_magnitude_with_rewind"
        plan["description"] = f"Iterative pruning to {sparsity*100:.0f}% with weight rewinding"
    return plan


def find_knee_point(sweep_results: list[dict], metric_key: str = "accuracy") -> dict | None:
    if len(sweep_results) < 3:
        return None
    sparsities = [r["sparsity"] for r in sweep_results]
    metrics = [r.get(metric_key, 0) for r in sweep_results]
    max_drop = 0
    knee_idx = None
    for i in range(1, len(metrics)):
        drop = metrics[i - 1] - metrics[i]
        if drop > max_drop:
            max_drop = drop
            knee_idx = i
    if knee_idx and knee_idx > 0:
        return {"sparsity": sparsities[knee_idx - 1],
                "metric_before_knee": round(metrics[knee_idx - 1], 6),
                "metric_after_knee": round(metrics[knee_idx], 6),
                "drop_at_knee": round(max_drop, 6)}
    return None


def estimate_speedup(sparsity: float) -> float:
    if sparsity <= 0:
        return 1.0
    return round(1.0 / (1.0 - sparsity * 0.7), 2)


def estimate_size_reduction(sparsity: float) -> float:
    return round(sparsity * 100, 1)


def analyze_pruning(
    sweep_results: list[dict] | None = None,
    exp_id: str | None = None,
    method: str = "magnitude",
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    config = load_config(config_path)
    primary_metric = config.get("evaluation", {}).get("primary_metric", "accuracy")

    if sweep_results:
        knee = find_knee_point(sweep_results, primary_metric)
        for r in sweep_results:
            r["speedup"] = estimate_speedup(r["sparsity"])
            r["size_reduction_pct"] = estimate_size_reduction(r["sparsity"])
        recommended = None
        for r in sweep_results:
            delta = abs(r.get(primary_metric, 0) - sweep_results[0].get(primary_metric, 0))
            if delta < 0.005 and r["sparsity"] > 0:
                recommended = r
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiment_id": exp_id, "method": method, "primary_metric": primary_metric,
            "sweep_results": sweep_results, "knee_point": knee,
            "recommended": recommended,
        }

    experiments = load_experiments(log_path)
    exp = next((e for e in experiments if e.get("experiment_id") == exp_id), None) if exp_id else None
    model_type = exp.get("config", {}).get("model_type", "unknown") if exp else "unknown"
    hyperparams = exp.get("config", {}).get("hyperparams", {}) if exp else {}

    levels = plan_sparsity_sweep()
    plans = [compute_pruning_plan(model_type, hyperparams, method, s["sparsity"]) for s in levels]
    return {
        "action": "plan", "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": exp_id, "model_type": model_type, "method": method,
        "sparsity_levels": levels, "plans": plans,
        "message": f"Run {len(levels)} experiments at sparsity levels: {', '.join(s['description'] for s in levels)}",
    }


def save_pruning_report(report: dict, output_dir: str = "experiments/pruning") -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("experiment_id", "unknown")
    fp = out / f"{exp_id}-pruning.yaml"
    with open(fp, "w") as f: yaml.dump(json.loads(json.dumps(report, default=str)), f, default_flow_style=False, sort_keys=False)
    return fp


def format_pruning_report(report: dict) -> str:
    if "error" in report: return f"ERROR: {report['error']}"
    if report.get("action") == "plan":
        lines = ["# Pruning Plan", "", f"**Model:** {report.get('model_type', '?')}", f"**Method:** {report.get('method', '?')}", ""]
        for p in report.get("plans", []):
            lines.append(f"- {p.get('sparsity', 0)*100:.0f}%: {p.get('strategy', '?')}")
        return "\n".join(lines)

    metric = report.get("primary_metric", "metric")
    lines = [f"# Pruning Results: {report.get('experiment_id', '?')}", "",
             f"| Sparsity | {metric} | Speedup | Size Reduction |",
             "|----------|--------|---------|----------------|"]
    for r in report.get("sweep_results", []):
        val = f"{r.get(metric, 0):.4f}" if isinstance(r.get(metric), (int, float)) else "N/A"
        lines.append(f"| {r['sparsity']*100:.0f}% | {val} | {r.get('speedup', '?')}x | {r.get('size_reduction_pct', '?')}% |")
    knee = report.get("knee_point")
    if knee:
        lines.extend(["", f"**Knee point:** {knee['sparsity']*100:.0f}% sparsity (accuracy drops {knee['drop_at_knee']:.4f})"])
    rec = report.get("recommended")
    if rec:
        lines.extend(["", f"**Recommended:** {rec['sparsity']*100:.0f}% sparsity ({rec.get('speedup', '?')}x speedup, <0.5% accuracy loss)"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weight pruning")
    parser.add_argument("exp_id", nargs="?")
    parser.add_argument("--sparsity", help="Comma-separated sparsity levels")
    parser.add_argument("--method", choices=PRUNING_METHODS, default="magnitude")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = analyze_pruning(exp_id=args.exp_id, method=args.method, config_path=args.config, log_path=args.log)
    if "error" not in report:
        fp = save_pruning_report(report); print(f"Saved to {fp}", file=sys.stderr)
    print(json.dumps(report, indent=2, default=str) if args.json else format_pruning_report(report))

if __name__ == "__main__": main()
