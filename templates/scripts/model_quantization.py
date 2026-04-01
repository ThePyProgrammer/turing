#!/usr/bin/env python3
"""Post-training quantization for the autoresearch pipeline.

Quantize model weights from FP32 to INT8/FP16, measure accuracy loss
per precision level, and plan quantization-aware training if needed.

Usage:
    python scripts/model_quantization.py exp-042
    python scripts/model_quantization.py exp-042 --precision int8
    python scripts/model_quantization.py --json
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
PRECISION_LEVELS = ["fp32", "fp16", "int8_dynamic", "int8_static"]
QAT_THRESHOLD = 0.01  # If PTQ accuracy loss > 1%, suggest QAT


def compute_quantization_plan(
    precision: str,
    model_size_bytes: int | None = None,
    latency_ms: float | None = None,
) -> dict:
    size_factors = {"fp32": 1.0, "fp16": 0.5, "int8_dynamic": 0.25, "int8_static": 0.25}
    latency_factors = {"fp32": 1.0, "fp16": 0.58, "int8_dynamic": 0.39, "int8_static": 0.37}

    factor_s = size_factors.get(precision, 1.0)
    factor_l = latency_factors.get(precision, 1.0)

    plan = {
        "precision": precision,
        "size_factor": factor_s,
        "latency_factor": factor_l,
        "estimated_size_bytes": int(model_size_bytes * factor_s) if model_size_bytes else None,
        "estimated_latency_ms": round(latency_ms * factor_l, 2) if latency_ms else None,
        "size_reduction_pct": round((1 - factor_s) * 100, 1),
        "speedup": round(1 / factor_l, 2) if factor_l > 0 else None,
    }

    if precision == "fp16":
        plan["description"] = "Half-precision floating point — GPU inference"
        plan["method"] = "cast_to_fp16"
    elif precision == "int8_dynamic":
        plan["description"] = "Dynamic INT8 — weights quantized, activations at runtime"
        plan["method"] = "dynamic_quantization"
    elif precision == "int8_static":
        plan["description"] = "Static INT8 — calibrated activation ranges, best accuracy"
        plan["method"] = "static_quantization"
        plan["requires_calibration"] = True
    else:
        plan["description"] = "Full precision (baseline)"
        plan["method"] = "none"

    return plan


def compare_precision_levels(
    sweep_results: list[dict] | None = None,
    model_size_bytes: int | None = None,
    latency_ms: float | None = None,
    primary_metric: str = "accuracy",
) -> dict:
    """Compare quantization results across precision levels."""
    if sweep_results:
        baseline = next((r for r in sweep_results if r.get("precision") == "fp32"), sweep_results[0])
        baseline_metric = baseline.get(primary_metric, 0)

        for r in sweep_results:
            r["delta"] = round(r.get(primary_metric, 0) - baseline_metric, 6)
            plan = compute_quantization_plan(r["precision"], model_size_bytes, latency_ms)
            r.update({k: v for k, v in plan.items() if k not in r})

        best = min(
            [r for r in sweep_results if r["precision"] != "fp32"],
            key=lambda r: abs(r.get("delta", 0)) + (1 - r.get("speedup", 1)) * 0.1,
            default=None,
        )

        needs_qat = any(abs(r.get("delta", 0)) > QAT_THRESHOLD for r in sweep_results if "int8" in r.get("precision", ""))

        return {
            "sweep_results": sweep_results,
            "recommended": best,
            "needs_qat": needs_qat,
        }

    # Plan mode
    plans = [compute_quantization_plan(p, model_size_bytes, latency_ms) for p in PRECISION_LEVELS]
    return {"action": "plan", "plans": plans}


def analyze_quantization(
    sweep_results: list[dict] | None = None,
    exp_id: str | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    config = load_config(config_path)
    primary_metric = config.get("evaluation", {}).get("primary_metric", "accuracy")

    experiments = load_experiments(log_path)
    exp = next((e for e in experiments if e.get("experiment_id") == exp_id), None) if exp_id else None

    model_size = exp.get("metrics", {}).get("model_size_bytes") if exp else None
    latency = exp.get("metrics", {}).get("latency_ms", exp.get("metrics", {}).get("inference_ms")) if exp else None

    comparison = compare_precision_levels(sweep_results, model_size, latency, primary_metric)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": exp_id,
        "primary_metric": primary_metric,
        **comparison,
    }


def save_quantization_report(report: dict, output_dir: str = "experiments/quantization") -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("experiment_id", "unknown")
    fp = out / f"{exp_id}-quantization.yaml"
    with open(fp, "w") as f: yaml.dump(json.loads(json.dumps(report, default=str)), f, default_flow_style=False, sort_keys=False)
    return fp


def format_quantization_report(report: dict) -> str:
    if "error" in report: return f"ERROR: {report['error']}"

    if report.get("action") == "plan":
        lines = ["# Quantization Plan", ""]
        for p in report.get("plans", []):
            lines.append(f"- **{p['precision']}**: {p['description']} (size: {p['size_reduction_pct']}% reduction, speedup: {p.get('speedup', '?')}x)")
        return "\n".join(lines)

    metric = report.get("primary_metric", "metric")
    lines = [f"# Quantization Results: {report.get('experiment_id', '?')}", "",
             f"| Precision | {metric} | Delta | Speedup | Size Reduction |",
             "|-----------|--------|-------|---------|----------------|"]
    for r in report.get("sweep_results", []):
        val = f"{r.get(metric, 0):.4f}" if isinstance(r.get(metric), (int, float)) else "N/A"
        delta = f"{r.get('delta', 0):+.4f}" if r.get("delta") is not None else "—"
        lines.append(f"| {r['precision']} | {val} | {delta} | {r.get('speedup', '?')}x | {r.get('size_reduction_pct', '?')}% |")

    rec = report.get("recommended")
    if rec:
        lines.extend(["", f"**Recommended:** {rec['precision']} ({rec.get('delta', 0):+.4f} accuracy, {rec.get('speedup', '?')}x speedup)"])
    if report.get("needs_qat"):
        lines.extend(["", "**Note:** INT8 accuracy loss > 1% — consider quantization-aware training (QAT)"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training quantization")
    parser.add_argument("exp_id", nargs="?")
    parser.add_argument("--precision", help="Specific precision level")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = analyze_quantization(exp_id=args.exp_id, config_path=args.config, log_path=args.log)
    if "error" not in report:
        fp = save_quantization_report(report); print(f"Saved to {fp}", file=sys.stderr)
    print(json.dumps(report, indent=2, default=str) if args.json else format_quantization_report(report))

if __name__ == "__main__": main()
