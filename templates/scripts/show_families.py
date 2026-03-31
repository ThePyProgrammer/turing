#!/usr/bin/env python3
"""Experiment family viewer for the autoresearch pipeline.

Groups experiments by strategic theme (family tag) and shows
per-family performance summaries. Tells the human when an entire
research direction is exhausted.

Usage:
    python scripts/show_families.py [--log experiments/log.jsonl] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from scripts.turing_io import load_experiments


def group_by_family(experiments: list[dict]) -> dict[str, list[dict]]:
    """Group experiments by family tag."""
    families: dict[str, list[dict]] = {}
    for exp in experiments:
        family = exp.get("family") or "untagged"
        families.setdefault(family, []).append(exp)
    return families


def family_stats(
    experiments: list[dict],
    metric: str,
    lower_is_better: bool,
) -> dict:
    """Compute stats for a family of experiments."""
    total = len(experiments)
    kept = [e for e in experiments if e.get("status") == "kept"]
    discarded = [e for e in experiments if e.get("status") == "discarded"]

    metric_vals = [
        e.get("metrics", {}).get(metric)
        for e in kept
        if e.get("metrics", {}).get(metric) is not None
    ]

    best_val = None
    best_id = None
    if metric_vals:
        if lower_is_better:
            best_val = min(metric_vals)
        else:
            best_val = max(metric_vals)
        for e in kept:
            if e.get("metrics", {}).get(metric) == best_val:
                best_id = e.get("experiment_id")
                break

    # Compute recent trend (last 3 experiments)
    recent = experiments[-3:] if len(experiments) >= 3 else experiments
    recent_kept = sum(1 for e in recent if e.get("status") == "kept")
    recent_rate = recent_kept / len(recent) if recent else 0

    return {
        "total": total,
        "kept": len(kept),
        "discarded": len(discarded),
        "keep_rate": round(len(kept) / total, 2) if total > 0 else 0,
        "best_metric": best_val,
        "best_experiment": best_id,
        "recent_keep_rate": round(recent_rate, 2),
        "exhausted": total >= 3 and recent_rate == 0,
    }


def format_families(
    families: dict[str, dict],
    metric_name: str,
) -> str:
    """Format family summaries as a table."""
    if not families:
        return "No experiments logged yet."

    lines = [
        f"{'Family':<25} {'Total':>6} {'Kept':>6} {'Rate':>6} {'Best ' + metric_name:>15} {'Status':<12}",
        "-" * 80,
    ]

    for name, stats in sorted(families.items(), key=lambda x: -(x[1].get("best_metric") or 0)):
        best_str = f"{stats['best_metric']:.4f}" if stats["best_metric"] is not None else "N/A"
        status = "EXHAUSTED" if stats["exhausted"] else "active"
        line = f"{name:<25} {stats['total']:>6} {stats['kept']:>6} {stats['keep_rate']:>5.0%} {best_str:>15} {status:<12}"
        lines.append(line)

    # Summary
    total_exps = sum(s["total"] for s in families.values())
    exhausted = sum(1 for s in families.values() if s["exhausted"])
    lines.extend([
        "",
        f"Total: {total_exps} experiments across {len(families)} families ({exhausted} exhausted)",
    ])

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Show experiment families")
    parser.add_argument("--log", default="experiments/log.jsonl")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    eval_cfg = config.get("evaluation", {})
    metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(args.log)
    families = group_by_family(experiments)

    family_data = {name: family_stats(exps, metric, lower_is_better) for name, exps in families.items()}

    print(format_families(family_data, metric))


if __name__ == "__main__":
    main()
