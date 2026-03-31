#!/usr/bin/env python3
"""Deployment model card generation for exported models.

Produces a structured model card with metrics, seed study results,
export format, equivalence check, latency benchmarks, and dependencies.
"""

from __future__ import annotations

import yaml
from datetime import datetime, timezone
from pathlib import Path

from scripts.turing_io import load_config, load_seed_study


def generate_export_card(
    experiment: dict,
    export_result: dict,
    equivalence: dict | None = None,
    latency: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Generate a deployment model card for an exported model.

    Args:
        experiment: Original experiment dict from log.jsonl.
        export_result: Result from export_formats.export_model().
        equivalence: Result from equivalence_checker.compare_outputs().
        latency: Latency comparison from latency_benchmark.compare_latency().
        config: Project config dict.

    Returns:
        Model card dict.
    """
    exp_id = experiment.get("experiment_id", "unknown")
    metrics = experiment.get("metrics", {})
    exp_config = experiment.get("config", {})
    model_type = exp_config.get("model_type", config.get("model", {}).get("type", "unknown") if config else "unknown")

    eval_cfg = config.get("evaluation", {}) if config else {}
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    task_desc = config.get("task_description", eval_cfg.get("primary_metric", "N/A")) if config else "N/A"

    card = {
        "name": f"{exp_id}-{model_type}",
        "experiment_id": exp_id,
        "task": task_desc,
        "model_type": model_type,
        "primary_metric": primary_metric,
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "export_format": export_result.get("format", "unknown"),
        "export_path": export_result.get("path"),
        "size_mb": export_result.get("size_mb", 0),
        "dependencies": export_result.get("dependencies", []),
        "training_date": experiment.get("timestamp", "unknown"),
        "export_date": datetime.now(timezone.utc).isoformat(),
    }

    # Seed study (if available)
    seed_study = load_seed_study(exp_id)
    if seed_study and "mean" in seed_study:
        card["seed_study"] = {
            "mean": seed_study["mean"],
            "std": seed_study.get("std", 0),
            "cv_percent": seed_study.get("cv_percent", 0),
            "seed_sensitive": seed_study.get("seed_sensitive", False),
            "seeds_tested": len(seed_study.get("seeds_run", [])),
        }

    # Equivalence check
    if equivalence:
        card["equivalence"] = {
            "verdict": equivalence.get("verdict", "unknown"),
            "max_delta": equivalence.get("max_delta", 0),
            "n_samples_tested": equivalence.get("n_samples", 0),
        }

    # Latency benchmark
    if latency and latency.get("verdict") != "error":
        card["inference_latency"] = {
            "exported_p50_ms": latency.get("exported_p50_ms"),
            "exported_p95_ms": latency.get("exported_p95_ms"),
            "original_p50_ms": latency.get("original_p50_ms"),
            "speedup": latency.get("speedup_ratio"),
        }

    # Environment
    env = experiment.get("environment")
    if env:
        card["training_environment"] = {
            "python_version": env.get("python_version"),
            "gpu": env.get("gpu_name") or env.get("gpu"),
        }

    return card


def save_export_card(card: dict, output_dir: str) -> Path:
    """Save export model card to YAML file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / "model_card.yaml"
    with open(filepath, "w") as f:
        yaml.dump(card, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_export_card(card: dict) -> str:
    """Format export model card as readable markdown."""
    lines = [
        f"# Export Model Card: {card.get('name', 'unknown')}",
        "",
        f"- **Experiment:** {card.get('experiment_id', '?')}",
        f"- **Task:** {card.get('task', 'N/A')}",
        f"- **Model type:** {card.get('model_type', '?')}",
        f"- **Export format:** {card.get('export_format', '?')}",
        f"- **Size:** {card.get('size_mb', 0):.2f} MB",
        f"- **Dependencies:** {', '.join(card.get('dependencies', []))}",
        "",
        "## Metrics",
        "",
    ]

    for metric, value in card.get("metrics", {}).items():
        if isinstance(value, float):
            lines.append(f"- **{metric}:** {value:.4f}")
        else:
            lines.append(f"- **{metric}:** {value}")

    # Seed study
    seed = card.get("seed_study")
    if seed:
        status = "SEED-SENSITIVE" if seed.get("seed_sensitive") else "STABLE"
        lines.extend([
            "",
            "## Seed Study",
            "",
            f"- **Status:** {status}",
            f"- **Mean ± Std:** {seed['mean']:.4f} ± {seed.get('std', 0):.4f}",
            f"- **CV:** {seed.get('cv_percent', 0):.2f}%",
            f"- **Seeds tested:** {seed.get('seeds_tested', 0)}",
        ])

    # Equivalence
    eq = card.get("equivalence")
    if eq:
        verdict_markers = {
            "equivalent": "PASS (exact)",
            "approximately_equivalent": "PASS (approx)",
            "divergent": "FAIL",
        }
        marker = verdict_markers.get(eq["verdict"], eq["verdict"])
        lines.extend([
            "",
            "## Equivalence",
            "",
            f"- **Verdict:** {marker}",
            f"- **Max delta:** {eq.get('max_delta', 0):.2e}",
            f"- **Samples tested:** {eq.get('n_samples_tested', 0)}",
        ])

    # Latency
    lat = card.get("inference_latency")
    if lat:
        lines.extend([
            "",
            "## Inference Latency",
            "",
            f"- **Exported p50:** {lat.get('exported_p50_ms', 0):.2f} ms",
            f"- **Exported p95:** {lat.get('exported_p95_ms', 0):.2f} ms",
        ])
        if lat.get("original_p50_ms"):
            lines.append(f"- **Original p50:** {lat['original_p50_ms']:.2f} ms")
        if lat.get("speedup"):
            lines.append(f"- **Speedup:** {lat['speedup']:.1f}x")

    lines.extend([
        "",
        f"*Exported: {card.get('export_date', 'unknown')}*",
    ])

    return "\n".join(lines)
