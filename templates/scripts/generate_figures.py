#!/usr/bin/env python3
"""Presentation figure generation for the autoresearch pipeline.

Generates structured figure specifications (data + layout config)
for research presentations and papers. Produces JSON figure specs
rather than rendered images, since matplotlib may not be available
in all environments.

Supported figure types:
- training: metric trajectory over experiments
- comparison: model family comparison bar chart data
- ablation: ablation table with delta values
- pareto: accuracy vs latency/size scatter with Pareto frontier
- sensitivity: hyperparameter sensitivity heatmap data

Usage:
    python scripts/generate_figures.py training
    python scripts/generate_figures.py comparison --style dark
    python scripts/generate_figures.py ablation --format json
    python scripts/generate_figures.py pareto
    python scripts/generate_figures.py sensitivity
    python scripts/generate_figures.py --all --style poster
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
DEFAULT_OUTPUT_DIR = "paper/figures"
VALID_FIGURE_TYPES = ["training", "comparison", "ablation", "pareto", "sensitivity"]

STYLE_PRESETS = {
    "light": {
        "background": "#ffffff",
        "text_color": "#1e293b",
        "grid_color": "#e2e8f0",
        "palette": ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2"],
        "font_size": 12,
        "title_size": 16,
        "line_width": 2,
        "marker_size": 6,
    },
    "dark": {
        "background": "#0f172a",
        "text_color": "#e2e8f0",
        "grid_color": "#334155",
        "palette": ["#60a5fa", "#4ade80", "#f87171", "#fbbf24", "#a78bfa", "#22d3ee"],
        "font_size": 12,
        "title_size": 16,
        "line_width": 2,
        "marker_size": 6,
    },
    "poster": {
        "background": "#ffffff",
        "text_color": "#0f172a",
        "grid_color": "#cbd5e1",
        "palette": ["#1d4ed8", "#15803d", "#b91c1c", "#b45309", "#6d28d9", "#0e7490"],
        "font_size": 18,
        "title_size": 28,
        "line_width": 3,
        "marker_size": 10,
    },
}


# --- Figure Generators ---


def generate_training_figure(
    experiments: list[dict],
    config: dict,
    style: dict,
) -> dict:
    """Generate metric trajectory figure specification."""
    metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")
    lower_is_better = config.get("evaluation", {}).get("lower_is_better", False)

    data_points = []
    best_so_far = None
    best_envelope = []

    for exp in experiments:
        val = exp.get("metrics", {}).get(metric_name)
        if val is None or not isinstance(val, (int, float)):
            continue

        if best_so_far is None:
            best_so_far = val
        elif lower_is_better and val < best_so_far:
            best_so_far = val
        elif not lower_is_better and val > best_so_far:
            best_so_far = val

        data_points.append({
            "x": len(data_points),
            "experiment_id": exp.get("experiment_id", "?"),
            "value": round(val, 6),
            "status": exp.get("status", "unknown"),
        })
        best_envelope.append(round(best_so_far, 6))

    return {
        "type": "training",
        "title": f"{metric_name.replace('_', ' ').title()} Trajectory",
        "x_label": "Experiment Index",
        "y_label": metric_name,
        "style": style,
        "data": {
            "points": data_points,
            "best_envelope": best_envelope,
        },
        "annotations": {
            "total_experiments": len(data_points),
            "final_best": best_so_far,
            "lower_is_better": lower_is_better,
        },
    }


def generate_comparison_figure(
    experiments: list[dict],
    config: dict,
    style: dict,
) -> dict:
    """Generate model family comparison bar chart data."""
    metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")
    lower_is_better = config.get("evaluation", {}).get("lower_is_better", False)

    families: dict[str, list[float]] = {}
    for exp in experiments:
        family = exp.get("family") or exp.get("config", {}).get("model_type", "unknown")
        val = exp.get("metrics", {}).get(metric_name)
        if val is not None and isinstance(val, (int, float)):
            families.setdefault(family, []).append(val)

    bars = []
    for family, values in sorted(families.items()):
        n = len(values)
        mean = sum(values) / n
        sorted_vals = sorted(values)
        median = sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        best = min(values) if lower_is_better else max(values)
        variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0

        bars.append({
            "family": family,
            "mean": round(mean, 6),
            "median": round(median, 6),
            "best": round(best, 6),
            "std": round(variance ** 0.5, 6),
            "n_experiments": n,
        })

    # Sort by best performance
    bars.sort(key=lambda b: b["best"], reverse=not lower_is_better)

    return {
        "type": "comparison",
        "title": f"Model Family Comparison ({metric_name})",
        "x_label": "Model Family",
        "y_label": metric_name,
        "style": style,
        "data": {"bars": bars},
        "annotations": {
            "n_families": len(bars),
            "metric": metric_name,
            "lower_is_better": lower_is_better,
        },
    }


def generate_ablation_figure(
    experiments: list[dict],
    config: dict,
    style: dict,
) -> dict:
    """Generate ablation table with delta values.

    Identifies experiments that are ablation variants (share a base
    experiment or have ablation tags) and computes performance deltas.
    """
    metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")
    lower_is_better = config.get("evaluation", {}).get("lower_is_better", False)

    # Load ablation studies if available
    ablation_dir = Path("experiments/ablations")
    ablation_rows = []

    if ablation_dir.exists():
        for f in sorted(ablation_dir.glob("*-ablation.yaml")):
            try:
                with open(f) as fh:
                    study = yaml.safe_load(fh)
                if not study or not isinstance(study, dict):
                    continue
                base_val = study.get("baseline_metric")
                for variant in study.get("variants", []):
                    var_val = variant.get("metric_value")
                    if base_val is not None and var_val is not None:
                        delta = var_val - base_val
                        ablation_rows.append({
                            "experiment_id": study.get("experiment_id", "?"),
                            "component": variant.get("removed_component", "?"),
                            "baseline": round(base_val, 6),
                            "ablated": round(var_val, 6),
                            "delta": round(delta, 6),
                            "impact": "positive" if (delta > 0) != lower_is_better else "negative",
                        })
            except (yaml.YAMLError, OSError):
                continue

    # Fallback: infer from experiment descriptions containing "ablation" or "without"
    if not ablation_rows:
        kept = [e for e in experiments if e.get("status") == "kept"]
        if kept:
            # Use the best kept experiment as baseline
            best_exp = min(kept, key=lambda e: e.get("metrics", {}).get(metric_name, float("inf"))) \
                if lower_is_better else \
                max(kept, key=lambda e: e.get("metrics", {}).get(metric_name, float("-inf")))
            base_val = best_exp.get("metrics", {}).get(metric_name)

            if base_val is not None:
                for exp in experiments:
                    desc = (exp.get("description") or "").lower()
                    if "ablat" in desc or "without" in desc or "remove" in desc:
                        val = exp.get("metrics", {}).get(metric_name)
                        if val is not None:
                            delta = val - base_val
                            ablation_rows.append({
                                "experiment_id": exp.get("experiment_id", "?"),
                                "component": exp.get("description", "?")[:60],
                                "baseline": round(base_val, 6),
                                "ablated": round(val, 6),
                                "delta": round(delta, 6),
                                "impact": "positive" if (delta > 0) != lower_is_better else "negative",
                            })

    # Sort by absolute delta (biggest impact first)
    ablation_rows.sort(key=lambda r: abs(r["delta"]), reverse=True)

    return {
        "type": "ablation",
        "title": f"Ablation Study ({metric_name})",
        "columns": ["Component", "Baseline", "Ablated", "Delta", "Impact"],
        "style": style,
        "data": {"rows": ablation_rows},
        "annotations": {
            "n_ablations": len(ablation_rows),
            "metric": metric_name,
            "lower_is_better": lower_is_better,
        },
    }


def generate_pareto_figure(
    experiments: list[dict],
    config: dict,
    style: dict,
) -> dict:
    """Generate accuracy vs latency/size scatter with Pareto frontier."""
    metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")
    lower_is_better = config.get("evaluation", {}).get("lower_is_better", False)

    # Determine secondary axis (latency or model size)
    points = []
    secondary_label = "latency_seconds"

    for exp in experiments:
        metrics = exp.get("metrics", {})
        primary = metrics.get(metric_name)
        if primary is None or not isinstance(primary, (int, float)):
            continue

        # Try latency, then model_size, then training_time
        secondary = None
        for candidate in ["latency_seconds", "latency", "model_size", "training_time_seconds"]:
            secondary = metrics.get(candidate)
            if secondary is not None:
                secondary_label = candidate
                break

        if secondary is None:
            # Use training time from top-level if available
            secondary = exp.get("training_time")
            if secondary is not None:
                secondary_label = "training_time"

        if secondary is None:
            continue

        points.append({
            "experiment_id": exp.get("experiment_id", "?"),
            "primary": round(float(primary), 6),
            "secondary": round(float(secondary), 6),
            "family": exp.get("family") or exp.get("config", {}).get("model_type", "unknown"),
            "status": exp.get("status", "unknown"),
        })

    # Compute Pareto frontier
    frontier_ids = set()
    if points:
        # Sort by secondary ascending (lower cost is better)
        sorted_pts = sorted(points, key=lambda p: p["secondary"])
        best_primary = float("-inf") if not lower_is_better else float("inf")
        for pt in sorted_pts:
            if lower_is_better:
                if pt["primary"] <= best_primary:
                    best_primary = pt["primary"]
                    frontier_ids.add(pt["experiment_id"])
            else:
                if pt["primary"] >= best_primary:
                    best_primary = pt["primary"]
                    frontier_ids.add(pt["experiment_id"])

    for pt in points:
        pt["on_frontier"] = pt["experiment_id"] in frontier_ids

    return {
        "type": "pareto",
        "title": f"Pareto Frontier: {metric_name} vs {secondary_label}",
        "x_label": secondary_label.replace("_", " ").title(),
        "y_label": metric_name,
        "style": style,
        "data": {
            "points": points,
            "frontier_ids": list(frontier_ids),
        },
        "annotations": {
            "n_points": len(points),
            "n_frontier": len(frontier_ids),
            "metric": metric_name,
            "cost_axis": secondary_label,
        },
    }


def generate_sensitivity_figure(
    experiments: list[dict],
    config: dict,
    style: dict,
) -> dict:
    """Generate hyperparameter sensitivity heatmap data.

    Extracts hyperparameter values from experiment configs and
    correlates them with the primary metric to build a sensitivity matrix.
    """
    metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")

    # Collect param-value-metric triples
    param_values: dict[str, list[tuple[float, float]]] = {}

    for exp in experiments:
        val = exp.get("metrics", {}).get(metric_name)
        if val is None or not isinstance(val, (int, float)):
            continue

        exp_config = exp.get("config", {})
        for param, pval in exp_config.items():
            if isinstance(pval, (int, float)) and param != metric_name:
                param_values.setdefault(param, []).append((float(pval), float(val)))

    # Compute sensitivity score per parameter (range of metric across param values)
    heatmap_data = []
    for param, pairs in param_values.items():
        if len(pairs) < 2:
            continue

        metric_vals = [p[1] for p in pairs]
        metric_range = max(metric_vals) - min(metric_vals)
        metric_mean = sum(metric_vals) / len(metric_vals)
        metric_std = (sum((v - metric_mean) ** 2 for v in metric_vals) / len(metric_vals)) ** 0.5

        # Bin parameter values for heatmap cells
        param_vals = sorted(set(p[0] for p in pairs))
        cells = []
        for pv in param_vals:
            associated = [p[1] for p in pairs if p[0] == pv]
            cells.append({
                "param_value": pv,
                "metric_mean": round(sum(associated) / len(associated), 6),
                "n_experiments": len(associated),
            })

        heatmap_data.append({
            "parameter": param,
            "sensitivity_score": round(metric_range, 6),
            "metric_std": round(metric_std, 6),
            "n_unique_values": len(param_vals),
            "n_experiments": len(pairs),
            "cells": cells,
        })

    # Sort by sensitivity score descending
    heatmap_data.sort(key=lambda h: h["sensitivity_score"], reverse=True)

    return {
        "type": "sensitivity",
        "title": f"Hyperparameter Sensitivity ({metric_name})",
        "x_label": "Parameter Value",
        "y_label": "Parameter",
        "style": style,
        "data": {"parameters": heatmap_data},
        "annotations": {
            "n_parameters": len(heatmap_data),
            "most_sensitive": heatmap_data[0]["parameter"] if heatmap_data else None,
            "metric": metric_name,
        },
    }


# --- Report ---


FIGURE_GENERATORS = {
    "training": generate_training_figure,
    "comparison": generate_comparison_figure,
    "ablation": generate_ablation_figure,
    "pareto": generate_pareto_figure,
    "sensitivity": generate_sensitivity_figure,
}


def format_figures_report(figures: list[dict]) -> str:
    """Format figure specifications as a readable summary."""
    lines = [
        f"# Figure Specifications ({len(figures)} figures)",
        "",
    ]

    for fig in figures:
        ftype = fig.get("type", "?")
        title = fig.get("title", "Untitled")
        annotations = fig.get("annotations", {})

        lines.append(f"## {title}")
        lines.append(f"Type: {ftype}")

        for key, value in annotations.items():
            lines.append(f"  {key}: {value}")

        data = fig.get("data", {})
        if "points" in data:
            lines.append(f"  Data points: {len(data['points'])}")
        if "bars" in data:
            lines.append(f"  Bars: {len(data['bars'])}")
        if "rows" in data:
            lines.append(f"  Rows: {len(data['rows'])}")
        if "parameters" in data:
            lines.append(f"  Parameters: {len(data['parameters'])}")

        lines.append("")

    return "\n".join(lines)


def save_figures_report(
    figures: list[dict],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Save each figure specification as a JSON file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    for fig in figures:
        ftype = fig.get("type", "unknown")
        filename = f"{ftype}-{timestamp}.json"
        filepath = out / filename
        with open(filepath, "w") as f:
            json.dump(fig, f, indent=2, default=str)
        saved.append(filepath)

    return saved


# --- Orchestration ---


def run_generate_figures(
    figure_types: list[str],
    style_name: str = "light",
    log_path: str = DEFAULT_LOG_PATH,
    config_path: str = "config.yaml",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save: bool = True,
) -> dict:
    """Generate figure specifications for requested types."""
    timestamp = datetime.now(timezone.utc).isoformat()
    config = load_config(config_path)
    experiments = load_experiments(log_path)
    style = STYLE_PRESETS.get(style_name, STYLE_PRESETS["light"])

    if not experiments:
        return {"timestamp": timestamp, "error": "No experiments found in log"}

    figures = []
    errors = []
    for ftype in figure_types:
        generator = FIGURE_GENERATORS.get(ftype)
        if not generator:
            errors.append(f"Unknown figure type: {ftype}")
            continue
        try:
            fig = generator(experiments, config, style)
            fig["generated_at"] = timestamp
            fig["style_name"] = style_name
            figures.append(fig)
        except Exception as e:
            errors.append(f"{ftype}: {e}")

    saved_paths = []
    if save and figures:
        saved_paths = save_figures_report(figures, output_dir)

    return {
        "timestamp": timestamp,
        "figures": figures,
        "saved_to": [str(p) for p in saved_paths],
        "errors": errors if errors else None,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate presentation figure specifications from experiment data",
    )
    parser.add_argument("figure_types", nargs="*", default=[],
                        help="Figure types to generate (training, comparison, ablation, pareto, sensitivity)")
    parser.add_argument("--all", action="store_true",
                        help="Generate all figure types")
    parser.add_argument("--style", choices=["light", "dark", "poster"], default="light",
                        help="Visual style preset")
    parser.add_argument("--format", dest="fmt", choices=["json"], default="json",
                        help="Output format")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for figure specs")
    parser.add_argument("--no-save", action="store_true",
                        help="Print to stdout instead of saving files")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.all:
        figure_types = VALID_FIGURE_TYPES
    elif args.figure_types:
        figure_types = args.figure_types
    else:
        print("ERROR: Specify figure types or use --all", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Validate types
    for ft in figure_types:
        if ft not in VALID_FIGURE_TYPES:
            print(f"ERROR: Unknown figure type '{ft}'. Valid: {VALID_FIGURE_TYPES}",
                  file=sys.stderr)
            sys.exit(1)

    report = run_generate_figures(
        figure_types=figure_types,
        style_name=args.style,
        log_path=args.log,
        config_path=args.config,
        output_dir=args.output_dir,
        save=not args.no_save,
    )

    if args.json or args.fmt == "json":
        print(json.dumps(report, indent=2, default=str))
    else:
        if "error" in report:
            print(f"ERROR: {report['error']}", file=sys.stderr)
            sys.exit(1)
        figures = report.get("figures", [])
        print(format_figures_report(figures))
        saved = report.get("saved_to", [])
        if saved:
            print(f"Saved {len(saved)} figure(s) to {args.output_dir}/")
            for p in saved:
                print(f"  {p}")


if __name__ == "__main__":
    main()
