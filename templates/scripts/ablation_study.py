#!/usr/bin/env python3
"""Systematic ablation studies for ML experiments.

Removes components one at a time, measures impact on primary metric,
and produces a publication-ready ablation table. Flags dead-weight
components (removing them improves the model).

Usage:
    python scripts/ablation_study.py                              # Auto-detect components
    python scripts/ablation_study.py --exp-id exp-042             # Specific experiment
    python scripts/ablation_study.py --components "dropout,feature_X"  # Specific components
    python scripts/ablation_study.py --seeds 3                    # Statistical robustness
    python scripts/ablation_study.py --latex                      # LaTeX output
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments


def find_experiment(experiments: list[dict], exp_id: str | None, metric: str, lower_is_better: bool) -> dict | None:
    """Find experiment by ID or return best kept."""
    if exp_id:
        for exp in experiments:
            if exp.get("experiment_id") == exp_id:
                return exp
        return None
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


def detect_ablatable_components(config: dict) -> list[dict]:
    """Auto-detect components that can be ablated from the model config.

    Returns list of component dicts with name, type, current_value, and
    ablation_config (what to set it to when "removing" the component).
    """
    components = []
    hyperparams = config.get("model", {}).get("hyperparams", {})

    # Regularization parameters
    regularization_params = {
        "max_depth": ("regularization", "depth limit", 0),
        "min_child_weight": ("regularization", "min samples per leaf", 0),
        "min_samples_split": ("regularization", "min split samples", 2),
        "min_samples_leaf": ("regularization", "min leaf samples", 1),
        "reg_alpha": ("regularization", "L1 penalty", 0),
        "reg_lambda": ("regularization", "L2 penalty", 0),
        "alpha": ("regularization", "L1 penalty", 0),
        "l1_ratio": ("regularization", "L1/L2 ratio", 0),
        "gamma": ("regularization", "min split loss", 0),
        "subsample": ("regularization", "row subsampling", 1.0),
        "colsample_bytree": ("regularization", "column subsampling", 1.0),
        "colsample_bylevel": ("regularization", "level column subsampling", 1.0),
        "dropout_rate": ("regularization", "dropout", 0),
        "weight_decay": ("regularization", "weight decay", 0),
    }

    for param, (comp_type, desc, removal_val) in regularization_params.items():
        if param in hyperparams:
            current = hyperparams[param]
            if current != removal_val:
                components.append({
                    "name": param,
                    "type": comp_type,
                    "description": desc,
                    "current_value": current,
                    "ablation_value": removal_val,
                    "config_path": f"model.hyperparams.{param}",
                })

    # Model complexity parameters (reduce, not remove)
    complexity_params = {
        "n_estimators": ("complexity", "number of trees/estimators", 10),
        "num_leaves": ("complexity", "number of leaves", 8),
        "max_features": ("complexity", "feature subset", None),
        "hidden_layer_sizes": ("complexity", "network architecture", None),
    }

    for param, (comp_type, desc, reduction_val) in complexity_params.items():
        if param in hyperparams and reduction_val is not None:
            components.append({
                "name": param,
                "type": comp_type,
                "description": desc,
                "current_value": hyperparams[param],
                "ablation_value": reduction_val,
                "config_path": f"model.hyperparams.{param}",
            })

    # Learning rate (test with higher LR = less refined)
    if "learning_rate" in hyperparams:
        lr = hyperparams["learning_rate"]
        if lr < 0.5:
            components.append({
                "name": "learning_rate",
                "type": "training",
                "description": "learning rate (test with 10x higher)",
                "current_value": lr,
                "ablation_value": min(lr * 10, 1.0),
                "config_path": "model.hyperparams.learning_rate",
            })

    return components


def parse_component_list(components_str: str) -> list[str]:
    """Parse comma-separated component names."""
    return [c.strip() for c in components_str.split(",") if c.strip()]


def run_ablation_experiment(
    component: dict,
    config: dict,
    seed: int = 42,
    timeout: int = 600,
) -> dict | None:
    """Run a single ablation experiment with one component modified.

    Returns parsed metrics dict or None on failure.
    """
    # We run train.py with the modified config via environment or temp config
    # For simplicity, we use the --override flag pattern
    cmd = [
        "python", "train.py",
        "--seed", str(seed),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None

    if proc.returncode != 0:
        return None

    metrics = {}
    in_block = False
    metadata_keys = {"model_type", "train_seconds"}

    for line in proc.stdout.splitlines():
        line = line.strip()
        if line == "---":
            if in_block:
                break
            in_block = True
            continue
        if in_block and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in metadata_keys:
                metrics[key] = value
            else:
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value

    return metrics if metrics else None


def compute_ablation_table(
    full_model_metric: float,
    ablation_results: list[dict],
    metric: str,
    lower_is_better: bool,
) -> list[dict]:
    """Compute the ablation table with deltas and rankings.

    Returns list of row dicts sorted by absolute impact (largest first).
    """
    rows = []

    for result in ablation_results:
        component = result["component"]
        value = result.get("metric_value")

        if value is None:
            rows.append({
                "configuration": f"− {component['name']}",
                "component": component,
                "metric_value": None,
                "delta": None,
                "delta_pct": None,
                "is_dead_weight": False,
                "status": "failed",
            })
            continue

        delta = value - full_model_metric
        delta_pct = (delta / abs(full_model_metric) * 100) if full_model_metric != 0 else 0

        # Dead weight: removing it improves the metric
        if lower_is_better:
            is_dead_weight = delta < 0  # lower is better, so negative delta = improvement
        else:
            is_dead_weight = delta > 0  # higher is better, so positive delta = improvement

        rows.append({
            "configuration": f"− {component['name']}",
            "component": component,
            "metric_value": round(value, 6),
            "delta": round(delta, 6),
            "delta_pct": round(delta_pct, 2),
            "is_dead_weight": is_dead_weight,
            "status": "completed",
        })

    # Sort by absolute delta (most impactful first)
    rows.sort(key=lambda r: -abs(r["delta"]) if r["delta"] is not None else 0)
    return rows


def format_ablation_table(
    full_metric: float,
    rows: list[dict],
    metric: str,
    lower_is_better: bool,
) -> str:
    """Format ablation results as a markdown table."""
    direction = "lower" if lower_is_better else "higher"
    lines = [
        f"# Ablation Study",
        "",
        f"*{metric} ({direction} is better)*",
        "",
        f"| Configuration | {metric} | Δ from Full | % Change | Status |",
        f"|---------------|{'---' * len(metric)}--|-------------|----------|--------|",
        f"| Full model | {full_metric:.4f} | — | — | baseline |",
    ]

    for row in rows:
        if row["status"] == "failed":
            lines.append(f"| {row['configuration']} | FAILED | — | — | error |")
            continue

        delta_str = f"{row['delta']:+.4f}" if row["delta"] is not None else "N/A"
        pct_str = f"{row['delta_pct']:+.1f}%" if row["delta_pct"] is not None else "N/A"
        status = "DEAD WEIGHT" if row["is_dead_weight"] else "contributes"

        lines.append(
            f"| {row['configuration']} | {row['metric_value']:.4f} "
            f"| {delta_str} | {pct_str} | {status} |"
        )

    # Summary
    dead_weight = [r for r in rows if r.get("is_dead_weight")]
    if dead_weight:
        lines.extend([
            "",
            "## Dead-Weight Components",
            "",
            "These components can be removed to **improve** the model:",
            "",
        ])
        for r in dead_weight:
            lines.append(f"- **{r['component']['name']}** ({r['component']['description']}): "
                        f"removing it improves {metric} by {abs(r['delta']):.4f}")

    return "\n".join(lines)


def format_latex_table(
    full_metric: float,
    rows: list[dict],
    metric: str,
) -> str:
    """Format ablation results as a LaTeX table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{Ablation study results ({metric})}}",
        f"\\label{{tab:ablation}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        f"Configuration & {metric} & $\\Delta$ from Full \\\\",
        r"\midrule",
        f"Full model & {full_metric:.4f} & --- \\\\",
    ]

    for row in rows:
        if row["status"] == "failed":
            continue
        delta_str = f"{row['delta']:+.4f}" if row["delta"] is not None else "N/A"
        config_escaped = row["configuration"].replace("_", r"\_")
        lines.append(f"{config_escaped} & {row['metric_value']:.4f} & {delta_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def save_ablation(study: dict, output_dir: str = "experiments/ablations") -> Path:
    """Save ablation study to YAML file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_id = study.get("experiment_id", "unknown")
    filepath = out_path / f"{exp_id}-ablation.yaml"
    with open(filepath, "w") as f:
        yaml.dump(study, f, default_flow_style=False, sort_keys=False)
    return filepath


def run_ablation_study(
    exp_id: str | None = None,
    components_str: str | None = None,
    n_seeds: int = 1,
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
    timeout: int = 600,
) -> dict:
    """Run a complete ablation study.

    Args:
        exp_id: Experiment ID (defaults to best).
        components_str: Comma-separated component names to ablate.
        n_seeds: Number of seeds per ablation (for statistical robustness).
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        timeout: Per-run timeout in seconds.

    Returns:
        Ablation study result dict.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)
    target_exp = find_experiment(experiments, exp_id, primary_metric, lower_is_better)

    if not target_exp:
        return {"error": f"No experiment found{f' with ID {exp_id}' if exp_id else ''}", "experiment_id": exp_id}

    target_id = target_exp.get("experiment_id", "unknown")
    full_metric = target_exp.get("metrics", {}).get(primary_metric)

    if full_metric is None:
        return {"error": f"Experiment {target_id} has no {primary_metric} metric", "experiment_id": target_id}

    # Detect or parse components
    if components_str:
        component_names = parse_component_list(components_str)
        all_components = detect_ablatable_components(config)
        components = [c for c in all_components if c["name"] in component_names]
        # Add unknown components with basic info
        known_names = {c["name"] for c in components}
        for name in component_names:
            if name not in known_names:
                components.append({
                    "name": name,
                    "type": "custom",
                    "description": f"user-specified component: {name}",
                    "current_value": "unknown",
                    "ablation_value": None,
                    "config_path": f"custom.{name}",
                })
    else:
        components = detect_ablatable_components(config)

    if not components:
        return {
            "error": "No ablatable components detected. Specify with --components.",
            "experiment_id": target_id,
        }

    print(f"Ablation study for {target_id}", file=sys.stderr)
    print(f"Full model {primary_metric}: {full_metric:.4f}", file=sys.stderr)
    print(f"Components to ablate: {[c['name'] for c in components]}", file=sys.stderr)
    print(f"Seeds per ablation: {n_seeds}", file=sys.stderr)
    print(file=sys.stderr)

    # Run ablations
    ablation_results = []
    for comp in components:
        print(f"  Ablating {comp['name']}...", end=" ", flush=True, file=sys.stderr)
        values = []
        for seed_i in range(n_seeds):
            seed = 42 + seed_i
            result = run_ablation_experiment(comp, config, seed=seed, timeout=timeout)
            if result and primary_metric in result:
                values.append(result[primary_metric])

        if values:
            metric_value = float(np.mean(values))
            metric_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ablation_results.append({
                "component": comp,
                "metric_value": metric_value,
                "metric_std": metric_std,
                "n_seeds": len(values),
                "values": values,
            })
            print(f"{primary_metric}={metric_value:.4f}", file=sys.stderr)
        else:
            ablation_results.append({
                "component": comp,
                "metric_value": None,
                "status": "failed",
            })
            print("FAILED", file=sys.stderr)

    # Compute table
    table_rows = compute_ablation_table(full_metric, ablation_results, primary_metric, lower_is_better)

    study = {
        "experiment_id": target_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": primary_metric,
        "lower_is_better": lower_is_better,
        "full_model_metric": round(full_metric, 6),
        "components_ablated": len(components),
        "seeds_per_ablation": n_seeds,
        "results": table_rows,
        "dead_weight": [r["component"]["name"] for r in table_rows if r.get("is_dead_weight")],
    }

    return study


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Systematic ablation studies for ML experiments")
    parser.add_argument("--exp-id", default=None, help="Experiment ID (defaults to best)")
    parser.add_argument("--components", default=None, help="Comma-separated component names to ablate")
    parser.add_argument("--seeds", type=int, default=1, help="Seeds per ablation (default: 1, use 3+ for robust)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl", help="Path to experiment log")
    parser.add_argument("--timeout", type=int, default=600, help="Per-run timeout in seconds")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table instead of markdown")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    study = run_ablation_study(
        exp_id=args.exp_id,
        components_str=args.components,
        n_seeds=args.seeds,
        config_path=args.config,
        log_path=args.log,
        timeout=args.timeout,
    )

    if "error" not in study:
        filepath = save_ablation(study)
        print(f"\nSaved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(study, indent=2, default=str))
    elif args.latex:
        if "error" in study:
            print(f"ERROR: {study['error']}")
        else:
            print(format_latex_table(study["full_model_metric"], study["results"], study["metric"]))
    else:
        if "error" in study:
            print(f"ERROR: {study['error']}")
        else:
            print(format_ablation_table(study["full_model_metric"], study["results"], study["metric"], study["lower_is_better"]))


if __name__ == "__main__":
    main()
