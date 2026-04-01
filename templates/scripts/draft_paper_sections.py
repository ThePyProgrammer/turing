#!/usr/bin/env python3
"""Paper section drafting from experiment logs.

Drafts the mechanical sections of an ML paper directly from experiment
data: experimental setup, results tables, ablation tables, and
hyperparameter appendices. Eliminates transcription errors.

Usage:
    python scripts/draft_paper_sections.py                                    # All sections
    python scripts/draft_paper_sections.py --sections setup,results           # Specific sections
    python scripts/draft_paper_sections.py --format latex                     # LaTeX output
    python scripts/draft_paper_sections.py --format markdown                  # Markdown output
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments


VALID_SECTIONS = ["setup", "results", "ablation", "hyperparameters"]
DEFAULT_FORMAT = "latex"


def load_seed_studies_for_paper(seed_dir: str = "experiments/seed_studies") -> dict[str, dict]:
    """Load all seed studies indexed by experiment ID."""
    path = Path(seed_dir)
    studies = {}
    if not path.exists():
        return studies
    for f in path.glob("*-seeds.yaml"):
        try:
            with open(f) as fh:
                study = yaml.safe_load(fh)
                if study and isinstance(study, dict) and "experiment_id" in study:
                    studies[study["experiment_id"]] = study
        except (yaml.YAMLError, OSError):
            continue
    return studies


def load_ablation_studies(ablation_dir: str = "experiments/ablations") -> dict[str, dict]:
    """Load all ablation studies indexed by experiment ID."""
    path = Path(ablation_dir)
    studies = {}
    if not path.exists():
        return studies
    for f in path.glob("*-ablation.yaml"):
        try:
            with open(f) as fh:
                study = yaml.safe_load(fh)
                if study and isinstance(study, dict) and "experiment_id" in study:
                    studies[study["experiment_id"]] = study
        except (yaml.YAMLError, OSError):
            continue
    return studies


def get_top_experiments(
    experiments: list[dict],
    metric: str,
    lower_is_better: bool,
    top_k: int = 10,
) -> list[dict]:
    """Get top-K kept experiments by primary metric."""
    kept = [e for e in experiments if e.get("status") == "kept" and e.get("metrics", {}).get(metric) is not None]
    kept.sort(key=lambda e: e["metrics"][metric], reverse=not lower_is_better)
    return kept[:top_k]


def group_by_model_type(experiments: list[dict]) -> dict[str, list[dict]]:
    """Group experiments by model type, keeping best per type."""
    groups: dict[str, list[dict]] = {}
    for exp in experiments:
        mt = exp.get("config", {}).get("model_type", "unknown")
        groups.setdefault(mt, []).append(exp)
    return groups


def draft_setup_section(
    config: dict,
    experiments: list[dict],
    seed_studies: dict[str, dict],
    output_format: str = "latex",
) -> str:
    """Draft the experimental setup section."""
    eval_cfg = config.get("evaluation", {})
    data_cfg = config.get("data", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    metrics = eval_cfg.get("metrics", [primary_metric])
    lower_is_better = eval_cfg.get("lower_is_better", False)

    task_desc = config.get("task_description", "the classification task")
    data_source = data_cfg.get("source", "the provided dataset")
    split_ratios = data_cfg.get("split_ratios", {})
    random_state = data_cfg.get("random_state", 42)

    # Determine seed study info
    n_seeds = 0
    for study in seed_studies.values():
        n_seeds = max(n_seeds, len(study.get("seeds_run", [])))

    # Build prose
    split_text = ""
    if split_ratios:
        parts = [f"{int(v*100)}\\%" if output_format == "latex" else f"{int(v*100)}%" for v in split_ratios.values()]
        split_names = list(split_ratios.keys())
        split_text = "/".join(parts) + " " + "/".join(split_names) + " split"

    direction = "lower" if lower_is_better else "higher"
    metric_list = ", ".join(metrics)

    if output_format == "latex":
        lines = [
            r"\subsection{Experimental Setup}",
            "",
            f"We evaluate on {data_source} using {metric_list} as evaluation metrics "
            f"({direction} is better for {primary_metric}).",
        ]
        if split_text:
            lines.append(f"Data is partitioned using a {split_text} with random state {random_state}.")
        if n_seeds > 0:
            lines.append(
                f"Results are reported as mean $\\pm$ standard deviation over {n_seeds} random seeds "
                f"to account for seed sensitivity."
            )
        lines.append(f"All experiments use {task_desc} as the target task.")
    else:
        lines = [
            "## Experimental Setup",
            "",
            f"We evaluate on {data_source} using {metric_list} as evaluation metrics "
            f"({direction} is better for {primary_metric}).",
        ]
        if split_text:
            lines.append(f"Data is partitioned using a {split_text} with random state {random_state}.")
        if n_seeds > 0:
            lines.append(
                f"Results are reported as mean +/- standard deviation over {n_seeds} random seeds "
                f"to account for seed sensitivity."
            )
        lines.append(f"All experiments use {task_desc} as the target task.")

    return "\n".join(lines)


def draft_results_table(
    experiments: list[dict],
    metrics: list[str],
    primary_metric: str,
    lower_is_better: bool,
    seed_studies: dict[str, dict],
    output_format: str = "latex",
    dataset_name: str = "the dataset",
) -> str:
    """Draft the results comparison table."""
    # Group by model type, take best per type
    groups = group_by_model_type(experiments)
    rows = []
    for mt, exps in groups.items():
        # Best experiment per model type
        best = exps[0]  # Already sorted by caller
        row = {"model_type": mt, "experiment_id": best.get("experiment_id", "?")}
        for m in metrics:
            val = best.get("metrics", {}).get(m)
            seed = seed_studies.get(best.get("experiment_id", ""))
            if seed and seed.get("metric") == m:
                row[m] = {"value": val, "mean": seed.get("mean"), "std": seed.get("std")}
            else:
                row[m] = {"value": val}
        rows.append(row)

    # Find best value per metric
    best_per_metric = {}
    for m in metrics:
        values = [(r["model_type"], r[m].get("mean") or r[m].get("value")) for r in rows if r[m].get("value") is not None]
        if values:
            if lower_is_better:
                best_per_metric[m] = min(values, key=lambda x: x[1] if x[1] is not None else float("inf"))[0]
            else:
                best_per_metric[m] = max(values, key=lambda x: x[1] if x[1] is not None else float("-inf"))[0]

    if output_format == "latex":
        return _format_results_latex(rows, metrics, best_per_metric, dataset_name)
    else:
        return _format_results_markdown(rows, metrics, best_per_metric, dataset_name)


def _format_results_latex(rows: list[dict], metrics: list[str], best_per: dict, dataset: str) -> str:
    """Format results as LaTeX table."""
    n_cols = len(metrics)
    col_spec = "l" + "c" * n_cols
    metric_headers = " & ".join(m.replace("_", r"\_") for m in metrics)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{Comparison of model architectures on {dataset}.}}",
        r"\label{tab:results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Model & {metric_headers} \\\\",
        r"\midrule",
    ]

    for row in rows:
        mt = row["model_type"].replace("_", r"\_")
        cells = [mt]
        for m in metrics:
            data = row[m]
            val = data.get("mean") or data.get("value")
            std = data.get("std")
            if val is None:
                cells.append("---")
            elif std:
                cell = f"{val:.3f} $\\pm$ {std:.3f}"
                if best_per.get(m) == row["model_type"]:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            else:
                cell = f"{val:.4f}"
                if best_per.get(m) == row["model_type"]:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _format_results_markdown(rows: list[dict], metrics: list[str], best_per: dict, dataset: str) -> str:
    """Format results as markdown table."""
    header = f"| Model |"
    sep = "|-------|"
    for m in metrics:
        header += f" {m} |"
        sep += f"{'---' * max(len(m) // 3, 1)}--|"

    lines = [
        f"## Results on {dataset}",
        "",
        header,
        sep,
    ]

    for row in rows:
        line = f"| {row['model_type']} |"
        for m in metrics:
            data = row[m]
            val = data.get("mean") or data.get("value")
            std = data.get("std")
            if val is None:
                line += " --- |"
            elif std:
                cell = f"{val:.3f} +/- {std:.3f}"
                if best_per.get(m) == row["model_type"]:
                    cell = f"**{cell}**"
                line += f" {cell} |"
            else:
                cell = f"{val:.4f}"
                if best_per.get(m) == row["model_type"]:
                    cell = f"**{cell}**"
                line += f" {cell} |"
        lines.append(line)

    return "\n".join(lines)


def draft_ablation_table(
    ablation_studies: dict[str, dict],
    output_format: str = "latex",
) -> str:
    """Draft ablation table from ablation study results."""
    if not ablation_studies:
        return "No ablation studies available. Run `/turing:ablate` first."

    # Use the most recent ablation study
    study = list(ablation_studies.values())[-1]
    metric = study.get("metric", "accuracy")
    full_metric = study.get("full_model_metric", 0)
    results = study.get("results", [])

    if not results:
        return "Ablation study has no results."

    if output_format == "latex":
        metric_escaped = metric.replace("_", r"\_")
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            f"\\caption{{Ablation study results ({metric_escaped}).}}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            f"Configuration & {metric_escaped} & $\\Delta$ from Full \\\\",
            r"\midrule",
            f"Full model & {full_metric:.4f} & --- \\\\",
        ]
        for r in results:
            if r.get("status") == "failed":
                continue
            config = r.get("configuration", "?").replace("_", r"\_")
            val = r.get("metric_value", 0)
            delta = r.get("delta", 0)
            delta_str = f"{delta:+.4f}" if delta is not None else "---"
            lines.append(f"{config} & {val:.4f} & {delta_str} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)
    else:
        lines = [
            f"## Ablation Study ({metric})",
            "",
            f"| Configuration | {metric} | Delta from Full |",
            f"|---------------|{'---' * max(len(metric) // 3, 1)}--|-----------------|",
            f"| Full model | {full_metric:.4f} | --- |",
        ]
        for r in results:
            if r.get("status") == "failed":
                continue
            config = r.get("configuration", "?")
            val = r.get("metric_value", 0)
            delta = r.get("delta", 0)
            delta_str = f"{delta:+.4f}" if delta is not None else "---"
            lines.append(f"| {config} | {val:.4f} | {delta_str} |")
        return "\n".join(lines)


def draft_hyperparameter_table(
    experiments: list[dict],
    output_format: str = "latex",
) -> str:
    """Draft hyperparameter appendix table."""
    groups = group_by_model_type(experiments)

    if output_format == "latex":
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Hyperparameters for reported models.}",
            r"\label{tab:hyperparams}",
            r"\begin{tabular}{llr}",
            r"\toprule",
            r"Model & Parameter & Value \\",
            r"\midrule",
        ]
        for mt, exps in groups.items():
            best = exps[0]
            hyperparams = best.get("config", {}).get("hyperparams", {})
            first = True
            for param, value in sorted(hyperparams.items()):
                model_col = mt.replace("_", r"\_") if first else ""
                param_escaped = param.replace("_", r"\_")
                lines.append(f"{model_col} & {param_escaped} & {value} \\\\")
                first = False
            lines.append(r"\midrule")
        if lines[-1] == r"\midrule":
            lines.pop()
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)
    else:
        lines = [
            "## Hyperparameters",
            "",
            "| Model | Parameter | Value |",
            "|-------|-----------|-------|",
        ]
        for mt, exps in groups.items():
            best = exps[0]
            hyperparams = best.get("config", {}).get("hyperparams", {})
            for param, value in sorted(hyperparams.items()):
                lines.append(f"| {mt} | {param} | {value} |")
        return "\n".join(lines)


def save_paper_sections(sections: dict[str, str], output_dir: str = "paper/sections") -> list[Path]:
    """Save each section to its own file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, content in sections.items():
        ext = ".tex" if r"\begin" in content else ".md"
        filepath = out_path / f"{name}{ext}"
        with open(filepath, "w") as f:
            f.write(content)
        saved.append(filepath)

    return saved


def draft_paper(
    sections_str: str | None = None,
    output_format: str = DEFAULT_FORMAT,
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
) -> dict:
    """Draft all requested paper sections.

    Args:
        sections_str: Comma-separated section names (setup,results,ablation,hyperparameters).
        output_format: "latex" or "markdown".
        config_path: Path to config.yaml.
        log_path: Path to experiment log.

    Returns:
        Dict with section name -> content mappings.
    """
    if sections_str:
        sections = [s.strip() for s in sections_str.split(",")]
    else:
        sections = VALID_SECTIONS

    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    all_metrics = eval_cfg.get("metrics", [primary_metric])
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)
    top_exps = get_top_experiments(experiments, primary_metric, lower_is_better)

    seed_studies = load_seed_studies_for_paper()
    ablation_studies = load_ablation_studies()

    dataset_name = config.get("data", {}).get("source", "the dataset")

    result = {"format": output_format, "sections": {}}

    for section in sections:
        if section == "setup":
            result["sections"]["setup"] = draft_setup_section(
                config, experiments, seed_studies, output_format,
            )
        elif section == "results":
            result["sections"]["results"] = draft_results_table(
                top_exps, all_metrics, primary_metric, lower_is_better,
                seed_studies, output_format, dataset_name,
            )
        elif section == "ablation":
            result["sections"]["ablation"] = draft_ablation_table(
                ablation_studies, output_format,
            )
        elif section == "hyperparameters":
            result["sections"]["hyperparameters"] = draft_hyperparameter_table(
                top_exps, output_format,
            )

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Draft paper sections from experiment logs")
    parser.add_argument("--sections", default=None, help="Comma-separated sections: setup,results,ablation,hyperparameters")
    parser.add_argument("--format", default=DEFAULT_FORMAT, dest="output_format", choices=["latex", "markdown"])
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl", help="Path to experiment log")
    parser.add_argument("--output", default="paper/sections", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    result = draft_paper(
        sections_str=args.sections,
        output_format=args.output_format,
        config_path=args.config,
        log_path=args.log,
    )

    # Save sections
    if result.get("sections"):
        saved = save_paper_sections(result["sections"], args.output)
        for f in saved:
            print(f"Saved: {f}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        for name, content in result.get("sections", {}).items():
            print(f"\n{'=' * 60}")
            print(f"  {name.upper()}")
            print(f"{'=' * 60}\n")
            print(content)
            print()


if __name__ == "__main__":
    main()
