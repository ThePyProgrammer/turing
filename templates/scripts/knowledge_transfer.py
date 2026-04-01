#!/usr/bin/env python3
"""Cross-project knowledge transfer for the autoresearch pipeline.

Scans prior Turing projects for similar task characteristics and surfaces
what worked. Builds institutional ML memory across projects — the system
remembers what the researcher would otherwise rediscover.

Usage:
    python scripts/knowledge_transfer.py
    python scripts/knowledge_transfer.py --from ~/projects/fraud-detection
    python scripts/knowledge_transfer.py --auto
    python scripts/knowledge_transfer.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_INDEX_PATH = os.path.expanduser("~/.turing/project_index.yaml")
SCAN_DEPTH = 4  # Max directory depth to search for projects


# --- Project Signature ---


def extract_project_signature(
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Extract a project signature from config and experiment history.

    The signature captures task type, dataset characteristics, best model,
    and key insights — enough to match against other projects.

    Returns:
        Project signature dict.
    """
    config = load_config(config_path)
    experiments = load_experiments(log_path)

    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)
    task_type = config.get("task", {}).get("type", "classification")

    # Dataset characteristics
    dataset = config.get("dataset", config.get("data", {}))
    dataset_sig = {
        "task_type": task_type,
        "n_samples": dataset.get("n_samples", dataset.get("size")),
        "n_features": dataset.get("n_features", dataset.get("dimensionality")),
        "class_balance": dataset.get("class_balance"),
        "feature_types": dataset.get("feature_types", "mixed"),
    }

    # Best experiment
    kept = [e for e in experiments if e.get("status") == "kept"]
    best = None
    if kept:
        if lower_is_better:
            best = min(kept, key=lambda e: e.get("metrics", {}).get(primary_metric, float("inf")))
        else:
            best = max(kept, key=lambda e: e.get("metrics", {}).get(primary_metric, float("-inf")))

    best_sig = None
    if best:
        best_sig = {
            "experiment_id": best.get("experiment_id"),
            "model_type": best.get("config", {}).get("model_type", "unknown"),
            "primary_metric": primary_metric,
            "metric_value": best.get("metrics", {}).get(primary_metric),
            "hyperparams": best.get("config", {}).get("hyperparams", {}),
        }

    # What worked and what didn't
    model_stats = {}
    for exp in experiments:
        mt = exp.get("config", {}).get("model_type", "unknown")
        if mt not in model_stats:
            model_stats[mt] = {"kept": 0, "discarded": 0, "total": 0}
        model_stats[mt]["total"] += 1
        if exp.get("status") == "kept":
            model_stats[mt]["kept"] += 1
        elif exp.get("status") == "discarded":
            model_stats[mt]["discarded"] += 1

    # Key insights (from experiment patterns)
    insights = _extract_insights(experiments, model_stats, primary_metric)

    return {
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": primary_metric,
        "lower_is_better": lower_is_better,
        "dataset": dataset_sig,
        "best_experiment": best_sig,
        "model_stats": model_stats,
        "total_experiments": len(experiments),
        "kept_experiments": len(kept),
        "insights": insights,
    }


def _extract_insights(
    experiments: list[dict],
    model_stats: dict,
    primary_metric: str,
) -> list[str]:
    """Extract key insights from experiment history."""
    insights = []

    # Best model family
    best_family = None
    best_rate = 0
    for mt, stats in model_stats.items():
        if stats["total"] >= 2:
            rate = stats["kept"] / stats["total"]
            if rate > best_rate:
                best_rate = rate
                best_family = mt
    if best_family:
        insights.append(f"{best_family} had highest keep rate ({best_rate:.0%})")

    # Worst model family
    worst_family = None
    worst_rate = 1.0
    for mt, stats in model_stats.items():
        if stats["total"] >= 2:
            rate = stats["kept"] / stats["total"]
            if rate < worst_rate:
                worst_rate = rate
                worst_family = mt
    if worst_family and worst_family != best_family:
        insights.append(f"{worst_family} had lowest keep rate ({worst_rate:.0%})")

    # Experiment count
    if len(experiments) > 20:
        insights.append(f"Extensive search ({len(experiments)} experiments)")
    elif len(experiments) < 5:
        insights.append(f"Limited exploration ({len(experiments)} experiments)")

    return insights


# --- Project Similarity ---


def compute_similarity(sig_a: dict, sig_b: dict) -> float:
    """Compute similarity between two project signatures.

    Uses a weighted combination of task type match, dataset similarity,
    and feature type overlap.

    Returns:
        Similarity score in [0, 1].
    """
    scores = []
    weights = []

    # Task type (exact match)
    ds_a = sig_a.get("dataset", {})
    ds_b = sig_b.get("dataset", {})
    task_match = 1.0 if ds_a.get("task_type") == ds_b.get("task_type") else 0.0
    scores.append(task_match)
    weights.append(3.0)  # High weight

    # Feature types
    ft_a = ds_a.get("feature_types", "")
    ft_b = ds_b.get("feature_types", "")
    ft_match = 1.0 if ft_a == ft_b else 0.3
    scores.append(ft_match)
    weights.append(1.0)

    # Dataset size similarity (log scale)
    n_a = ds_a.get("n_samples")
    n_b = ds_b.get("n_samples")
    if n_a and n_b and n_a > 0 and n_b > 0:
        import math
        log_ratio = abs(math.log10(n_a) - math.log10(n_b))
        size_sim = max(0, 1 - log_ratio / 3)  # 1000x difference = 0
        scores.append(size_sim)
        weights.append(1.0)

    # Dimensionality similarity
    d_a = ds_a.get("n_features")
    d_b = ds_b.get("n_features")
    if d_a and d_b and d_a > 0 and d_b > 0:
        import math
        log_ratio = abs(math.log10(d_a) - math.log10(d_b))
        dim_sim = max(0, 1 - log_ratio / 2)
        scores.append(dim_sim)
        weights.append(0.5)

    # Class balance similarity
    bal_a = ds_a.get("class_balance")
    bal_b = ds_b.get("class_balance")
    if bal_a and bal_b:
        if bal_a == bal_b:
            scores.append(1.0)
        elif isinstance(bal_a, str) and isinstance(bal_b, str):
            # Both "balanced" or "imbalanced"
            scores.append(0.5)
        else:
            scores.append(0.0)
        weights.append(1.0)

    if not weights:
        return 0.0

    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return round(weighted_sum / sum(weights), 4)


# --- Project Index ---


def load_project_index(index_path: str = DEFAULT_INDEX_PATH) -> list[dict]:
    """Load the cross-project index."""
    path = Path(index_path)
    if not path.exists():
        return []

    with open(path) as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, list) else []


def save_project_index(index: list[dict], index_path: str = DEFAULT_INDEX_PATH) -> None:
    """Save the cross-project index."""
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(index, f, default_flow_style=False, sort_keys=False)


def index_project(
    project_path: str,
    signature: dict,
    index_path: str = DEFAULT_INDEX_PATH,
) -> None:
    """Add or update a project in the index."""
    index = load_project_index(index_path)

    # Remove existing entry for this path
    index = [p for p in index if p.get("path") != project_path]

    index.append({
        "path": project_path,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
    })

    save_project_index(index, index_path)


def scan_for_projects(
    search_roots: list[str] | None = None,
    max_depth: int = SCAN_DEPTH,
) -> list[str]:
    """Scan for Turing projects on the machine.

    Looks for directories containing both config.yaml and experiments/log.jsonl.

    Returns:
        List of project directory paths.
    """
    if search_roots is None:
        home = os.path.expanduser("~")
        search_roots = [
            os.path.join(home, "projects"),
            os.path.join(home, "ml"),
            os.path.join(home, "research"),
            os.getcwd(),
        ]

    projects = []
    seen = set()

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        _scan_dir(root, projects, seen, 0, max_depth)

    return projects


def _scan_dir(
    path: str,
    projects: list[str],
    seen: set[str],
    depth: int,
    max_depth: int,
) -> None:
    """Recursively scan for Turing projects."""
    if depth > max_depth:
        return

    real_path = os.path.realpath(path)
    if real_path in seen:
        return
    seen.add(real_path)

    config_path = os.path.join(path, "config.yaml")
    log_path = os.path.join(path, "experiments", "log.jsonl")

    if os.path.isfile(config_path) and os.path.isfile(log_path):
        projects.append(path)
        return  # Don't recurse into projects

    try:
        entries = os.listdir(path)
    except PermissionError:
        return

    for entry in sorted(entries):
        if entry.startswith(".") or entry in ("node_modules", ".venv", "__pycache__", "venv"):
            continue
        child = os.path.join(path, entry)
        if os.path.isdir(child):
            _scan_dir(child, projects, seen, depth + 1, max_depth)


# --- Transfer Recommendations ---


def generate_recommendations(
    current_sig: dict,
    similar_projects: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Generate transfer recommendations from similar projects.

    Args:
        current_sig: Current project signature.
        similar_projects: List of {path, similarity, signature} dicts.
        top_k: Number of top recommendations.

    Returns:
        List of recommendation dicts.
    """
    recommendations = []

    for proj in similar_projects[:top_k]:
        sig = proj.get("signature", {})
        best = sig.get("best_experiment")
        insights = sig.get("insights", [])

        rec = {
            "project_path": proj.get("path", "?"),
            "similarity": proj.get("similarity", 0),
            "task_type": sig.get("dataset", {}).get("task_type", "?"),
            "total_experiments": sig.get("total_experiments", 0),
        }

        if best:
            rec["winner"] = {
                "model_type": best.get("model_type", "?"),
                "metric_value": best.get("metric_value"),
                "metric_name": best.get("primary_metric", "?"),
            }

            # Generate hypothesis from winner
            model_type = best.get("model_type", "")
            hypothesis = f"Try {model_type}"
            hyperparams = best.get("hyperparams", {})
            key_params = []
            for k in ("max_depth", "n_estimators", "learning_rate", "hidden_size"):
                if k in hyperparams:
                    key_params.append(f"{k}={hyperparams[k]}")
            if key_params:
                hypothesis += f" with {', '.join(key_params)}"
            hypothesis += f" (transferred from {os.path.basename(proj.get('path', '?'))})"
            rec["hypothesis"] = hypothesis

        rec["insights"] = insights
        recommendations.append(rec)

    return recommendations


# --- Full Pipeline ---


def knowledge_transfer(
    from_path: str | None = None,
    auto_queue: bool = False,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
) -> dict:
    """Run cross-project knowledge transfer.

    Args:
        from_path: Specific project path to transfer from.
        auto_queue: Auto-queue hypotheses from recommendations.
        config_path: Current project config.
        log_path: Current project log.
        index_path: Cross-project index path.

    Returns:
        Transfer report dict.
    """
    # Extract current project signature
    current_sig = extract_project_signature(config_path, log_path)

    # Index current project
    cwd = os.getcwd()
    index_project(cwd, current_sig, index_path)

    if from_path:
        # Transfer from specific project
        from_config = os.path.join(from_path, "config.yaml")
        from_log = os.path.join(from_path, "experiments", "log.jsonl")
        if not os.path.isfile(from_config):
            return {"error": f"No config.yaml found at {from_path}"}
        if not os.path.isfile(from_log):
            return {"error": f"No experiments/log.jsonl found at {from_path}"}

        from_sig = extract_project_signature(from_config, from_log)
        similarity = compute_similarity(current_sig, from_sig)
        similar = [{"path": from_path, "similarity": similarity, "signature": from_sig}]
    else:
        # Search index for similar projects
        index = load_project_index(index_path)
        similar = []
        for entry in index:
            if entry.get("path") == cwd:
                continue  # Skip self
            sig = entry.get("signature", {})
            sim = compute_similarity(current_sig, sig)
            if sim > 0.3:  # Minimum similarity threshold
                similar.append({
                    "path": entry["path"],
                    "similarity": sim,
                    "signature": sig,
                })

        similar.sort(key=lambda x: x["similarity"], reverse=True)

    recommendations = generate_recommendations(current_sig, similar)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_project": cwd,
        "current_signature": {
            "task_type": current_sig.get("dataset", {}).get("task_type"),
            "total_experiments": current_sig.get("total_experiments", 0),
            "primary_metric": current_sig.get("primary_metric"),
        },
        "similar_projects_found": len(similar),
        "recommendations": recommendations,
    }

    if auto_queue and recommendations:
        report["auto_queued"] = [r.get("hypothesis") for r in recommendations if r.get("hypothesis")]

    return report


# --- Report Formatting ---


def save_transfer_report(report: dict, output_dir: str = "experiments/transfers") -> Path:
    """Save transfer report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filepath = out_path / f"transfer-{timestamp}.yaml"

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_transfer_report(report: dict) -> str:
    """Format transfer report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    lines = [
        "# Knowledge Transfer",
        "",
        f"*Generated {report.get('generated_at', 'N/A')[:19]}*",
        "",
    ]

    n_found = report.get("similar_projects_found", 0)
    if n_found == 0:
        lines.extend([
            "No similar prior projects found.",
            "",
            "Run `/turing:transfer` again after completing more projects,",
            "or specify a project directly with `--from /path/to/project`.",
        ])
        return "\n".join(lines)

    lines.append(f"**{n_found} similar project(s) found.**")
    lines.append("")

    for i, rec in enumerate(report.get("recommendations", []), 1):
        sim = rec.get("similarity", 0)
        path = rec.get("project_path", "?")
        task = rec.get("task_type", "?")
        n_exp = rec.get("total_experiments", 0)

        lines.extend([
            f"## {i}. {os.path.basename(path)} (similarity: {sim:.2f})",
            "",
            f"**Path:** {path}",
            f"**Task:** {task}, {n_exp} experiments",
        ])

        winner = rec.get("winner")
        if winner:
            val = winner.get("metric_value")
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            lines.append(
                f"**Winner:** {winner.get('model_type', '?')}, "
                f"{winner.get('metric_name', '?')}={val_str}"
            )

        insights = rec.get("insights", [])
        if insights:
            lines.append("")
            lines.append("**Insights:**")
            for ins in insights:
                lines.append(f"- {ins}")

        hypothesis = rec.get("hypothesis")
        if hypothesis:
            lines.extend(["", f"**Suggested hypothesis:** {hypothesis}"])

        lines.append("")

    # Auto-queued
    queued = report.get("auto_queued", [])
    if queued:
        lines.extend([
            "## Auto-Queued Hypotheses",
            "",
        ])
        for h in queued:
            lines.append(f"- {h}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-project knowledge transfer",
    )
    parser.add_argument(
        "--from", dest="from_path",
        help="Specific project path to transfer from",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-queue hypotheses from transfer recommendations",
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
        "--index", default=DEFAULT_INDEX_PATH,
        help=f"Path to project index (default: {DEFAULT_INDEX_PATH})",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    report = knowledge_transfer(
        from_path=args.from_path,
        auto_queue=args.auto,
        config_path=args.config,
        log_path=args.log,
        index_path=args.index,
    )

    if "error" not in report:
        filepath = save_transfer_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_transfer_report(report))


if __name__ == "__main__":
    main()
