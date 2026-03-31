#!/usr/bin/env python3
"""Research briefing generator for the autoresearch pipeline.

Produces a structured intelligence report from experiment history:
what's been learned, what's promising, what's exhausted, and what
the human should consider next.

This closes the taste-leverage loop: the agent reports intelligence,
the human applies taste, the human injects hypotheses, the agent
executes them.

Usage:
    python scripts/generate_brief.py [--config config.yaml] [--log experiments/log.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_experiments(log_path: str) -> list[dict]:
    """Load all experiments from JSONL log."""
    path = Path(log_path)
    if not path.exists():
        return []
    experiments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    experiments.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return experiments


def load_hypotheses(queue_path: str) -> list[dict]:
    """Load hypothesis queue."""
    path = Path(queue_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def load_config(config_path: str) -> dict:
    """Load config, return empty dict on failure."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def compute_campaign_summary(experiments: list[dict]) -> dict:
    """Compute high-level campaign statistics."""
    total = len(experiments)
    kept = sum(1 for e in experiments if e.get("status") == "kept")
    discarded = sum(1 for e in experiments if e.get("status") == "discarded")

    timestamps = []
    for e in experiments:
        ts = e.get("timestamp", "")
        if ts:
            timestamps.append(ts)

    first_ts = min(timestamps) if timestamps else None
    last_ts = max(timestamps) if timestamps else None

    return {
        "total": total,
        "kept": kept,
        "discarded": discarded,
        "keep_rate": round(kept / total, 2) if total > 0 else 0,
        "first_experiment": first_ts,
        "last_experiment": last_ts,
    }


def find_best(experiments: list[dict], metric: str, lower_is_better: bool) -> dict | None:
    """Find best kept experiment by primary metric."""
    best = None
    best_val = float("inf") if lower_is_better else float("-inf")
    for e in experiments:
        if e.get("status") != "kept":
            continue
        val = e.get("metrics", {}).get(metric)
        if val is None:
            continue
        if (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val = val
            best = e
    return best


def compute_trajectory(experiments: list[dict], metric: str, lower_is_better: bool) -> list[dict]:
    """Compute improvement trajectory — best metric value after each kept experiment."""
    trajectory = []
    best_val = float("inf") if lower_is_better else float("-inf")
    for e in experiments:
        if e.get("status") != "kept":
            continue
        val = e.get("metrics", {}).get(metric)
        if val is None:
            continue
        if (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val = val
        trajectory.append({
            "experiment_id": e.get("experiment_id", "?"),
            "value": val,
            "best_so_far": best_val,
        })
    return trajectory


def identify_model_types(experiments: list[dict]) -> dict[str, dict]:
    """Group experiments by model type and compute stats."""
    models: dict[str, list[dict]] = {}
    for e in experiments:
        mt = e.get("config", {}).get("model_type", "unknown")
        models.setdefault(mt, []).append(e)

    result = {}
    for mt, exps in models.items():
        kept = [e for e in exps if e.get("status") == "kept"]
        result[mt] = {
            "total": len(exps),
            "kept": len(kept),
            "discarded": len(exps) - len(kept),
        }
    return result


def cluster_failures(experiments: list[dict]) -> list[dict]:
    """Identify patterns across failed (discarded) experiments.

    Groups discarded experiments by common traits and reports clusters
    where multiple experiments share the same failure characteristic.

    Returns list of cluster dicts with: trait, count, experiments, description.
    """
    discarded = [e for e in experiments if e.get("status") == "discarded"]
    if len(discarded) < 2:
        return []

    clusters: dict[str, list[str]] = {}

    for exp in discarded:
        exp_id = exp.get("experiment_id", "?")
        config = exp.get("config", {})
        hyperparams = config.get("hyperparams", {})

        # Cluster by model type
        mt = config.get("model_type", "unknown")
        key = f"model_type={mt}"
        clusters.setdefault(key, []).append(exp_id)

        # Cluster by hyperparameter ranges
        for param, value in hyperparams.items():
            if isinstance(value, (int, float)):
                # Bin into high/low relative to a simple threshold
                key = f"{param}>={value}" if isinstance(value, int) else f"{param}~{value}"
                clusters.setdefault(key, []).append(exp_id)

        # Cluster by family
        family = exp.get("family")
        if family:
            key = f"family={family}"
            clusters.setdefault(key, []).append(exp_id)

    # Filter to clusters with 2+ experiments
    result = []
    for trait, exp_ids in clusters.items():
        if len(exp_ids) >= 2:
            result.append({
                "trait": trait,
                "count": len(exp_ids),
                "experiments": exp_ids,
                "description": f"{len(exp_ids)} discarded experiments share: {trait}",
            })

    # Sort by count descending
    result.sort(key=lambda c: -c["count"])
    return result[:5]  # Top 5 clusters


def detect_environment_drift(experiments: list[dict]) -> list[str]:
    """Detect environment changes across experiments.

    Compares the most recent experiment's environment against the best
    experiment's environment. Flags differences in python version,
    key package versions, or hardware that could affect reproducibility.
    """
    if len(experiments) < 2:
        return []

    # Find most recent and best experiments with environment data
    recent = None
    best_env_exp = None
    for e in reversed(experiments):
        if e.get("environment") and not recent:
            recent = e
        if e.get("status") == "kept" and e.get("environment") and not best_env_exp:
            best_env_exp = e
        if recent and best_env_exp:
            break

    if not recent or not best_env_exp or recent == best_env_exp:
        return []

    warnings = []
    env_new = recent["environment"]
    env_old = best_env_exp["environment"]

    # Python version
    if env_new.get("python_version") != env_old.get("python_version"):
        warnings.append(
            f"Python version changed: {env_old.get('python_version')} -> {env_new.get('python_version')}"
        )

    # Key packages
    pkgs_new = env_new.get("packages", {})
    pkgs_old = env_old.get("packages", {})
    for pkg in set(pkgs_new) | set(pkgs_old):
        v_new = pkgs_new.get(pkg)
        v_old = pkgs_old.get(pkg)
        if v_new and v_old and v_new != v_old:
            warnings.append(f"{pkg}: {v_old} -> {v_new}")

    # Config hash drift
    hash_new = env_new.get("config_hash")
    hash_old = env_old.get("config_hash")
    if hash_new and hash_old and hash_new != hash_old:
        warnings.append("config.yaml has changed since best experiment")

    return warnings


def format_brief(
    campaign: dict,
    best: dict | None,
    trajectory: list[dict],
    model_types: dict[str, dict],
    hypotheses: list[dict],
    metric: str,
    lower_is_better: bool,
    failure_clusters: list[dict] | None = None,
    env_warnings: list[str] | None = None,
) -> str:
    """Format the research briefing as markdown."""
    direction = "lower" if lower_is_better else "higher"
    lines = [
        "# Research Briefing",
        "",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "---",
        "",
        "## Campaign Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total experiments | {campaign['total']} |",
        f"| Kept | {campaign['kept']} ({campaign['keep_rate']:.0%} keep rate) |",
        f"| Discarded | {campaign['discarded']} |",
        f"| Primary metric | {metric} ({direction} is better) |",
    ]
    if campaign["first_experiment"]:
        lines.append(f"| First experiment | {campaign['first_experiment'][:19]} |")
    if campaign["last_experiment"]:
        lines.append(f"| Last experiment | {campaign['last_experiment'][:19]} |")

    lines.extend(["", "## Current Best", ""])
    if best:
        metrics_str = ", ".join(f"{k}={v}" for k, v in best.get("metrics", {}).items())
        lines.extend([
            f"- **Experiment:** {best.get('experiment_id', '?')}",
            f"- **Model:** {best.get('config', {}).get('model_type', '?')}",
            f"- **Metrics:** {metrics_str}",
            f"- **Description:** {best.get('description', 'N/A')}",
        ])
    else:
        lines.append("No kept experiments yet.")

    lines.extend(["", "## Improvement Trajectory", ""])
    if trajectory:
        lines.append(f"| Experiment | {metric} | Best So Far |")
        lines.append(f"|------------|{'---' * len(metric)}--|-------------|")
        for t in trajectory[-10:]:  # Last 10
            lines.append(f"| {t['experiment_id']} | {t['value']:.4f} | {t['best_so_far']:.4f} |")
        if len(trajectory) > 10:
            lines.insert(-10, f"*...showing last 10 of {len(trajectory)} kept experiments*")
    else:
        lines.append("No trajectory data yet.")

    lines.extend(["", "## Model Types Explored", ""])
    if model_types:
        lines.append("| Model | Experiments | Kept | Discarded |")
        lines.append("|-------|-------------|------|-----------|")
        for mt, stats in sorted(model_types.items()):
            lines.append(f"| {mt} | {stats['total']} | {stats['kept']} | {stats['discarded']} |")
    else:
        lines.append("No experiments yet.")

    lines.extend(["", "## Hypothesis Queue", ""])
    queued = [h for h in hypotheses if h.get("status") == "queued"]
    tested = [h for h in hypotheses if h.get("status") in ("tested", "promising", "dead-end")]
    if queued:
        lines.append(f"**{len(queued)} queued:**")
        for h in queued:
            priority_marker = " (HIGH)" if h.get("priority") == "high" else ""
            source_marker = " [human]" if h.get("source") == "human" else ""
            lines.append(f"- {h['id']}: {h.get('description', '?')}{priority_marker}{source_marker}")
    else:
        lines.append("No queued hypotheses. Use `/turing:try` to inject ideas.")

    if tested:
        lines.extend(["", f"**{len(tested)} tested:**"])
        for h in tested:
            result = f" -> {h['result_experiment']}" if h.get("result_experiment") else ""
            lines.append(f"- {h['id']}: {h.get('description', '?')} [{h.get('status')}]{result}")

    # Failure patterns
    if failure_clusters:
        lines.extend(["", "## Failure Patterns", ""])
        for cluster in failure_clusters:
            exps = ", ".join(cluster["experiments"][:5])
            lines.append(f"- **{cluster['trait']}** — {cluster['count']} discarded experiments ({exps})")
        lines.append("")
        lines.append("*Consider avoiding these traits in future experiments.*")

    # Environment drift warnings
    if env_warnings:
        lines.extend(["", "## Environment Drift", ""])
        lines.append("The runtime environment has changed since the best experiment:")
        for w in env_warnings:
            lines.append(f"- {w}")
        lines.append("")
        lines.append("*Results may not be directly comparable. Consider re-running the best experiment in the current environment.*")

    lines.extend([
        "",
        "## Recommendations",
        "",
        "Based on experiment history:",
        "",
    ])

    if not trajectory:
        lines.append("- Run `/turing:train` to begin the experiment loop")
    elif len(trajectory) < 3:
        lines.append("- Too few experiments for meaningful recommendations. Continue training.")
    else:
        # Check if recent experiments are improving
        recent = trajectory[-3:]
        improving = recent[-1]["best_so_far"] != recent[0]["best_so_far"]
        if improving:
            lines.append("- Current direction is productive — continue exploring this model type")
        else:
            lines.append("- Improvement has plateaued — consider switching model type or feature approach")

        # Check model diversity
        if len(model_types) == 1:
            lines.append("- Only one model type explored — try alternatives (LightGBM, RandomForest, MLP)")

        # Check if hypotheses are exhausted
        if not queued:
            lines.append("- No hypotheses queued — inject ideas with `/turing:try`")

    lines.extend(["", "---", "", "*Use `/turing:try` to inject hypotheses. Use `/turing:train` to execute.*"])

    return "\n".join(lines)


def generate_brief(
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
    hypotheses_path: str = "hypotheses.yaml",
) -> str:
    """Generate a research briefing from experiment history."""
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)
    hypotheses = load_hypotheses(hypotheses_path)

    campaign = compute_campaign_summary(experiments)
    best = find_best(experiments, metric, lower_is_better)
    trajectory = compute_trajectory(experiments, metric, lower_is_better)
    model_types = identify_model_types(experiments)
    failures = cluster_failures(experiments)
    env_warnings = detect_environment_drift(experiments)

    return format_brief(
        campaign, best, trajectory, model_types, hypotheses,
        metric, lower_is_better, failures, env_warnings,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate research briefing")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl")
    parser.add_argument("--hypotheses", default="hypotheses.yaml")
    args = parser.parse_args()

    brief = generate_brief(args.config, args.log, args.hypotheses)
    print(brief)


if __name__ == "__main__":
    main()
