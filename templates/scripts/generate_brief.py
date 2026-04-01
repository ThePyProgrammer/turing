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

from scripts.cost_frontier import compute_pareto_frontier, load_cost_data, _format_seconds
from scripts.turing_io import load_config, load_experiments, load_hypotheses
from scripts.seed_runner import CV_THRESHOLD


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


def load_queue_summary(queue_path: str = "experiments/queue-summary.yaml") -> dict | None:
    """Load the most recent queue execution summary."""
    path = Path(queue_path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return None


def load_profiles(profile_dir: str = "experiments/profiles") -> list[dict]:
    """Load all profiling results from YAML files."""
    path = Path(profile_dir)
    if not path.exists():
        return []
    profiles = []
    for f in sorted(path.glob("*-profile.yaml")):
        try:
            with open(f) as fh:
                profile = yaml.safe_load(fh)
                if profile and isinstance(profile, dict):
                    profiles.append(profile)
        except (yaml.YAMLError, OSError):
            continue
    return profiles


def load_diagnoses(diag_dir: str = "experiments/diagnoses") -> list[dict]:
    """Load all diagnosis reports from YAML files."""
    path = Path(diag_dir)
    if not path.exists():
        return []
    diagnoses = []
    for f in sorted(path.glob("*-diagnosis.yaml")):
        try:
            with open(f) as fh:
                diag = yaml.safe_load(fh)
                if diag and isinstance(diag, dict):
                    diagnoses.append(diag)
        except (yaml.YAMLError, OSError):
            continue
    return diagnoses


def load_seed_studies(seed_dir: str = "experiments/seed_studies") -> list[dict]:
    """Load all seed study results from YAML files."""
    path = Path(seed_dir)
    if not path.exists():
        return []
    studies = []
    for f in sorted(path.glob("*-seeds.yaml")):
        try:
            with open(f) as fh:
                study = yaml.safe_load(fh)
                if study and isinstance(study, dict):
                    studies.append(study)
        except (yaml.YAMLError, OSError):
            continue
    return studies


def load_reproductions(repro_dir: str = "experiments/reproductions") -> list[dict]:
    """Load all reproduction reports from YAML files."""
    path = Path(repro_dir)
    if not path.exists():
        return []
    reports = []
    for f in sorted(path.glob("*-repro.yaml")):
        try:
            with open(f) as fh:
                report = yaml.safe_load(fh)
                if report and isinstance(report, dict):
                    reports.append(report)
        except (yaml.YAMLError, OSError):
            continue
    return reports


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
    cost_data: list | None = None,
    cost_frontier: list | None = None,
    seed_studies: list[dict] | None = None,
    reproductions: list[dict] | None = None,
    diagnoses: list[dict] | None = None,
    profiles: list[dict] | None = None,
    queue_summary: dict | None = None,
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
            source = h.get("source", "")
            source_marker = f" [{source}]" if source in ("human", "treequest", "literature") else ""
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

    # Cost-performance analysis (only if train_seconds data exists)
    if cost_data and cost_frontier is not None:
        lines.extend(["", "## Cost-Performance Analysis", ""])

        frontier_ids = {r.experiment_id for r in cost_frontier}

        lines.append("**Pareto frontier (efficient set):**")
        if cost_frontier:
            for r in cost_frontier:
                lines.append(
                    f"- {r.experiment_id} ({r.model_type}): "
                    f"{metric}={r.metric_value:.4f}, time={_format_seconds(r.train_seconds)}"
                )
        else:
            lines.append("- No Pareto-optimal experiments found.")

        # Compare best metric vs cheapest frontier alternative
        if len(cost_frontier) >= 2:
            if lower_is_better:
                best_cost_exp = min(cost_data, key=lambda r: r.metric_value)
            else:
                best_cost_exp = max(cost_data, key=lambda r: r.metric_value)

            cheapest = cost_frontier[0]  # sorted by train_seconds

            if best_cost_exp.experiment_id != cheapest.experiment_id:
                metric_diff = abs(best_cost_exp.metric_value - cheapest.metric_value)
                if cheapest.metric_value != 0:
                    pct = metric_diff / abs(cheapest.metric_value) * 100
                else:
                    pct = 0.0
                if cheapest.train_seconds > 0:
                    ratio = best_cost_exp.train_seconds / cheapest.train_seconds
                else:
                    ratio = float("inf")

                lines.extend([
                    "",
                    f"Current best: **{best_cost_exp.experiment_id}** "
                    f"({best_cost_exp.metric_value:.4f}, {_format_seconds(best_cost_exp.train_seconds)})",
                    f"Cheapest acceptable: **{cheapest.experiment_id}** "
                    f"({cheapest.metric_value:.4f}, {_format_seconds(cheapest.train_seconds)})",
                    f"The {pct:.1f}% improvement costs {ratio:.0f}x more compute.",
                ])

    # Seed studies
    if seed_studies:
        lines.extend(["", "## Seed Studies", ""])
        for study in seed_studies:
            exp_id = study.get("experiment_id", "?")
            sensitive = study.get("seed_sensitive", False)
            status = "SEED-SENSITIVE" if sensitive else "STABLE"
            lines.append(
                f"- **{exp_id}:** {study.get('metric', metric)} = "
                f"{study.get('mean', 0):.4f} +/- {study.get('std', 0):.4f} "
                f"(CV={study.get('cv_percent', 0):.1f}%) — **{status}**"
            )
            if sensitive:
                lines.append(
                    f"  - 95% CI: [{study['ci_95'][0]:.4f}, {study['ci_95'][1]:.4f}] "
                    f"over {len(study.get('seeds_run', []))} seeds"
                )
        if any(s.get("seed_sensitive") for s in seed_studies):
            lines.extend(["", "*Some results are seed-sensitive. Report distributions, not point estimates.*"])

    # Reproduction reports
    if reproductions:
        lines.extend(["", "## Reproducibility", ""])
        verdict_markers = {
            "reproducible": "PASS",
            "approximately_reproducible": "PASS (approx)",
            "not_reproducible": "FAIL",
            "environment_changed": "WARN (env)",
        }
        for report in reproductions:
            exp_id = report.get("experiment_id", "?")
            verdict = report.get("verdict", "unknown")
            marker = verdict_markers.get(verdict, verdict)
            lines.append(f"- **{exp_id}:** {marker} — {report.get('reason', 'N/A')}")
        failed = [r for r in reproductions if r.get("verdict") in ("not_reproducible", "environment_changed")]
        if failed:
            lines.extend(["", f"*{len(failed)} experiment(s) failed reproducibility checks.*"])

    # Queue report
    if queue_summary and queue_summary.get("total"):
        qs = queue_summary
        lines.extend(["", "## Queue Report", ""])
        lines.append(
            f"**{qs.get('status', '?')}** — {qs.get('completed', 0)} completed, "
            f"{qs.get('failed', 0)} failed, {qs.get('skipped', 0)} skipped "
            f"of {qs.get('total', 0)} queued"
        )

    # Profiles
    if profiles:
        lines.extend(["", "## Performance Profile", ""])
        for prof in profiles[-1:]:  # Show most recent
            exp_id = prof.get("experiment_id", "?")
            p = prof.get("profile", {})
            bn = prof.get("bottleneck", {})
            lines.append(f"**{exp_id}:** {p.get('total_time_sec', 0):.1f}s total")
            mem = p.get("memory", {})
            if mem.get("peak_rss_mb"):
                lines.append(f"- Peak memory: {mem['peak_rss_mb']:.0f} MB")
            if bn.get("type") and bn["type"] != "none_detected":
                lines.append(f"- Bottleneck: **{bn['type']}** ({bn.get('severity', 'unknown')})")
            recs = prof.get("recommendations", [])
            if recs:
                lines.append(f"- Top recommendation: {recs[0]}")

    # Diagnoses (error analysis)
    if diagnoses:
        lines.extend(["", "## Error Analysis", ""])
        for diag in diagnoses:
            exp_id = diag.get("experiment_id", "?")
            modes = diag.get("failure_modes", [])
            if modes:
                lines.append(f"**{exp_id}** — {len(modes)} failure mode(s):")
                for mode in modes[:3]:
                    lines.append(f"- {mode.get('id', '?')}: {mode.get('description', 'N/A')}")
                if len(modes) > 3:
                    lines.append(f"  *...and {len(modes) - 3} more (see full diagnosis)*")
        auto_hyps = sum(len(d.get("auto_hypotheses", [])) for d in diagnoses)
        if auto_hyps:
            lines.append(f"\n*{auto_hyps} auto-generated hypotheses from failure analysis.*")

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
            lines.append("- No hypotheses queued — inject ideas with `/turing:try` or explore with `/turing:explore`")

    lines.extend(["", "---", "", "*Use `/turing:try` to inject hypotheses, `/turing:explore` for tree search, `/turing:train` to execute.*"])

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

    # Load cost-performance data if available
    cost_records = load_cost_data(log_path, metric)
    pareto = compute_pareto_frontier(cost_records, lower_is_better) if cost_records else []

    # Load seed studies, reproduction reports, diagnoses, and profiles
    seed_studies = load_seed_studies()
    reproductions = load_reproductions()
    diagnoses = load_diagnoses()
    profiles = load_profiles()
    queue_summary = load_queue_summary()

    return format_brief(
        campaign, best, trajectory, model_types, hypotheses,
        metric, lower_is_better, failures, env_warnings,
        cost_data=cost_records if cost_records else None,
        cost_frontier=pareto if cost_records else None,
        seed_studies=seed_studies if seed_studies else None,
        reproductions=reproductions if reproductions else None,
        diagnoses=diagnoses if diagnoses else None,
        profiles=profiles if profiles else None,
        queue_summary=queue_summary,
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
