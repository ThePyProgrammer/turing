#!/usr/bin/env python3
"""Deep experiment comparison for the autoresearch pipeline.

Goes beyond simple metric tables to answer "at what point did these two
experiments diverge and why?" Shows config diffs with magnitudes, metric
deltas with statistical significance, per-class performance regressions,
training curve divergence points, and feature importance shifts.

Usage:
    python scripts/experiment_diff.py exp-042 exp-053
    python scripts/experiment_diff.py exp-042 exp-053 --code
    python scripts/experiment_diff.py exp-042 exp-053 --json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"


def find_experiment(experiments: list[dict], exp_id: str) -> dict | None:
    """Find an experiment by ID."""
    for exp in experiments:
        if exp.get("experiment_id") == exp_id:
            return exp
    return None


# --- Config Diff ---


def diff_configs(config_a: dict, config_b: dict) -> list[dict]:
    """Compute config differences between two experiments.

    Flattens nested config dicts and computes magnitude of change
    for numeric values.

    Returns:
        List of diff dicts with keys: key, value_a, value_b, changed,
        and optionally pct_change for numeric values.
    """
    flat_a = _flatten_dict(config_a)
    flat_b = _flatten_dict(config_b)
    all_keys = sorted(set(flat_a) | set(flat_b))

    diffs = []
    for key in all_keys:
        val_a = flat_a.get(key)
        val_b = flat_b.get(key)
        changed = val_a != val_b

        entry = {
            "key": key,
            "value_a": val_a,
            "value_b": val_b,
            "changed": changed,
        }

        if changed and isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            if val_a != 0:
                entry["pct_change"] = (val_b - val_a) / abs(val_a) * 100
            else:
                entry["pct_change"] = float("inf") if val_b != 0 else 0.0

        diffs.append(entry)

    return diffs


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict with dot-separated keys."""
    items = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, full_key))
        else:
            items[full_key] = v
    return items


# --- Metric Diff ---


def diff_metrics(
    metrics_a: dict,
    metrics_b: dict,
    lower_is_better_metrics: set[str] | None = None,
    seed_studies: dict[str, dict] | None = None,
) -> list[dict]:
    """Compute metric differences with optional significance testing.

    Args:
        metrics_a: Metrics from experiment A.
        metrics_b: Metrics from experiment B.
        lower_is_better_metrics: Set of metric names where lower is better.
        seed_studies: Map of exp_id -> seed study data for significance.

    Returns:
        List of metric diff dicts.
    """
    if lower_is_better_metrics is None:
        lower_is_better_metrics = set()

    all_keys = sorted(set(metrics_a) | set(metrics_b))
    # Filter out non-numeric metadata
    metadata_keys = {"model_type", "train_seconds", "n_params", "model_size_bytes"}

    diffs = []
    for key in all_keys:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)

        entry = {
            "metric": key,
            "value_a": val_a,
            "value_b": val_b,
        }

        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            delta = val_b - val_a
            entry["delta"] = round(delta, 6)

            if key in lower_is_better_metrics:
                entry["direction"] = "better" if delta < 0 else "worse" if delta > 0 else "same"
            elif key not in metadata_keys:
                entry["direction"] = "better" if delta > 0 else "worse" if delta < 0 else "same"
            else:
                entry["direction"] = "N/A"

            # Significance from seed studies if available
            if seed_studies:
                entry["significance"] = _check_significance(
                    key, val_a, val_b, seed_studies,
                )

        diffs.append(entry)

    return diffs


def _check_significance(
    metric: str,
    val_a: float,
    val_b: float,
    seed_studies: dict[str, dict],
) -> dict | None:
    """Check if a metric difference is statistically significant.

    Uses seed study standard deviations to estimate a rough p-value
    via the pooled two-sample z-test approximation.
    """
    # Collect std estimates from any available seed studies
    stds = []
    for study in seed_studies.values():
        per_metric = study.get("per_metric", {})
        if metric in per_metric and "std" in per_metric[metric]:
            stds.append(per_metric[metric]["std"])

    if not stds:
        return None

    pooled_std = sum(stds) / len(stds)
    if pooled_std == 0:
        return {"significant": val_a != val_b, "method": "zero_variance"}

    z = abs(val_b - val_a) / (pooled_std * math.sqrt(2))

    # Approximate two-tailed p-value from z-score
    # Using the complementary error function approximation
    p_value = 2 * (1 - _norm_cdf(z))

    return {
        "z_score": round(z, 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "method": "pooled_z_test",
    }


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# --- Per-Class Diff ---


def diff_per_class(
    class_metrics_a: dict | None,
    class_metrics_b: dict | None,
) -> list[dict]:
    """Compare per-class performance between two experiments.

    Args:
        class_metrics_a: Dict of {class_name: {metric: value}} from exp A.
        class_metrics_b: Dict of {class_name: {metric: value}} from exp B.

    Returns:
        List of per-class diffs, highlighting regressions.
    """
    if not class_metrics_a or not class_metrics_b:
        return []

    all_classes = sorted(set(class_metrics_a) | set(class_metrics_b))
    diffs = []

    for cls in all_classes:
        a_metrics = class_metrics_a.get(cls, {})
        b_metrics = class_metrics_b.get(cls, {})
        all_metrics = sorted(set(a_metrics) | set(b_metrics))

        for metric in all_metrics:
            val_a = a_metrics.get(metric)
            val_b = b_metrics.get(metric)

            entry = {
                "class": cls,
                "metric": metric,
                "value_a": val_a,
                "value_b": val_b,
            }

            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                delta = val_b - val_a
                entry["delta"] = round(delta, 6)
                entry["regression"] = delta < -0.01  # Flag meaningful regressions

            diffs.append(entry)

    return diffs


# --- Training Curve Divergence ---


def find_curve_divergence(
    curve_a: list[dict] | None,
    curve_b: list[dict] | None,
    metric: str = "loss",
    threshold: float = 0.05,
) -> dict | None:
    """Find the epoch where two training curves meaningfully diverge.

    Args:
        curve_a: List of {epoch, metric_value} from experiment A.
        curve_b: List of {epoch, metric_value} from experiment B.
        metric: Which metric to compare.
        threshold: Relative difference to consider "diverged".

    Returns:
        Dict with divergence_epoch, metric values at divergence, or None.
    """
    if not curve_a or not curve_b:
        return None

    # Build epoch -> value maps
    map_a = {}
    map_b = {}
    for entry in curve_a:
        epoch = entry.get("epoch")
        val = entry.get(metric)
        if epoch is not None and val is not None:
            map_a[epoch] = val
    for entry in curve_b:
        epoch = entry.get("epoch")
        val = entry.get(metric)
        if epoch is not None and val is not None:
            map_b[epoch] = val

    common_epochs = sorted(set(map_a) & set(map_b))
    if not common_epochs:
        return None

    for epoch in common_epochs:
        va = map_a[epoch]
        vb = map_b[epoch]
        denom = abs(va) if va != 0 else 1.0
        rel_diff = abs(vb - va) / denom

        if rel_diff > threshold:
            return {
                "divergence_epoch": epoch,
                "value_a": round(va, 6),
                "value_b": round(vb, 6),
                "relative_diff": round(rel_diff, 4),
                "metric": metric,
                "total_common_epochs": len(common_epochs),
            }

    return None


# --- Feature Importance Diff ---


def diff_feature_importance(
    importance_a: dict | None,
    importance_b: dict | None,
    top_k: int = 10,
) -> list[dict]:
    """Compare feature importances between experiments.

    Args:
        importance_a: {feature_name: importance_value} from exp A.
        importance_b: {feature_name: importance_value} from exp B.
        top_k: Show top K features by absolute importance change.

    Returns:
        List of feature importance diffs, sorted by absolute delta.
    """
    if not importance_a or not importance_b:
        return []

    all_features = set(importance_a) | set(importance_b)
    diffs = []

    for feat in all_features:
        val_a = importance_a.get(feat, 0.0)
        val_b = importance_b.get(feat, 0.0)
        delta = val_b - val_a

        diffs.append({
            "feature": feat,
            "importance_a": round(val_a, 6),
            "importance_b": round(val_b, 6),
            "delta": round(delta, 6),
            "abs_delta": round(abs(delta), 6),
        })

    diffs.sort(key=lambda d: d["abs_delta"], reverse=True)
    return diffs[:top_k]


# --- Code Diff ---


def get_code_diff(commit_a: str | None, commit_b: str | None) -> str | None:
    """Get git diff of train.py between two experiment commits.

    Returns None if commits not available or git fails.
    """
    if not commit_a or not commit_b:
        return None

    try:
        result = subprocess.run(
            ["git", "diff", commit_a, commit_b, "--", "train.py"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


# --- Full Diff ---


def experiment_diff(
    exp_id_a: str,
    exp_id_b: str,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    include_code: bool = False,
) -> dict:
    """Compute a comprehensive diff between two experiments.

    Args:
        exp_id_a: First experiment ID.
        exp_id_b: Second experiment ID.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        include_code: Include git diff of train.py.

    Returns:
        Complete diff report dict.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)
    lower_is_better_metrics = set(eval_cfg.get("metrics", [])) if lower_is_better else set()

    experiments = load_experiments(log_path)

    exp_a = find_experiment(experiments, exp_id_a)
    exp_b = find_experiment(experiments, exp_id_b)

    if not exp_a:
        return {"error": f"Experiment {exp_id_a} not found in {log_path}"}
    if not exp_b:
        return {"error": f"Experiment {exp_id_b} not found in {log_path}"}

    # Load seed studies if available
    seed_studies = _load_seed_studies(exp_id_a, exp_id_b)

    report = {
        "experiment_a": exp_id_a,
        "experiment_b": exp_id_b,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": primary_metric,
    }

    # Config diff
    config_a = exp_a.get("config", {})
    config_b = exp_b.get("config", {})
    report["config_diff"] = diff_configs(config_a, config_b)

    # Metric diff
    metrics_a = exp_a.get("metrics", {})
    metrics_b = exp_b.get("metrics", {})
    report["metric_diff"] = diff_metrics(
        metrics_a, metrics_b, lower_is_better_metrics, seed_studies,
    )

    # Per-class diff
    class_a = exp_a.get("per_class_metrics")
    class_b = exp_b.get("per_class_metrics")
    report["per_class_diff"] = diff_per_class(class_a, class_b)

    # Training curve divergence
    curve_a = exp_a.get("training_curve")
    curve_b = exp_b.get("training_curve")
    report["curve_divergence"] = find_curve_divergence(curve_a, curve_b)

    # Feature importance diff
    imp_a = exp_a.get("feature_importance")
    imp_b = exp_b.get("feature_importance")
    report["feature_importance_diff"] = diff_feature_importance(imp_a, imp_b)

    # Code diff
    if include_code:
        commit_a = exp_a.get("git_commit")
        commit_b = exp_b.get("git_commit")
        report["code_diff"] = get_code_diff(commit_a, commit_b)

    # Summary verdict
    report["summary"] = _build_summary(report, primary_metric)

    return report


def _load_seed_studies(exp_id_a: str, exp_id_b: str) -> dict[str, dict]:
    """Load seed studies for both experiments if available."""
    from scripts.turing_io import load_seed_study

    studies = {}
    for exp_id in (exp_id_a, exp_id_b):
        study = load_seed_study(exp_id)
        if study:
            studies[exp_id] = study
    return studies


def _build_summary(report: dict, primary_metric: str) -> dict:
    """Build a summary of the key differences."""
    config_changes = [d for d in report.get("config_diff", []) if d["changed"]]
    metric_diffs = report.get("metric_diff", [])
    regressions = [d for d in report.get("per_class_diff", []) if d.get("regression")]
    divergence = report.get("curve_divergence")
    fi_shifts = report.get("feature_importance_diff", [])

    # Find primary metric change
    primary_change = None
    for m in metric_diffs:
        if m["metric"] == primary_metric:
            primary_change = m
            break

    return {
        "config_changes": len(config_changes),
        "metric_changes": len([d for d in metric_diffs if d.get("delta", 0) != 0]),
        "per_class_regressions": len(regressions),
        "has_curve_divergence": divergence is not None,
        "divergence_epoch": divergence["divergence_epoch"] if divergence else None,
        "feature_importance_shifts": len(fi_shifts),
        "primary_metric_delta": primary_change.get("delta") if primary_change else None,
        "primary_metric_direction": primary_change.get("direction") if primary_change else None,
    }


def save_diff_report(report: dict, output_dir: str = "experiments/diffs") -> Path:
    """Save diff report to YAML file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    a = report.get("experiment_a", "unknown")
    b = report.get("experiment_b", "unknown")
    filename = f"{a}-vs-{b}.yaml"
    filepath = out_path / filename

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_diff_report(report: dict) -> str:
    """Format diff report as human-readable markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    a = report["experiment_a"]
    b = report["experiment_b"]
    primary = report.get("primary_metric", "accuracy")

    lines = [
        f"# Experiment Diff: {a} vs {b}",
        "",
        f"*Generated {report.get('generated_at', 'N/A')[:19]}*",
        "",
    ]

    # Config diff
    config_diffs = report.get("config_diff", [])
    changed = [d for d in config_diffs if d["changed"]]
    if changed:
        lines.extend(["## Config Changes", ""])
        lines.append(f"| Parameter | {a} | {b} | Change |")
        lines.append("|-----------|-----|-----|--------|")
        for d in changed:
            pct = f" ({d['pct_change']:+.0f}%)" if "pct_change" in d else ""
            lines.append(
                f"| {d['key']} | {d['value_a']} | {d['value_b']} | {pct} |"
            )
        lines.append("")
    else:
        lines.extend(["## Config Changes", "", "No config differences.", ""])

    # Metric diff
    metric_diffs = report.get("metric_diff", [])
    if metric_diffs:
        lines.extend(["## Metric Comparison", ""])
        lines.append(f"| Metric | {a} | {b} | Delta | Verdict |")
        lines.append("|--------|-----|-----|-------|---------|")
        for m in metric_diffs:
            va = m.get("value_a")
            vb = m.get("value_b")
            va_str = f"{va:.4f}" if isinstance(va, float) else str(va)
            vb_str = f"{vb:.4f}" if isinstance(vb, float) else str(vb)
            delta_str = f"{m['delta']:+.4f}" if "delta" in m else "N/A"
            direction = m.get("direction", "")
            sig = m.get("significance")
            if sig and sig.get("significant"):
                direction += f" (p={sig['p_value']:.3f} sig)"
            elif sig and not sig.get("significant"):
                direction += f" (p={sig['p_value']:.3f} ns)"
            lines.append(
                f"| {m['metric']} | {va_str} | {vb_str} | {delta_str} | {direction} |"
            )
        lines.append("")

    # Per-class diff
    class_diffs = report.get("per_class_diff", [])
    if class_diffs:
        regressions = [d for d in class_diffs if d.get("regression")]
        lines.extend(["## Per-Class Performance", ""])
        if regressions:
            lines.append(f"**{len(regressions)} class regression(s) detected:**")
            lines.append("")
        lines.append(f"| Class | Metric | {a} | {b} | Delta | |")
        lines.append("|-------|--------|-----|-----|-------|-|")
        for d in class_diffs:
            va = d.get("value_a")
            vb = d.get("value_b")
            va_str = f"{va:.4f}" if isinstance(va, float) else str(va)
            vb_str = f"{vb:.4f}" if isinstance(vb, float) else str(vb)
            delta_str = f"{d['delta']:+.4f}" if "delta" in d else "N/A"
            flag = "REGRESSION" if d.get("regression") else ""
            lines.append(
                f"| {d['class']} | {d['metric']} | {va_str} | {vb_str} | {delta_str} | {flag} |"
            )
        lines.append("")

    # Curve divergence
    divergence = report.get("curve_divergence")
    if divergence:
        lines.extend([
            "## Training Curve Divergence",
            "",
            f"Curves diverge at **epoch {divergence['divergence_epoch']}** "
            f"({divergence['metric']}: {divergence['value_a']:.4f} vs {divergence['value_b']:.4f}, "
            f"{divergence['relative_diff']:.1%} relative difference)",
            f"out of {divergence['total_common_epochs']} common epochs.",
            "",
        ])

    # Feature importance
    fi_diffs = report.get("feature_importance_diff", [])
    if fi_diffs:
        lines.extend(["## Feature Importance Shifts", ""])
        lines.append(f"| Feature | {a} | {b} | Delta |")
        lines.append("|---------|-----|-----|-------|")
        for f in fi_diffs:
            lines.append(
                f"| {f['feature']} | {f['importance_a']:.4f} | {f['importance_b']:.4f} | {f['delta']:+.4f} |"
            )
        lines.append("")

    # Code diff
    code_diff = report.get("code_diff")
    if code_diff:
        lines.extend([
            "## Code Changes (train.py)",
            "",
            "```diff",
            code_diff,
            "```",
            "",
        ])

    # Summary
    summary = report.get("summary", {})
    lines.extend([
        "## Summary",
        "",
        f"- **Config changes:** {summary.get('config_changes', 0)}",
        f"- **Metric changes:** {summary.get('metric_changes', 0)}",
        f"- **Per-class regressions:** {summary.get('per_class_regressions', 0)}",
    ])
    if summary.get("has_curve_divergence"):
        lines.append(f"- **Curves diverge at epoch:** {summary['divergence_epoch']}")
    if summary.get("primary_metric_delta") is not None:
        lines.append(
            f"- **{primary} delta:** {summary['primary_metric_delta']:+.4f} "
            f"({summary.get('primary_metric_direction', 'N/A')})"
        )

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deep experiment comparison",
    )
    parser.add_argument(
        "exp_a",
        help="First experiment ID (e.g., exp-042)",
    )
    parser.add_argument(
        "exp_b",
        help="Second experiment ID (e.g., exp-053)",
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
        "--code", action="store_true",
        help="Include git diff of train.py between experiments",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    report = experiment_diff(
        exp_id_a=args.exp_a,
        exp_id_b=args.exp_b,
        config_path=args.config,
        log_path=args.log,
        include_code=args.code,
    )

    # Save report
    if "error" not in report:
        filepath = save_diff_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    # Output
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_diff_report(report))

    # Exit code: 1 if regressions detected
    summary = report.get("summary", {})
    if summary.get("per_class_regressions", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
