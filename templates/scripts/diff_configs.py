"""Compare configurations and metrics between two experiments.

Computes a structured diff of config dicts (recursively) and metric deltas,
showing what changed, what was added, and what was removed between two
experiment log entries.

Usage:
    python scripts/diff_configs.py exp-005 exp-012              # Human-readable diff
    python scripts/diff_configs.py exp-005 exp-012 --json       # Machine-readable JSON
    python scripts/diff_configs.py exp-005 best                 # Compare against current best

The special keyword "best" resolves to the best kept experiment according to
the primary_metric and lower_is_better settings in config.yaml.

Ignored metadata fields: timestamp, experiment_id, git_commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"

# Fields stripped before diffing — they change every run and carry no signal
IGNORED_METADATA_FIELDS = {"timestamp", "experiment_id", "git_commit"}

# Hyperparameter-related config keys to surface in the focused HP section
HYPERPARAMETER_KEYS = {"hyperparams", "model", "learning_rate", "n_estimators",
                       "max_depth", "dropout", "weight_decay", "batch_size",
                       "optimizer", "scheduler", "epochs", "num_layers"}


# ---------------------------------------------------------------------------
# Experiment loading helpers
# ---------------------------------------------------------------------------

def load_experiment(log_path: str, experiment_id: str) -> dict | None:
    """Load a single experiment entry by ID from experiments/log.jsonl.

    Args:
        log_path: Path to the JSONL log file.
        experiment_id: Experiment ID string, e.g. "exp-005".

    Returns:
        Experiment dict, or None if the ID is not found.
    """
    path = Path(log_path)
    if not path.exists():
        return None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("experiment_id") == experiment_id:
                    return entry
            except json.JSONDecodeError:
                continue
    return None


def resolve_best(log_path: str, primary_metric: str, lower_is_better: bool) -> dict | None:
    """Return the best kept experiment by primary metric.

    Args:
        log_path: Path to the JSONL log file.
        primary_metric: Metric name used for ranking.
        lower_is_better: True for loss/error metrics, False for accuracy/F1.

    Returns:
        Best experiment dict, or None if no kept experiments exist.
    """
    experiments = load_experiments(log_path)
    best: dict | None = None
    best_value = float("inf") if lower_is_better else float("-inf")

    for exp in experiments:
        if exp.get("status") != "kept":
            continue
        value = exp.get("metrics", {}).get(primary_metric)
        if value is None:
            continue
        if (lower_is_better and value < best_value) or (not lower_is_better and value > best_value):
            best_value = value
            best = exp

    return best


# ---------------------------------------------------------------------------
# Core diff logic
# ---------------------------------------------------------------------------

def flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict with dot-separated keys.

    Args:
        d: Dict to flatten (may contain nested dicts).
        prefix: Key prefix accumulated during recursion.

    Returns:
        Flat dict mapping dot-separated paths to leaf values.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, full_key))
        else:
            out[full_key] = v
    return out


def strip_ignored(d: dict) -> dict:
    """Remove ignored metadata fields from a shallow dict copy."""
    return {k: v for k, v in d.items() if k not in IGNORED_METADATA_FIELDS}


def compute_config_diff(config_a: dict, config_b: dict) -> dict:
    """Compute a structured diff between two config dicts.

    Recursively compares nested dicts using dot-separated key paths.

    Args:
        config_a: Config from experiment A (baseline).
        config_b: Config from experiment B (comparison target).

    Returns:
        Dict with keys:
            "changed": {key: {"old": val_a, "new": val_b}}
            "added":   {key: val}   — keys only in B
            "removed": {key: val}   — keys only in A
            "unchanged_count": int  — number of identical keys (not listed)
    """
    flat_a = flatten_dict(strip_ignored(config_a))
    flat_b = flatten_dict(strip_ignored(config_b))

    all_keys = set(flat_a) | set(flat_b)
    changed: dict[str, dict] = {}
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    unchanged_count = 0

    for key in sorted(all_keys):
        in_a = key in flat_a
        in_b = key in flat_b

        if in_a and in_b:
            if flat_a[key] != flat_b[key]:
                changed[key] = {"old": flat_a[key], "new": flat_b[key]}
            else:
                unchanged_count += 1
        elif in_b:
            added[key] = flat_b[key]
        else:
            removed[key] = flat_a[key]

    return {
        "changed": changed,
        "added": added,
        "removed": removed,
        "unchanged_count": unchanged_count,
    }


def compute_metric_diff(
    metrics_a: dict,
    metrics_b: dict,
    primary_metric: str,
    lower_is_better: bool,
) -> dict:
    """Compute metric deltas with improvement direction indicators.

    Args:
        metrics_a: Metrics from experiment A.
        metrics_b: Metrics from experiment B.
        primary_metric: Name of the primary metric for the project.
        lower_is_better: True for loss/error metrics.

    Returns:
        Dict with keys:
            "primary": {metric: {val_a, val_b, delta, direction, is_improvement}}
            "others":  {metric: {val_a, val_b, delta, direction}}
            "added":   {metric: val}   — metrics only in B
            "removed": {metric: val}   — metrics only in A
    """
    all_keys = set(metrics_a) | set(metrics_b)
    primary_result: dict[str, Any] = {}
    others: dict[str, dict] = {}
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}

    for key in sorted(all_keys):
        in_a = key in metrics_a
        in_b = key in metrics_b

        if in_a and in_b:
            val_a = metrics_a[key]
            val_b = metrics_b[key]

            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                delta = val_b - val_a
                if abs(delta) < 1e-10:
                    direction = "="
                    is_improvement = False
                elif lower_is_better:
                    direction = "↓" if delta < 0 else "↑"
                    is_improvement = delta < 0
                else:
                    direction = "↑" if delta > 0 else "↓"
                    is_improvement = delta > 0

                entry = {
                    "val_a": val_a,
                    "val_b": val_b,
                    "delta": delta,
                    "direction": direction,
                    "is_improvement": is_improvement,
                }
            else:
                entry = {
                    "val_a": val_a,
                    "val_b": val_b,
                    "delta": None,
                    "direction": "=" if val_a == val_b else "~",
                    "is_improvement": False,
                }

            if key == primary_metric:
                primary_result[key] = entry
            else:
                others[key] = entry

        elif in_b:
            added[key] = metrics_b[key]
        else:
            removed[key] = metrics_a[key]

    return {
        "primary": primary_result,
        "others": others,
        "added": added,
        "removed": removed,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _is_hp_key(key: str) -> bool:
    """Return True if a flat config key belongs to a hyperparameter section."""
    parts = key.split(".")
    return bool(set(parts) & HYPERPARAMETER_KEYS)


def format_value(v: Any) -> str:
    """Render a value for display; floats use 6 significant figures."""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def format_text_diff(
    exp_a: dict,
    exp_b: dict,
    config_diff: dict,
    metric_diff: dict,
    primary_metric: str,
    lower_is_better: bool,
) -> str:
    """Render a human-readable diff report.

    Args:
        exp_a: Full experiment A dict.
        exp_b: Full experiment B dict.
        config_diff: Output of compute_config_diff.
        metric_diff: Output of compute_metric_diff.
        primary_metric: Name of primary metric.
        lower_is_better: True for loss/error metrics.

    Returns:
        Formatted multi-line string ready for printing.
    """
    id_a = exp_a.get("experiment_id", "?")
    id_b = exp_b.get("experiment_id", "?")

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"  Config & Metric Diff: {id_a}  →  {id_b}")
    lines.append("=" * 70)

    # --- Primary metric spotlight ---
    lines.append("")
    lines.append("PRIMARY METRIC")
    lines.append("-" * 40)
    if config_diff.get("primary"):
        # Shouldn't happen (primary_result is keyed by metric name), just a guard
        pass
    primary = metric_diff.get("primary", {})
    if primary:
        for metric_name, info in primary.items():
            val_a = info["val_a"]
            val_b = info["val_b"]
            delta = info["delta"]
            direction = info["direction"]
            improvement_label = ""
            if direction != "=":
                improvement_label = "  [IMPROVED]" if info["is_improvement"] else "  [REGRESSED]"

            direction_label = lower_is_better and "lower=better" or "higher=better"
            lines.append(
                f"  {metric_name:<20s}  {format_value(val_a):>12}  →  {format_value(val_b):<12}"
                f"  {direction}  Δ={format_value(delta) if delta is not None else 'N/A'}"
                f"  ({direction_label}){improvement_label}"
            )
    else:
        lines.append(f"  {primary_metric} not found in one or both experiments.")

    # --- Hyperparameter changes ---
    changed = config_diff.get("changed", {})
    added_cfg = config_diff.get("added", {})
    removed_cfg = config_diff.get("removed", {})

    hp_changed = {k: v for k, v in changed.items() if _is_hp_key(k)}
    hp_added = {k: v for k, v in added_cfg.items() if _is_hp_key(k)}
    hp_removed = {k: v for k, v in removed_cfg.items() if _is_hp_key(k)}

    other_changed = {k: v for k, v in changed.items() if not _is_hp_key(k)}
    other_added = {k: v for k, v in added_cfg.items() if not _is_hp_key(k)}
    other_removed = {k: v for k, v in removed_cfg.items() if not _is_hp_key(k)}

    if hp_changed or hp_added or hp_removed:
        lines.append("")
        lines.append("HYPERPARAMETER CHANGES")
        lines.append("-" * 40)
        for key, diff in sorted(hp_changed.items()):
            lines.append(f"  ~ {key:<30s}  {format_value(diff['old']):>15}  →  {format_value(diff['new'])}")
        for key, val in sorted(hp_added.items()):
            lines.append(f"  + {key:<30s}  {'':>15}     {format_value(val)}  (added)")
        for key, val in sorted(hp_removed.items()):
            lines.append(f"  - {key:<30s}  {format_value(val):>15}     {''}  (removed)")
    else:
        lines.append("")
        lines.append("HYPERPARAMETER CHANGES")
        lines.append("-" * 40)
        lines.append("  (no hyperparameter changes)")

    # --- Other config changes ---
    if other_changed or other_added or other_removed:
        lines.append("")
        lines.append("OTHER CONFIG CHANGES")
        lines.append("-" * 40)
        for key, diff in sorted(other_changed.items()):
            lines.append(f"  ~ {key:<30s}  {format_value(diff['old']):>15}  →  {format_value(diff['new'])}")
        for key, val in sorted(other_added.items()):
            lines.append(f"  + {key:<30s}  {'':>15}     {format_value(val)}  (added)")
        for key, val in sorted(other_removed.items()):
            lines.append(f"  - {key:<30s}  {format_value(val):>15}     {''}  (removed)")

    unchanged_count = config_diff.get("unchanged_count", 0)
    lines.append("")
    lines.append(f"  ({unchanged_count} config keys unchanged)")

    # --- Other metrics ---
    other_metrics = metric_diff.get("others", {})
    metric_added = metric_diff.get("added", {})
    metric_removed = metric_diff.get("removed", {})

    if other_metrics or metric_added or metric_removed:
        lines.append("")
        lines.append("OTHER METRICS")
        lines.append("-" * 40)
        for metric_name, info in sorted(other_metrics.items()):
            direction = info["direction"]
            delta = info["delta"]
            delta_str = format_value(delta) if delta is not None else "N/A"
            lines.append(
                f"  {metric_name:<22s}  {format_value(info['val_a']):>12}  →  {format_value(info['val_b']):<12}"
                f"  {direction}  Δ={delta_str}"
            )
        for metric_name, val in sorted(metric_added.items()):
            lines.append(f"  + {metric_name:<20s}  (only in {id_b}): {format_value(val)}")
        for metric_name, val in sorted(metric_removed.items()):
            lines.append(f"  - {metric_name:<20s}  (only in {id_a}): {format_value(val)}")

    # --- Experiment metadata footer ---
    lines.append("")
    lines.append("EXPERIMENT INFO")
    lines.append("-" * 40)
    lines.append(f"  {'ID':<12}  {id_a:<24}  {id_b}")
    lines.append(f"  {'Status':<12}  {exp_a.get('status', '?'):<24}  {exp_b.get('status', '?')}")
    lines.append(f"  {'Timestamp':<12}  {exp_a.get('timestamp', '?')[:19]:<24}  {exp_b.get('timestamp', '?')[:19]}")
    desc_a = (exp_a.get("description") or "")[:50]
    desc_b = (exp_b.get("description") or "")[:50]
    if desc_a or desc_b:
        lines.append(f"  {'Description':<12}  {desc_a:<24}  {desc_b}")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_json_diff(
    exp_a: dict,
    exp_b: dict,
    config_diff: dict,
    metric_diff: dict,
    primary_metric: str,
) -> str:
    """Render a machine-readable JSON diff.

    Args:
        exp_a: Full experiment A dict.
        exp_b: Full experiment B dict.
        config_diff: Output of compute_config_diff.
        metric_diff: Output of compute_metric_diff.
        primary_metric: Name of primary metric.

    Returns:
        JSON string.
    """
    result = {
        "experiment_a": exp_a.get("experiment_id"),
        "experiment_b": exp_b.get("experiment_id"),
        "primary_metric": primary_metric,
        "config_diff": config_diff,
        "metric_diff": metric_diff,
    }
    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for config and metric diffing."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare configurations and metrics between two experiments. "
            'Use "best" as the second experiment ID to compare against the '
            "current champion."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/diff_configs.py exp-005 exp-012\n"
            "  python scripts/diff_configs.py exp-005 exp-012 --json\n"
            "  python scripts/diff_configs.py exp-005 best\n"
        ),
    )
    parser.add_argument("exp_a", help='Baseline experiment ID (e.g. "exp-005")')
    parser.add_argument(
        "exp_b",
        help='Target experiment ID, or "best" to use the current champion',
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output machine-readable JSON instead of a text table",
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_PATH,
        help=f"Path to experiments/log.jsonl (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )

    args = parser.parse_args()

    # Load project config for primary_metric and direction
    cfg = load_config(args.config)
    eval_cfg = cfg.get("evaluation", {})
    primary_metric: str = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better: bool = eval_cfg.get("lower_is_better", False)

    # Resolve experiment A
    exp_a = load_experiment(args.log, args.exp_a)
    if exp_a is None:
        print(f"Error: experiment '{args.exp_a}' not found in {args.log}", file=sys.stderr)
        sys.exit(1)

    # Resolve experiment B (with "best" keyword support)
    if args.exp_b == "best":
        exp_b = resolve_best(args.log, primary_metric, lower_is_better)
        if exp_b is None:
            print(
                f"Error: no kept experiments found in {args.log} to use as 'best'.",
                file=sys.stderr,
            )
            sys.exit(1)
        resolved_id = exp_b.get("experiment_id", "?")
        if not args.json_output:
            print(f"  Resolved 'best' → {resolved_id}  (primary metric: {primary_metric})\n")
    else:
        exp_b = load_experiment(args.log, args.exp_b)
        if exp_b is None:
            print(f"Error: experiment '{args.exp_b}' not found in {args.log}", file=sys.stderr)
            sys.exit(1)

    # Compute diffs
    config_a = exp_a.get("config", {})
    config_b = exp_b.get("config", {})
    config_diff = compute_config_diff(config_a, config_b)

    metrics_a = exp_a.get("metrics", {})
    metrics_b = exp_b.get("metrics", {})
    metric_diff = compute_metric_diff(metrics_a, metrics_b, primary_metric, lower_is_better)

    # Render output
    if args.json_output:
        print(format_json_diff(exp_a, exp_b, config_diff, metric_diff, primary_metric))
    else:
        print(format_text_diff(exp_a, exp_b, config_diff, metric_diff, primary_metric, lower_is_better))


if __name__ == "__main__":
    main()
