"""Ranked leaderboard of experiments by primary metric.

One-glance view of where we stand: experiments ranked best-to-worst
with model type, key hyperparams, delta vs leader, and kept/discarded status.
Reads experiments/log.jsonl and config.yaml.

Usage:
    python scripts/leaderboard.py                     # Top kept experiments
    python scripts/leaderboard.py --top 5             # Top 5 only
    python scripts/leaderboard.py --status all        # Include discarded
    python scripts/leaderboard.py --format markdown   # For docs/README
    python scripts/leaderboard.py --metric f1_weighted  # Rank by F1
    python scripts/leaderboard.py --compact           # Minimal output
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"

# Key hyperparameter names to surface in the compact column.
# Order matters — first match wins when space is limited.
_HYPERPARAM_KEYS = [
    "max_depth",
    "depth",
    "learning_rate",
    "lr",
    "n_estimators",
    "n_est",
    "num_leaves",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
    "C",
    "gamma",
    "hidden_size",
    "num_layers",
    "dropout",
    "batch_size",
    "epochs",
]

_HYPERPARAM_ALIASES = {
    "max_depth": "depth",
    "learning_rate": "lr",
    "n_estimators": "n_est",
    "num_leaves": "n_leaves",
    "colsample_bytree": "col_samp",
    "min_child_weight": "min_cw",
    "hidden_size": "hidden",
    "num_layers": "layers",
    "batch_size": "bs",
}


def _compact_hyperparams(hyperparams: dict, max_pairs: int = 4) -> str:
    """Format hyperparams as a compact key=value string.

    Picks the most informative parameters (bias toward the key ones defined
    in _HYPERPARAM_KEYS), aliases long names, and truncates beyond max_pairs.
    """
    if not hyperparams:
        return "—"

    chosen: list[tuple[str, object]] = []

    # First pass: grab known interesting keys in priority order.
    for key in _HYPERPARAM_KEYS:
        if key in hyperparams:
            alias = _HYPERPARAM_ALIASES.get(key, key)
            chosen.append((alias, hyperparams[key]))
        if len(chosen) >= max_pairs:
            break

    # Second pass: fill remaining slots with leftover keys.
    if len(chosen) < max_pairs:
        seen = {k for k, _ in chosen}
        for key, val in hyperparams.items():
            if key not in seen and isinstance(val, (int, float, str, bool)):
                alias = _HYPERPARAM_ALIASES.get(key, key)
                chosen.append((alias, val))
                if len(chosen) >= max_pairs:
                    break

    parts = []
    for k, v in chosen:
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")

    return ", ".join(parts) if parts else "—"


def _fmt_metric(value: float | None, precision: int = 4) -> str:
    """Format a metric value, returning '—' for None."""
    if value is None:
        return "—"
    return f"{value:.{precision}f}"


def _fmt_delta(delta: float | None, lower_is_better: bool) -> str:
    """Format delta vs leader.  Always non-negative; sign indicates direction."""
    if delta is None:
        return "—"
    if abs(delta) < 1e-9:
        return "—"  # This IS the leader; no delta shown.
    # delta is already (leader_value - this_value) in absolute terms,
    # normalised so that positive = worse than leader.
    return f"-{abs(delta):.4f}"


def _status_marker(status: str) -> str:
    return "✓" if status == "kept" else "✗"


def _sort_key(exp: dict, metric: str, lower_is_better: bool):
    """Sort key: best first regardless of direction."""
    val = exp.get("metrics", {}).get(metric)
    if val is None:
        # Push experiments with missing metric to the bottom.
        return float("inf") if lower_is_better else float("-inf")
    return val if lower_is_better else -val


def rank_experiments(
    experiments: list[dict],
    metric: str,
    lower_is_better: bool,
    status_filter: str,
) -> list[dict]:
    """Filter and rank experiments, returning list with injected 'rank' field."""
    if status_filter == "kept":
        filtered = [e for e in experiments if e.get("status") == "kept"]
    else:
        filtered = list(experiments)

    ranked = sorted(filtered, key=lambda e: _sort_key(e, metric, lower_is_better))

    # Inject rank
    for i, exp in enumerate(ranked):
        exp = dict(exp)  # shallow copy — don't mutate the original
        exp["_rank"] = i + 1
        ranked[i] = exp

    return ranked


def compute_delta(ranked: list[dict], metric: str, lower_is_better: bool) -> list[dict]:
    """Inject '_delta' key: absolute gap behind the leader (positive = worse)."""
    if not ranked:
        return ranked

    leader_val = ranked[0].get("metrics", {}).get(metric)

    result = []
    for exp in ranked:
        exp = dict(exp)
        val = exp.get("metrics", {}).get(metric)
        if leader_val is not None and val is not None:
            # For higher-is-better: leader - this (positive means worse).
            # For lower-is-better: this - leader (positive means worse).
            if lower_is_better:
                exp["_delta"] = val - leader_val
            else:
                exp["_delta"] = leader_val - val
        else:
            exp["_delta"] = None
        result.append(exp)

    return result


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _build_rows(ranked: list[dict], metric: str, lower_is_better: bool) -> list[dict]:
    """Build uniform row dicts for all formatters."""
    rows = []
    for exp in ranked:
        rank = exp["_rank"]
        exp_id = exp.get("experiment_id", "?")
        model_type = exp.get("config", {}).get("model_type", "?")
        hyperparams = exp.get("config", {}).get("hyperparams", {})
        metric_val = exp.get("metrics", {}).get(metric)
        delta = exp.get("_delta")
        status = exp.get("status", "?")

        rows.append({
            "rank": rank,
            "rank_label": f"#{rank}",
            "experiment_id": exp_id,
            "model_type": model_type,
            "metric_value": metric_val,
            "metric_str": _fmt_metric(metric_val),
            "hyperparams": _compact_hyperparams(hyperparams),
            "delta": _fmt_delta(delta, lower_is_better),
            "status": status,
            "status_marker": _status_marker(status),
        })
    return rows


def format_text(rows: list[dict], metric: str) -> str:
    """Render as a fixed-width text table."""
    if not rows:
        return "No experiments to display."

    # Column widths
    w_rank = 4
    w_id = max(len(r["experiment_id"]) for r in rows)
    w_id = max(w_id, len("Experiment"))
    w_model = max(len(r["model_type"]) for r in rows)
    w_model = max(w_model, len("Model"))
    w_metric = max(len(r["metric_str"]) for r in rows)
    w_metric = max(w_metric, len(metric))
    w_hp = max(len(r["hyperparams"]) for r in rows)
    w_hp = max(w_hp, len("Hyperparams"))
    w_delta = max(len(r["delta"]) for r in rows)
    w_delta = max(w_delta, len("vs #1"))
    w_status = 6  # "Status"

    def sep():
        return (
            "+"
            + "-" * (w_rank + 2)
            + "+"
            + "-" * (w_id + 2)
            + "+"
            + "-" * (w_model + 2)
            + "+"
            + "-" * (w_metric + 2)
            + "+"
            + "-" * (w_hp + 2)
            + "+"
            + "-" * (w_delta + 2)
            + "+"
            + "-" * (w_status + 2)
            + "+"
        )

    def row_line(rank, exp_id, model, metric_v, hp, delta, status, highlight=False):
        metric_cell = metric_v.ljust(w_metric)
        if highlight:
            metric_cell = f"[{metric_cell.strip()}]".ljust(w_metric)
        return (
            f"| {rank:<{w_rank}} "
            f"| {exp_id:<{w_id}} "
            f"| {model:<{w_model}} "
            f"| {metric_cell} "
            f"| {hp:<{w_hp}} "
            f"| {delta:<{w_delta}} "
            f"| {status:<{w_status}} |"
        )

    lines = [sep()]
    header = row_line(
        "Rank", "Experiment", "Model", metric[:w_metric], "Hyperparams", "vs #1", "Status"
    )
    lines.append(header)
    lines.append(sep())

    for r in rows:
        highlight = r["rank"] == 1 and r["metric_value"] is not None
        line = row_line(
            r["rank_label"],
            r["experiment_id"],
            r["model_type"],
            r["metric_str"],
            r["hyperparams"],
            r["delta"],
            r["status_marker"],
            highlight=highlight,
        )
        lines.append(line)

    lines.append(sep())
    return "\n".join(lines)


def format_markdown(rows: list[dict], metric: str) -> str:
    """Render as a GitHub-flavored Markdown table."""
    if not rows:
        return "_No experiments to display._"

    header = f"| Rank | Experiment | Model | {metric} | Hyperparams | vs #1 | Status |"
    sep = "|------|------------|-------|" + "-" * (len(metric) + 2) + "|-------------|-------|--------|"

    lines = [header, sep]
    for r in rows:
        metric_cell = r["metric_str"]
        if r["rank"] == 1 and r["metric_value"] is not None:
            metric_cell = f"**{metric_cell}**"
        lines.append(
            f"| {r['rank_label']} "
            f"| {r['experiment_id']} "
            f"| {r['model_type']} "
            f"| {metric_cell} "
            f"| {r['hyperparams']} "
            f"| {r['delta']} "
            f"| {r['status_marker']} |"
        )

    return "\n".join(lines)


def format_csv(rows: list[dict], metric: str) -> str:
    """Render as CSV."""
    if not rows:
        return ""

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["rank", "experiment_id", "model_type", metric, "hyperparams", "vs_leader", "status"])
    for r in rows:
        writer.writerow([
            r["rank"],
            r["experiment_id"],
            r["model_type"],
            r["metric_str"],
            r["hyperparams"],
            r["delta"],
            r["status"],
        ])
    return buf.getvalue().rstrip()


def format_compact(rows: list[dict], metric: str) -> str:
    """One line per experiment, no borders."""
    if not rows:
        return "No experiments to display."

    lines = []
    for r in rows:
        marker = "*" if r["rank"] == 1 else " "
        lines.append(
            f"{marker}{r['rank_label']:<4}  {r['experiment_id']:<10}  "
            f"{r['model_type']:<15}  {metric}={r['metric_str']}  "
            f"({r['hyperparams']})  vs#1={r['delta']}  {r['status_marker']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def build_footer(
    all_experiments: list[dict],
    ranked: list[dict],
    metric: str,
) -> str:
    """Compose the summary footer."""
    total = len(all_experiments)
    total_kept = sum(1 for e in all_experiments if e.get("status") == "kept")

    lines = []

    if not ranked:
        lines.append(f"Total experiments: {total}  |  Kept: {total_kept}")
        return "\n".join(lines)

    leader = ranked[0]
    leader_id = leader.get("experiment_id", "?")
    leader_val = leader.get("metrics", {}).get(metric)
    leader_val_str = _fmt_metric(leader_val)

    lines.append(
        f"Total: {total} experiments  |  Kept: {total_kept}"
        f"  |  Best {metric}: {leader_val_str} ({leader_id})"
    )

    # Gap between #1 and #2
    if len(ranked) >= 2:
        second = ranked[1]
        second_val = second.get("metrics", {}).get(metric)
        if leader_val is not None and second_val is not None:
            gap = abs(leader_val - second_val)
            lines.append(f"Gap #1 → #2: {gap:.4f}")

    # Timestamp of most recent experiment (by timestamp field, not rank)
    timestamps = [
        e.get("timestamp", "")
        for e in all_experiments
        if e.get("timestamp")
    ]
    if timestamps:
        latest = max(timestamps)
        lines.append(f"Most recent: {latest[:19]} UTC")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ranked leaderboard of experiments by primary metric.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_PATH,
        help=f"Path to experiment log (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--status",
        choices=["kept", "all"],
        default="kept",
        help="Which experiments to include: 'kept' (default) or 'all'",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Show top N experiments only (default: all)",
    )
    parser.add_argument(
        "--metric",
        default=None,
        metavar="NAME",
        help="Rank by this metric (default: primary_metric from config.yaml)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "csv"],
        default="text",
        dest="fmt",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="One-line-per-experiment minimal output (overrides --format)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config()
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)
    metric = args.metric if args.metric else primary_metric

    # Load experiments
    all_experiments = load_experiments(args.log)

    if not all_experiments:
        print(f"No experiments found in {args.log}.", file=sys.stderr)
        sys.exit(0)

    # Filter and rank
    ranked = rank_experiments(all_experiments, metric, lower_is_better, args.status)

    if not ranked:
        if args.status == "kept":
            print("No kept experiments found.", file=sys.stderr)
        else:
            print("No experiments match the filter.", file=sys.stderr)
        sys.exit(0)

    # Compute deltas vs leader
    ranked = compute_delta(ranked, metric, lower_is_better)

    # Apply --top
    display = ranked[: args.top] if args.top and args.top > 0 else ranked

    # Build row data
    rows = _build_rows(display, metric, lower_is_better)

    # Render
    if args.compact:
        body = format_compact(rows, metric)
    elif args.fmt == "markdown":
        body = format_markdown(rows, metric)
    elif args.fmt == "csv":
        body = format_csv(rows, metric)
    else:
        body = format_text(rows, metric)

    print(body)

    # Footer (skip for CSV and compact — they're meant for machines/scripts)
    if args.fmt not in ("csv",) and not args.compact:
        footer = build_footer(all_experiments, ranked, metric)
        if footer:
            print()
            print(footer)

    # Show seed study status for #1 if available
    if ranked and args.fmt not in ("csv",):
        from scripts.turing_io import load_seed_study
        best_id = ranked[0].get("experiment_id")
        if best_id:
            study = load_seed_study(best_id)
            if study and "mean" in study:
                sensitive = "SEED-SENSITIVE" if study.get("seed_sensitive") else "STABLE"
                print(f"\n  Seed study: {metric}={study['mean']:.4f}±{study.get('std',0):.4f} ({sensitive})")


if __name__ == "__main__":
    main()
