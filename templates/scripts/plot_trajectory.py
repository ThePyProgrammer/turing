#!/usr/bin/env python3
"""Metric trajectory chart generator for the autoresearch pipeline.

Reads experiments/log.jsonl and plots the primary metric over experiment
sequence. Produces publication-ready SVG or PNG charts suitable for papers
and slides, or opens an interactive window for exploratory analysis.

Chart anatomy:
  - Green dots: kept experiments (passed evaluation)
  - Red dots:   discarded experiments (failed evaluation or were dropped)
  - Blue step line: "best so far" running maximum (or minimum, if lower_is_better)
  - Dashed horizontal line: convergence threshold band around best value
  - Star annotation: best experiment ID and value

Usage:
    python scripts/plot_trajectory.py                                   # Interactive
    python scripts/plot_trajectory.py --output trajectory.svg           # SVG for papers
    python scripts/plot_trajectory.py --output trajectory.png --dpi 300 # High-res PNG
    python scripts/plot_trajectory.py --last 20 --no-discarded          # Clean recent view
    python scripts/plot_trajectory.py --metric f1_weighted              # Specific metric

Exit codes:
    0  = success
    1  = error (no experiments, missing metric, bad args)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load relevant settings from config.yaml.

    Returns dict with keys: primary_metric, lower_is_better, patience,
    improvement_threshold, project_name.

    Falls back to safe defaults if config is missing or malformed.
    """
    defaults: dict = {
        "primary_metric": "accuracy",
        "lower_is_better": False,
        "patience": 3,
        "improvement_threshold": 0.005,
        "project_name": "ML Project",
    }

    path = Path(config_path)
    if not path.exists():
        return defaults

    try:
        import yaml  # yaml is already in the autoresearch env

        with open(path) as f:
            config = yaml.safe_load(f) or {}

        eval_cfg = config.get("evaluation", {})
        conv_cfg = config.get("convergence", {})

        # Attempt to derive a human-readable project name from the data source
        # or fall back to the directory name of the config file.
        data_source = config.get("data", {}).get("source", "")
        project_name = (
            Path(data_source).stem.replace("_", " ").title()
            if data_source and not data_source.startswith("{{")
            else Path(config_path).parent.name.replace("_", " ").title()
        )

        return {
            "primary_metric": eval_cfg.get("primary_metric", defaults["primary_metric"]),
            "lower_is_better": eval_cfg.get("lower_is_better", defaults["lower_is_better"]),
            "patience": conv_cfg.get("patience", defaults["patience"]),
            "improvement_threshold": conv_cfg.get(
                "improvement_threshold", defaults["improvement_threshold"]
            ),
            "project_name": project_name,
        }
    except Exception as exc:  # pragma: no cover
        print(f"plot_trajectory: Warning — could not parse config: {exc}", file=sys.stderr)
        return defaults


# ---------------------------------------------------------------------------
# Experiment log loading
# ---------------------------------------------------------------------------


def load_experiments(log_path: str, metric: str) -> list[dict]:
    """Load all experiments from log.jsonl that contain the requested metric.

    Args:
        log_path: Path to experiments/log.jsonl.
        metric:   Metric name to extract.

    Returns:
        List of dicts, each with keys:
          index          int   — 1-based chronological sequence number
          experiment_id  str
          value          float — metric value
          status         str   — "kept" | "discarded" | other
    """
    path = Path(log_path)
    if not path.exists():
        return []

    results = []
    seq = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            value = entry.get("metrics", {}).get(metric)
            if value is None:
                continue  # Skip entries without the requested metric

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            seq += 1
            results.append({
                "index": seq,
                "experiment_id": entry.get("experiment_id", f"exp-{seq:03d}"),
                "value": value,
                "status": entry.get("status", "unknown"),
            })

    return results


# ---------------------------------------------------------------------------
# Best-so-far computation
# ---------------------------------------------------------------------------


def compute_best_so_far(
    experiments: list[dict],
    lower_is_better: bool,
) -> list[float]:
    """Return a list of running-best values aligned with experiments.

    For each position i, best_so_far[i] is the best metric value seen
    among experiments[0..i] (inclusive), considering all statuses.
    """
    best: Optional[float] = None
    result: list[float] = []
    for exp in experiments:
        v = exp["value"]
        if best is None:
            best = v
        else:
            if lower_is_better:
                best = min(best, v)
            else:
                best = max(best, v)
        result.append(best)
    return result


# ---------------------------------------------------------------------------
# Convergence detection (reused logic from check_convergence.py)
# ---------------------------------------------------------------------------


def detect_convergence(
    experiments: list[dict],
    patience: int,
    improvement_threshold: float,
    lower_is_better: bool,
) -> bool:
    """Return True if the last *patience* kept experiments show no meaningful gain."""
    kept = [e for e in experiments if e["status"] == "kept"]
    if len(kept) < patience:
        return False

    for i in range(len(kept) - patience, len(kept)):
        prior_values = [e["value"] for e in kept[:i]]
        if not prior_values:
            return False
        prior_best = min(prior_values) if lower_is_better else max(prior_values)
        current = kept[i]["value"]
        if prior_best == 0:
            improvement = 1.0 if current != 0 else 0.0
        elif lower_is_better:
            improvement = (prior_best - current) / abs(prior_best)
        else:
            improvement = (current - prior_best) / abs(prior_best)

        if improvement >= improvement_threshold:
            return False  # At least one non-trivial improvement in the window

    return True


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def parse_figsize(figsize_str: str) -> tuple[float, float]:
    """Parse a WxH string (e.g. '10x6') into a (width, height) tuple."""
    try:
        w, h = figsize_str.lower().split("x")
        return float(w), float(h)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --figsize '{figsize_str}'. Expected format: WxH (e.g. 10x6)"
        ) from exc


def _best_experiment(
    experiments: list[dict],
    lower_is_better: bool,
) -> dict:
    """Return the experiment with the single best metric value."""
    return (min if lower_is_better else max)(experiments, key=lambda e: e["value"])


def plot_trajectory(
    experiments: list[dict],
    metric: str,
    config: dict,
    args: argparse.Namespace,
) -> None:
    """Build and emit the trajectory chart.

    Args:
        experiments: Filtered experiment list (already sliced by --last).
        metric:      Metric name being plotted.
        config:      Parsed config dict.
        args:        CLI arguments namespace.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Style — use seaborn-v0_8-whitegrid when available, fall back gracefully
    available_styles = plt.style.available
    for candidate in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"):
        if candidate in available_styles or candidate == "default":
            plt.style.use(candidate)
            break

    lower_is_better: bool = config["lower_is_better"]
    patience: int = config["patience"]
    improvement_threshold: float = config["improvement_threshold"]

    # Separate kept vs discarded
    kept = [e for e in experiments if e["status"] == "kept"]
    discarded = [e for e in experiments if e["status"] != "kept"]

    # Best-so-far step line (computed over all experiments)
    best_so_far = compute_best_so_far(experiments, lower_is_better)
    xs_all = [e["index"] for e in experiments]

    # Best experiment overall
    best_exp = _best_experiment(experiments, lower_is_better)
    best_value = best_exp["value"]

    # Convergence threshold band: best ± threshold × |best|
    threshold_delta = improvement_threshold * abs(best_value)
    if lower_is_better:
        threshold_y = best_value + threshold_delta
    else:
        threshold_y = best_value - threshold_delta

    # Convergence detection
    converged = detect_convergence(
        experiments, patience, improvement_threshold, lower_is_better
    )

    # -----------------------------------------------------------------------
    # Figure setup
    # -----------------------------------------------------------------------
    figsize = parse_figsize(args.figsize)
    fig, ax = plt.subplots(figsize=figsize)

    # -----------------------------------------------------------------------
    # Best-so-far step function
    # -----------------------------------------------------------------------
    ax.step(
        xs_all,
        best_so_far,
        where="post",
        color="#2166ac",
        linewidth=1.8,
        alpha=0.85,
        label="Best so far",
        zorder=2,
    )

    # -----------------------------------------------------------------------
    # Convergence threshold dashed line
    # -----------------------------------------------------------------------
    ax.axhline(
        threshold_y,
        color="#762a83",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"Convergence threshold ({improvement_threshold * 100:.1f}% from best)",
        zorder=2,
    )

    # -----------------------------------------------------------------------
    # Scatter: kept experiments (green)
    # -----------------------------------------------------------------------
    if kept:
        ax.scatter(
            [e["index"] for e in kept],
            [e["value"] for e in kept],
            color="#4dac26",
            s=55,
            zorder=4,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.6,
            label=f"Kept ({len(kept)})",
        )

    # -----------------------------------------------------------------------
    # Scatter: discarded experiments (red) — hidden if --no-discarded
    # -----------------------------------------------------------------------
    if discarded and not args.no_discarded:
        ax.scatter(
            [e["index"] for e in discarded],
            [e["value"] for e in discarded],
            color="#d01c8b",
            s=40,
            zorder=3,
            alpha=0.65,
            marker="x",
            linewidths=1.2,
            label=f"Discarded ({len(discarded)})",
        )

    # -----------------------------------------------------------------------
    # Annotate best experiment
    # -----------------------------------------------------------------------
    best_x = best_exp["index"]
    ax.scatter(
        [best_x],
        [best_value],
        color="#d73027",
        s=130,
        zorder=5,
        marker="*",
        edgecolors="#7f0000",
        linewidths=0.6,
    )

    # Choose annotation offset direction: push up for higher-is-better, down for lower
    vert_offset = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0] or 1.0)
    annotation_y = best_value + vert_offset if not lower_is_better else best_value - vert_offset

    ax.annotate(
        f"Best: {best_exp['experiment_id']}\n{best_value:.4f}",
        xy=(best_x, best_value),
        xytext=(best_x + max(1, len(experiments) * 0.04), annotation_y),
        fontsize=8,
        color="#7f0000",
        arrowprops=dict(
            arrowstyle="->",
            color="#7f0000",
            lw=1.0,
        ),
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="#7f0000",
            alpha=0.85,
            linewidth=0.8,
        ),
        zorder=6,
    )

    # -----------------------------------------------------------------------
    # Converged annotation
    # -----------------------------------------------------------------------
    if converged:
        ax.text(
            0.98,
            0.05,
            f"Converged\n(patience={patience})",
            transform=ax.transAxes,
            fontsize=8,
            color="#762a83",
            ha="right",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#f7f4f9",
                edgecolor="#762a83",
                alpha=0.9,
                linewidth=0.8,
            ),
        )

    # -----------------------------------------------------------------------
    # Labels, title, legend
    # -----------------------------------------------------------------------
    direction_hint = "(lower is better)" if lower_is_better else "(higher is better)"
    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel(f"{metric} {direction_hint}", fontsize=11)

    if args.title:
        title = args.title
    else:
        project_name = config.get("project_name", "ML Project")
        title = f"{project_name} — {metric} trajectory"

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Integer x-ticks only
    ax.xaxis.get_major_locator().set_params(integer=True)  # type: ignore[attr-defined]

    ax.legend(fontsize=9, framealpha=0.85, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        ext = out_path.suffix.lower()
        if ext == ".png":
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        elif ext == ".svg":
            fig.savefig(out_path, format="svg", bbox_inches="tight")
        else:
            print(
                f"plot_trajectory: Warning — unknown extension '{ext}', saving as PNG.",
                file=sys.stderr,
            )
            fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"plot_trajectory: Saved to {out_path}")
    else:
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot primary metric trajectory from experiments/log.jsonl.\n"
            "Output can be SVG (default, best for papers), PNG, or interactive."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/plot_trajectory.py\n"
            "  python scripts/plot_trajectory.py --output trajectory.svg\n"
            "  python scripts/plot_trajectory.py --output trajectory.png --dpi 300\n"
            "  python scripts/plot_trajectory.py --last 20 --no-discarded\n"
            "  python scripts/plot_trajectory.py --metric f1_weighted\n"
        ),
    )
    parser.add_argument(
        "--log",
        default="experiments/log.jsonl",
        metavar="PATH",
        help="Path to experiment log (default: experiments/log.jsonl)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output file path (.svg or .png). Omit for interactive display.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        metavar="NAME",
        help="Metric to plot (default: primary_metric from config.yaml)",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="Only plot the last N experiments",
    )
    parser.add_argument(
        "--no-discarded",
        action="store_true",
        help="Hide discarded experiments from the chart",
    )
    parser.add_argument(
        "--title",
        default=None,
        metavar="TEXT",
        help="Override the auto-generated chart title",
    )
    parser.add_argument(
        "--figsize",
        default="10x6",
        metavar="WxH",
        help="Figure size in inches, width x height (default: 10x6)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for PNG output in dots per inch (default: 150)",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    # Guard: matplotlib must be importable
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "plot_trajectory: Error — matplotlib is not installed.\n"
            "  Install it with:  pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = build_parser()
    args = parser.parse_args()

    # Validate --figsize early so we get a clean error message
    try:
        parse_figsize(args.figsize)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    # Load config
    config = load_config(args.config)

    # Resolve metric name
    metric = args.metric if args.metric else config["primary_metric"]

    # Load experiments
    experiments = load_experiments(args.log, metric)

    if not experiments:
        log_path = Path(args.log)
        if not log_path.exists():
            print(
                f"plot_trajectory: Error — log not found at '{args.log}'.\n"
                "  Run at least one experiment first.",
                file=sys.stderr,
            )
        else:
            print(
                f"plot_trajectory: Error — no experiments with metric '{metric}' "
                f"found in '{args.log}'.\n"
                "  Check the metric name or run experiments first.",
                file=sys.stderr,
            )
        sys.exit(1)

    # Apply --last filter
    if args.last is not None:
        if args.last < 1:
            parser.error("--last must be a positive integer")
        experiments = experiments[-args.last :]

    # Re-index so x-axis is contiguous after slicing
    if args.last is not None:
        for i, exp in enumerate(experiments, start=1):
            exp["index"] = i

    n_kept = sum(1 for e in experiments if e["status"] == "kept")
    n_disc = sum(1 for e in experiments if e["status"] != "kept")
    print(
        f"plot_trajectory: Plotting {len(experiments)} experiments "
        f"({n_kept} kept, {n_disc} discarded) for metric '{metric}'",
        file=sys.stderr,
    )

    plot_trajectory(experiments, metric, config, args)


if __name__ == "__main__":
    main()
