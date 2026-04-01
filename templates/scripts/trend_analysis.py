#!/usr/bin/env python3
"""Long-term experiment trend analysis for the autoresearch pipeline.

Computes improvement velocity, family ROI, diminishing returns detection,
and phase transition detection across experiment history. Answers "is this
research direction still productive?" and "how many more experiments before
the next meaningful gain?"

Usage:
    python scripts/trend_analysis.py [--config config.yaml] [--log experiments/log.jsonl]
    python scripts/trend_analysis.py --window 10 --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_WINDOW = 5


# --- Improvement Velocity ---


def compute_improvement_trajectory(
    experiments: list[dict],
    metric: str,
    lower_is_better: bool,
) -> list[dict]:
    """Build a trajectory of best-so-far metric values across kept experiments.

    Returns list of dicts with experiment_id, index, timestamp, metric value,
    and best_so_far.
    """
    trajectory = []
    best_val = float("inf") if lower_is_better else float("-inf")
    idx = 0

    for exp in experiments:
        if exp.get("status") != "kept":
            continue
        val = exp.get("metrics", {}).get(metric)
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        improved = (lower_is_better and val < best_val) or (
            not lower_is_better and val > best_val
        )
        if improved:
            best_val = val

        trajectory.append({
            "experiment_id": exp.get("experiment_id", "?"),
            "index": idx,
            "timestamp": exp.get("timestamp", ""),
            "value": val,
            "best_so_far": best_val,
            "improved": improved,
        })
        idx += 1

    return trajectory


def compute_velocity(
    trajectory: list[dict],
    window: int = DEFAULT_WINDOW,
) -> list[dict]:
    """Compute improvement velocity: metric change per experiment over sliding windows.

    Returns list of window summaries with start/end indices, improvement,
    and velocity (improvement / window_size).
    """
    if len(trajectory) < 2:
        return []

    velocities = []
    for i in range(len(trajectory) - window + 1):
        segment = trajectory[i : i + window]
        start_best = segment[0]["best_so_far"]
        end_best = segment[-1]["best_so_far"]
        improvement = end_best - start_best
        velocity = improvement / window

        velocities.append({
            "window_start": segment[0]["experiment_id"],
            "window_end": segment[-1]["experiment_id"],
            "start_index": i,
            "end_index": i + window - 1,
            "start_best": start_best,
            "end_best": end_best,
            "improvement": round(improvement, 6),
            "velocity": round(velocity, 6),
            "improvements_in_window": sum(1 for s in segment if s.get("improved")),
        })

    return velocities


# --- Family ROI ---


def compute_family_roi(
    experiments: list[dict],
    metric: str,
    lower_is_better: bool,
) -> list[dict]:
    """Compute experiments-per-unit-improvement for each experiment family.

    A family with high ROI delivers big metric improvements per experiment.
    A family with low ROI is burning compute for marginal gains.
    """
    families: dict[str, list[dict]] = {}
    for exp in experiments:
        family = exp.get("family") or "untagged"
        families.setdefault(family, []).append(exp)

    results = []
    for family, exps in sorted(families.items()):
        kept = [e for e in exps if e.get("status") == "kept"]
        total = len(exps)
        kept_count = len(kept)

        vals = []
        for e in kept:
            v = e.get("metrics", {}).get(metric)
            if v is not None:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    continue

        if len(vals) < 2:
            improvement = 0.0
        else:
            if lower_is_better:
                improvement = vals[0] - min(vals)
            else:
                improvement = max(vals) - vals[0]

        roi = improvement / total if total > 0 and improvement != 0 else 0.0
        cost_per_gain = total / improvement if improvement > 0 else float("inf")

        results.append({
            "family": family,
            "total_experiments": total,
            "kept": kept_count,
            "keep_rate": round(kept_count / total, 3) if total > 0 else 0,
            "improvement": round(improvement, 6),
            "roi": round(roi, 6),
            "experiments_per_unit_gain": round(cost_per_gain, 2) if cost_per_gain != float("inf") else None,
        })

    results.sort(key=lambda r: r["roi"], reverse=True)
    return results


# --- Diminishing Returns Detection ---


def detect_diminishing_returns(
    trajectory: list[dict],
    lower_is_better: bool,
) -> dict:
    """Fit a log curve to the improvement trajectory and predict effort for next 0.5% gain.

    Uses least-squares fit of best_so_far ~ a * ln(index + 1) + b.
    """
    if len(trajectory) < 3:
        return {"detected": False, "reason": "Too few data points"}

    # Extract (index, best_so_far) pairs
    xs = [t["index"] + 1 for t in trajectory]
    ys = [t["best_so_far"] for t in trajectory]

    # Fit: y = a * ln(x) + b via least squares
    ln_xs = [math.log(x) for x in xs]
    n = len(xs)
    sum_lnx = sum(ln_xs)
    sum_y = sum(ys)
    sum_lnx2 = sum(lx ** 2 for lx in ln_xs)
    sum_lnx_y = sum(lx * y for lx, y in zip(ln_xs, ys))

    denom = n * sum_lnx2 - sum_lnx ** 2
    if abs(denom) < 1e-12:
        return {"detected": False, "reason": "Degenerate fit"}

    a = (n * sum_lnx_y - sum_lnx * sum_y) / denom
    b = (sum_y - a * sum_lnx) / n

    # Compute R-squared
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a * lx + b)) ** 2 for y, lx in zip(ys, ln_xs))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Current best and target
    current_best = ys[-1]
    current_n = xs[-1]

    # Target: 0.5% improvement
    if lower_is_better:
        target = current_best * 0.995  # 0.5% lower
    else:
        target = current_best * 1.005  # 0.5% higher

    # Predict: target = a * ln(n_needed) + b => n_needed = exp((target - b) / a)
    predicted_n = None
    additional_experiments = None
    if abs(a) > 1e-12:
        try:
            predicted_n = int(math.exp((target - b) / a))
            additional_experiments = max(0, predicted_n - current_n)
        except (OverflowError, ValueError):
            pass

    # Diminishing returns if recent velocity is low relative to early velocity
    recent_improvement = abs(ys[-1] - ys[max(0, len(ys) - 4)]) if len(ys) >= 4 else 0
    early_improvement = abs(ys[min(3, len(ys) - 1)] - ys[0]) if len(ys) >= 2 else 0
    ratio = recent_improvement / early_improvement if early_improvement > 0 else 0

    return {
        "detected": ratio < 0.25 and len(trajectory) >= 6,
        "fit": {
            "a": round(a, 6),
            "b": round(b, 6),
            "r_squared": round(r_squared, 4),
        },
        "current_best": round(current_best, 6),
        "target_0_5_pct": round(target, 6),
        "predicted_experiments_for_target": predicted_n,
        "additional_experiments_needed": additional_experiments,
        "recent_vs_early_ratio": round(ratio, 4),
    }


# --- Phase Transition Detection ---


def detect_phase_transitions(
    trajectory: list[dict],
    threshold_factor: float = 3.0,
) -> list[dict]:
    """Detect step-changes (phase transitions) vs incremental improvement.

    A phase transition is a single improvement that exceeds threshold_factor
    times the median improvement magnitude.
    """
    if len(trajectory) < 3:
        return []

    # Compute per-step deltas in best_so_far
    deltas = []
    for i in range(1, len(trajectory)):
        delta = trajectory[i]["best_so_far"] - trajectory[i - 1]["best_so_far"]
        deltas.append({
            "index": i,
            "experiment_id": trajectory[i]["experiment_id"],
            "delta": delta,
            "abs_delta": abs(delta),
        })

    # Filter to only improvements (non-zero deltas in the right direction)
    improvements = [d for d in deltas if d["abs_delta"] > 0]
    if len(improvements) < 2:
        return []

    # Median absolute improvement
    sorted_abs = sorted(d["abs_delta"] for d in improvements)
    mid = len(sorted_abs) // 2
    median_abs = sorted_abs[mid] if len(sorted_abs) % 2 == 1 else (
        sorted_abs[mid - 1] + sorted_abs[mid]
    ) / 2

    if median_abs <= 0:
        return []

    # Find step-changes that exceed threshold * median
    transitions = []
    for d in deltas:
        if d["abs_delta"] > threshold_factor * median_abs:
            transitions.append({
                "experiment_id": d["experiment_id"],
                "index": d["index"],
                "delta": round(d["delta"], 6),
                "magnitude_vs_median": round(d["abs_delta"] / median_abs, 2),
                "type": "step_change",
            })

    return transitions


# --- Report ---


def save_trend_report(report: dict, output_dir: str = "experiments/trends") -> Path:
    """Save trend analysis report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    filepath = out_path / f"trend-{date_str}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_trend_report(report: dict) -> str:
    """Format trend analysis as a markdown report."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    lines = [
        "# Experiment Trend Analysis",
        "",
        f"*Generated {report.get('timestamp', '?')[:19]} UTC*",
        "",
    ]

    # Velocity summary
    velocities = report.get("velocity", [])
    if velocities:
        latest = velocities[-1]
        peak = max(velocities, key=lambda v: abs(v["velocity"]))
        lines.extend([
            "## Improvement Velocity",
            "",
            f"| Window | Velocity | Improvements |",
            f"|--------|----------|--------------|",
        ])
        for v in velocities[-5:]:
            lines.append(
                f"| {v['window_start']}..{v['window_end']} "
                f"| {v['velocity']:+.6f}/exp "
                f"| {v['improvements_in_window']}/{report.get('window', DEFAULT_WINDOW)} |"
            )
        lines.extend([
            "",
            f"**Current velocity:** {latest['velocity']:+.6f}/experiment",
            f"**Peak velocity:** {peak['velocity']:+.6f}/experiment "
            f"(window ending at {peak['window_end']})",
            "",
        ])
    else:
        lines.extend(["## Improvement Velocity", "", "Not enough data.", ""])

    # Family ROI
    family_roi = report.get("family_roi", [])
    if family_roi:
        lines.extend([
            "## Family ROI",
            "",
            "| Family | Experiments | Kept | Improvement | Exp/Unit Gain |",
            "|--------|-------------|------|-------------|---------------|",
        ])
        for f in family_roi:
            cost = f"~{f['experiments_per_unit_gain']:.0f}" if f["experiments_per_unit_gain"] is not None else "inf"
            lines.append(
                f"| {f['family']} | {f['total_experiments']} "
                f"| {f['kept']} | {f['improvement']:+.6f} | {cost} |"
            )
        lines.append("")

        # Flag exhausted families
        exhausted = [f for f in family_roi if f["experiments_per_unit_gain"] is None]
        if exhausted:
            names = ", ".join(f["family"] for f in exhausted)
            lines.append(f"**Zero-improvement families:** {names}")
            lines.append("")

    # Diminishing returns
    dr = report.get("diminishing_returns", {})
    if dr:
        lines.extend(["## Diminishing Returns", ""])
        if dr.get("detected"):
            lines.append("**DETECTED** — recent improvements are <25% of early improvements.")
        else:
            lines.append("Not detected (research is still productive).")

        fit = dr.get("fit", {})
        if fit.get("r_squared") is not None:
            lines.append(f"- Log-curve fit R2: {fit['r_squared']:.4f}")
        lines.append(f"- Current best: {dr.get('current_best', '?')}")
        lines.append(f"- Target (+0.5%): {dr.get('target_0_5_pct', '?')}")

        addl = dr.get("additional_experiments_needed")
        if addl is not None:
            lines.append(f"- Predicted experiments for +0.5%: **~{addl}**")
        else:
            lines.append("- Prediction unavailable (poor fit or degenerate data)")
        lines.append("")

    # Phase transitions
    transitions = report.get("phase_transitions", [])
    if transitions:
        lines.extend([
            "## Phase Transitions (Step-Changes)",
            "",
            "| Experiment | Delta | Magnitude vs Median |",
            "|------------|-------|---------------------|",
        ])
        for t in transitions:
            lines.append(
                f"| {t['experiment_id']} | {t['delta']:+.6f} | {t['magnitude_vs_median']:.1f}x |"
            )
        lines.append("")
    else:
        lines.extend(["## Phase Transitions", "", "No step-changes detected.", ""])

    # Summary
    lines.extend(["---", ""])
    if dr.get("detected"):
        lines.append("*Consider switching research direction or injecting novel hypotheses.*")
    elif velocities and velocities[-1]["velocity"] == 0:
        lines.append("*Velocity is zero — the last window produced no improvement.*")
    else:
        lines.append("*Research is still productive. Continue the current direction.*")

    return "\n".join(lines)


def run_trend_analysis(
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    window: int = DEFAULT_WINDOW,
) -> dict:
    """Run full trend analysis.

    Args:
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        window: Sliding window size for velocity computation.

    Returns:
        Trend analysis result dict.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)
    if not experiments:
        return {"error": "No experiments found", "log_path": log_path}

    trajectory = compute_improvement_trajectory(experiments, metric, lower_is_better)
    if len(trajectory) < 2:
        return {"error": "Need at least 2 kept experiments for trend analysis"}

    velocities = compute_velocity(trajectory, window)
    family_roi = compute_family_roi(experiments, metric, lower_is_better)
    diminishing = detect_diminishing_returns(trajectory, lower_is_better)
    transitions = detect_phase_transitions(trajectory)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
        "lower_is_better": lower_is_better,
        "total_experiments": len(experiments),
        "kept_experiments": len(trajectory),
        "window": window,
        "velocity": velocities,
        "family_roi": family_roi,
        "diminishing_returns": diminishing,
        "phase_transitions": transitions,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Long-term experiment trend analysis")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Sliding window size")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    report = run_trend_analysis(args.config, args.log, args.window)

    if "error" not in report:
        filepath = save_trend_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_trend_report(report))


if __name__ == "__main__":
    main()
