#!/usr/bin/env python3
"""Training curriculum optimization for the autoresearch pipeline.

Orders training data by difficulty and measures whether curriculum
learning improves convergence speed or final performance. Tests
easy-to-hard, hard-to-easy, self-paced, and random strategies.

Usage:
    python scripts/curriculum_optimizer.py exp-042
    python scripts/curriculum_optimizer.py --strategies easy-to-hard,random
    python scripts/curriculum_optimizer.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config

DEFAULT_STRATEGIES = ["random", "easy_to_hard", "hard_to_easy", "self_paced"]
IMPOSSIBLE_THRESHOLD = 0.9  # Samples with difficulty > this across all strategies


# --- Difficulty Scoring ---


def score_difficulty_by_loss(
    losses: np.ndarray,
) -> np.ndarray:
    """Score sample difficulty by loss value (higher loss = harder).

    Normalizes to [0, 1].
    """
    if len(losses) == 0:
        return np.array([])

    min_loss = np.min(losses)
    max_loss = np.max(losses)
    if max_loss == min_loss:
        return np.zeros(len(losses))

    return (losses - min_loss) / (max_loss - min_loss)


def score_difficulty_by_margin(
    margins: np.ndarray,
) -> np.ndarray:
    """Score sample difficulty by margin (smaller margin = harder).

    Margins = distance from decision boundary. Normalizes to [0, 1].
    """
    if len(margins) == 0:
        return np.array([])

    min_m = np.min(margins)
    max_m = np.max(margins)
    if max_m == min_m:
        return np.full(len(margins), 0.5)

    # Invert: small margin = high difficulty
    return 1.0 - (margins - min_m) / (max_m - min_m)


def score_difficulty_by_disagreement(
    multi_seed_predictions: list[np.ndarray],
    labels: np.ndarray,
) -> np.ndarray:
    """Score difficulty by prediction disagreement across seeds.

    Samples where different seeds disagree are "hard" (and possibly mislabeled).
    """
    if not multi_seed_predictions or len(labels) == 0:
        return np.array([])

    n_samples = len(labels)
    n_seeds = len(multi_seed_predictions)

    agreement = np.zeros(n_samples)
    for preds in multi_seed_predictions:
        if len(preds) == n_samples:
            agreement += (preds == labels).astype(float)

    agreement /= n_seeds  # Fraction of seeds that got it right

    # Disagreement = difficulty
    return 1.0 - agreement


# --- Curriculum Strategies ---


def apply_curriculum(
    indices: np.ndarray,
    difficulties: np.ndarray,
    strategy: str,
) -> np.ndarray:
    """Reorder sample indices according to curriculum strategy.

    Args:
        indices: Original sample indices.
        difficulties: Difficulty scores [0, 1] per sample.
        strategy: Curriculum strategy name.

    Returns:
        Reordered indices.
    """
    if len(indices) == 0:
        return indices

    if strategy == "random":
        np.random.shuffle(indices)
        return indices

    elif strategy == "easy_to_hard":
        order = np.argsort(difficulties)
        return indices[order]

    elif strategy == "hard_to_easy":
        order = np.argsort(difficulties)[::-1]
        return indices[order]

    elif strategy == "self_paced":
        # Start with easiest 20%, then gradually include harder
        order = np.argsort(difficulties)
        n = len(order)
        # Shuffle within difficulty bands
        bands = [order[:n // 5], order[n // 5:2 * n // 5],
                 order[2 * n // 5:3 * n // 5], order[3 * n // 5:4 * n // 5],
                 order[4 * n // 5:]]
        result = []
        for band in bands:
            np.random.shuffle(band)
            result.extend(band)
        return np.array(result)

    return indices


def detect_impossible_samples(
    difficulties: np.ndarray,
    threshold: float = IMPOSSIBLE_THRESHOLD,
) -> list[int]:
    """Find samples that are consistently difficult (likely mislabeled).

    Returns list of sample indices.
    """
    return [int(i) for i in range(len(difficulties)) if difficulties[i] > threshold]


# --- Strategy Comparison ---


def compare_strategies(
    strategy_results: dict[str, dict],
    primary_metric: str = "accuracy",
) -> dict:
    """Compare curriculum strategy results.

    Args:
        strategy_results: {strategy_name: {metric_value, convergence_epoch, ...}}

    Returns:
        Comparison report with best strategy and verdict.
    """
    if not strategy_results:
        return {"best_strategy": None, "verdict": "no_data"}

    # Find baseline (random)
    baseline = strategy_results.get("random", {})
    baseline_metric = baseline.get("metric_value", 0)
    baseline_epochs = baseline.get("convergence_epoch")

    results = []
    for name, data in strategy_results.items():
        metric = data.get("metric_value", 0)
        epochs = data.get("convergence_epoch")
        speedup = None
        if epochs and baseline_epochs and baseline_epochs > 0:
            speedup = round(1 - epochs / baseline_epochs, 4)

        results.append({
            "strategy": name,
            "metric_value": round(metric, 6) if metric else None,
            "convergence_epoch": epochs,
            "delta_vs_random": round(metric - baseline_metric, 6) if metric and baseline_metric else None,
            "speedup": speedup,
        })

    # Find best by metric
    with_metric = [r for r in results if r["metric_value"] is not None]
    best = max(with_metric, key=lambda r: r["metric_value"]) if with_metric else None

    verdict = "no_improvement"
    if best and best.get("delta_vs_random") and best["delta_vs_random"] > 0.005:
        verdict = "curriculum_helps"
    elif best and best.get("speedup") and best["speedup"] > 0.1:
        verdict = "faster_convergence"

    return {
        "results": results,
        "best_strategy": best.get("strategy") if best else None,
        "verdict": verdict,
    }


# --- Full Pipeline ---


def curriculum_analysis(
    difficulties: np.ndarray | None = None,
    strategy_results: dict[str, dict] | None = None,
    exp_id: str | None = None,
    config_path: str = "config.yaml",
) -> dict:
    """Run curriculum analysis."""
    config = load_config(config_path)
    primary_metric = config.get("evaluation", {}).get("primary_metric", "accuracy")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": exp_id,
        "primary_metric": primary_metric,
    }

    if difficulties is not None:
        impossible = detect_impossible_samples(difficulties)
        report["difficulty_stats"] = {
            "n_samples": len(difficulties),
            "mean_difficulty": round(float(np.mean(difficulties)), 4),
            "n_impossible": len(impossible),
            "impossible_indices": impossible[:20],
        }

    if strategy_results:
        comparison = compare_strategies(strategy_results, primary_metric)
        report["comparison"] = comparison
    else:
        report["note"] = "Provide strategy_results for comparison. Use /turing:curriculum to run strategies."
        report["available_strategies"] = DEFAULT_STRATEGIES

    return report


# --- Report Formatting ---


def save_curriculum_report(report: dict, output_dir: str = "experiments/curriculum") -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("experiment_id", "unknown")
    filepath = out_path / f"{exp_id}-curriculum.yaml"
    clean = json.loads(json.dumps(report, default=str))
    with open(filepath, "w") as f:
        yaml.dump(clean, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_curriculum_report(report: dict) -> str:
    if "error" in report:
        return f"ERROR: {report['error']}"

    exp_id = report.get("experiment_id", "?")
    metric = report.get("primary_metric", "metric")

    lines = [f"# Curriculum Analysis: {exp_id}", "",
             f"*Generated {report.get('generated_at', 'N/A')[:19]}*", ""]

    # Difficulty stats
    diff_stats = report.get("difficulty_stats")
    if diff_stats:
        lines.extend([
            "## Difficulty Distribution",
            f"- **Samples:** {diff_stats['n_samples']}",
            f"- **Mean difficulty:** {diff_stats['mean_difficulty']:.4f}",
            f"- **Impossible samples:** {diff_stats['n_impossible']} (likely mislabeled)",
            "",
        ])

    # Strategy comparison
    comparison = report.get("comparison")
    if comparison:
        results = comparison.get("results", [])
        if results:
            lines.extend(["## Strategy Comparison", "",
                         f"| Strategy | {metric} | Δ vs Random | Speedup |",
                         "|----------|--------|-------------|---------|"])
            best_name = comparison.get("best_strategy")
            for r in results:
                val = f"{r['metric_value']:.4f}" if r.get("metric_value") is not None else "N/A"
                delta = f"{r['delta_vs_random']:+.4f}" if r.get("delta_vs_random") is not None else "—"
                speedup = f"{r['speedup']:+.0%}" if r.get("speedup") is not None else "—"
                marker = " ← BEST" if r["strategy"] == best_name else ""
                lines.append(f"| {r['strategy']} | {val} | {delta} | {speedup} |{marker}")
            lines.append("")

        verdict_labels = {
            "curriculum_helps": "Curriculum learning improves final performance",
            "faster_convergence": "Curriculum learning converges faster (similar final performance)",
            "no_improvement": "No significant improvement from curriculum ordering",
        }
        verdict = comparison.get("verdict", "?")
        lines.extend(["## Verdict", "", f"**{verdict_labels.get(verdict, verdict.upper())}**"])
    elif report.get("note"):
        lines.append(f"*{report['note']}*")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training curriculum optimization")
    parser.add_argument("exp_id", nargs="?", help="Experiment ID")
    parser.add_argument("--strategies", help="Comma-separated strategies")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = curriculum_analysis(exp_id=args.exp_id, config_path=args.config)

    if "error" not in report:
        filepath = save_curriculum_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_curriculum_report(report))


if __name__ == "__main__":
    main()
