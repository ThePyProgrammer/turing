#!/usr/bin/env python3
"""Automated failure postmortem for the autoresearch pipeline.

When experiments stop improving, diagnoses the root cause: search space
exhaustion, systematic config error, data issue, metric ceiling, or
noise floor. Produces actionable next steps.

Usage:
    python scripts/failure_postmortem.py
    python scripts/failure_postmortem.py --window 10
    python scripts/failure_postmortem.py --auto-trigger 5
    python scripts/failure_postmortem.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_WINDOW = 10
DEFAULT_AUTO_TRIGGER = 5

DIAGNOSIS_TYPES = [
    "search_space_exhaustion",
    "systematic_config_error",
    "data_issue",
    "metric_ceiling",
    "noise_floor",
]


# --- Streak Detection ---


def detect_failure_streak(
    experiments: list[dict],
    primary_metric: str,
    lower_is_better: bool = False,
) -> dict:
    """Detect how many consecutive experiments failed to improve.

    Returns:
        Streak info with count, best metric, streak experiments.
    """
    if not experiments:
        return {"streak_length": 0, "best_metric": None, "streak_experiments": []}

    # Find the best metric value
    best_val = None
    for exp in experiments:
        val = exp.get("metrics", {}).get(primary_metric)
        if val is None:
            continue
        if best_val is None:
            best_val = val
        elif (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val = val

    if best_val is None:
        return {"streak_length": len(experiments), "best_metric": None, "streak_experiments": experiments}

    # Count consecutive non-improvements from the end
    streak = []
    best_so_far = None

    for exp in experiments:
        val = exp.get("metrics", {}).get(primary_metric)
        if val is None:
            continue
        if best_so_far is None:
            best_so_far = val
        elif (lower_is_better and val < best_so_far) or (not lower_is_better and val > best_so_far):
            best_so_far = val
            streak = []  # Reset streak on improvement

        streak.append(exp)

    # The streak is from last improvement to end
    # Remove the improving experiment itself if it's the first
    if streak and streak[0].get("metrics", {}).get(primary_metric) == best_so_far:
        streak = streak[1:]

    return {
        "streak_length": len(streak),
        "best_metric": best_val,
        "streak_experiments": streak,
    }


# --- Diagnosis Functions ---


def diagnose_search_space_exhaustion(
    streak_experiments: list[dict],
    primary_metric: str,
) -> dict:
    """Check if experiments cluster in a small config region."""
    if len(streak_experiments) < 3:
        return {"score": 0, "evidence": "Too few experiments for diagnosis"}

    # Extract hyperparameters from streak
    all_params = {}
    for exp in streak_experiments:
        config = exp.get("config", {})
        hyperparams = config.get("hyperparams", {})
        for k, v in hyperparams.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                all_params.setdefault(k, []).append(float(v))

    if not all_params:
        return {"score": 0, "evidence": "No numeric hyperparameters found"}

    # Measure coefficient of variation for each param
    low_variance_params = []
    for param, values in all_params.items():
        if len(values) < 2:
            continue
        mean = np.mean(values)
        if abs(mean) < 1e-10:
            continue
        cv = np.std(values) / abs(mean)
        if cv < 0.15:  # Less than 15% variation
            low_variance_params.append({"param": param, "cv": round(float(cv), 4), "mean": round(float(mean), 4)})

    # Check family diversity
    families = set()
    for exp in streak_experiments:
        family = exp.get("family", exp.get("config", {}).get("family", "unknown"))
        families.add(family)

    score = 0
    evidence = []

    if len(low_variance_params) > len(all_params) * 0.5:
        score += 0.4
        evidence.append(f"Config variance LOW: {len(low_variance_params)}/{len(all_params)} params within ±15%")

    if len(families) <= 1:
        score += 0.3
        evidence.append(f"All experiments in same family: {families}")

    # Check if metrics are clustered
    metrics = [exp.get("metrics", {}).get(primary_metric) for exp in streak_experiments
               if exp.get("metrics", {}).get(primary_metric) is not None]
    if len(metrics) >= 2:
        metric_cv = np.std(metrics) / abs(np.mean(metrics)) if abs(np.mean(metrics)) > 0 else 0
        if metric_cv < 0.02:
            score += 0.3
            evidence.append(f"Metric range very tight (CV={metric_cv:.4f})")

    return {
        "score": round(score, 2),
        "evidence": evidence if evidence else ["No strong evidence of exhaustion"],
        "low_variance_params": low_variance_params,
        "families": list(families),
    }


def diagnose_systematic_config_error(
    streak_experiments: list[dict],
    primary_metric: str,
    best_metric: float | None,
) -> dict:
    """Check if all experiments share a common bad config."""
    if len(streak_experiments) < 3:
        return {"score": 0, "evidence": "Too few experiments"}

    # Find params that are identical across all streak experiments
    common_params = {}
    first_config = streak_experiments[0].get("config", {}).get("hyperparams", {})

    for k, v in first_config.items():
        if not isinstance(v, (int, float, str)):
            continue
        all_same = all(
            exp.get("config", {}).get("hyperparams", {}).get(k) == v
            for exp in streak_experiments[1:]
        )
        if all_same:
            common_params[k] = v

    score = 0
    evidence = []

    if common_params:
        ratio = len(common_params) / max(len(first_config), 1)
        if ratio > 0.5:
            score += 0.5
            evidence.append(f"{len(common_params)} params unchanged across all {len(streak_experiments)} experiments")
            evidence.append(f"Common: {common_params}")

    # Check if all experiments are significantly worse than best
    if best_metric is not None:
        streak_metrics = [exp.get("metrics", {}).get(primary_metric) for exp in streak_experiments
                          if exp.get("metrics", {}).get(primary_metric) is not None]
        if streak_metrics:
            avg_gap = abs(np.mean(streak_metrics) - best_metric)
            if avg_gap > 0.02:
                score += 0.3
                evidence.append(f"Average gap from best: {avg_gap:.4f}")

    return {"score": round(score, 2), "evidence": evidence or ["No common config error detected"], "common_params": common_params}


def diagnose_data_issue(
    streak_experiments: list[dict],
    primary_metric: str,
) -> dict:
    """Check if all models fail similarly regardless of type."""
    if len(streak_experiments) < 3:
        return {"score": 0, "evidence": "Too few experiments"}

    # Check model type diversity
    model_types = set()
    for exp in streak_experiments:
        mt = exp.get("config", {}).get("model_type", "unknown")
        model_types.add(mt)

    score = 0
    evidence = []

    # If multiple model types all fail similarly → data issue
    if len(model_types) >= 2:
        metrics = [exp.get("metrics", {}).get(primary_metric) for exp in streak_experiments
                   if exp.get("metrics", {}).get(primary_metric) is not None]
        if len(metrics) >= 2:
            cv = np.std(metrics) / abs(np.mean(metrics)) if abs(np.mean(metrics)) > 0 else 0
            if cv < 0.03:
                score += 0.6
                evidence.append(f"{len(model_types)} different model types all perform similarly (CV={cv:.4f})")
                evidence.append(f"Model types: {model_types}")

    return {"score": round(score, 2), "evidence": evidence or ["No data issue pattern detected"], "model_types": list(model_types)}


def diagnose_metric_ceiling(
    streak_experiments: list[dict],
    primary_metric: str,
    best_metric: float | None,
) -> dict:
    """Check if metrics are plateauing near a theoretical limit."""
    if best_metric is None:
        return {"score": 0, "evidence": "No best metric available"}

    score = 0
    evidence = []

    # Check if best metric is very high (suggesting ceiling)
    if best_metric > 0.95:
        score += 0.4
        evidence.append(f"Current best {primary_metric}={best_metric:.4f} — near theoretical maximum")

    # Check improvement rate (are improvements getting tiny?)
    metrics = sorted([
        exp.get("metrics", {}).get(primary_metric)
        for exp in streak_experiments
        if exp.get("metrics", {}).get(primary_metric) is not None
    ])
    if len(metrics) >= 3:
        range_val = max(metrics) - min(metrics)
        if range_val < 0.005:
            score += 0.3
            evidence.append(f"Metric range in streak: {range_val:.4f} (< 0.005)")

    return {"score": round(score, 2), "evidence": evidence or ["No ceiling pattern detected"]}


def diagnose_noise_floor(
    streak_experiments: list[dict],
    primary_metric: str,
    seed_dir: str = "experiments/seed_studies",
) -> dict:
    """Check if improvements are within seed variance."""
    score = 0
    evidence = []

    # Check seed study data for variance estimate
    seed_path = Path(seed_dir)
    seed_variance = None
    if seed_path.exists():
        for f in sorted(seed_path.glob("*.yaml")):
            try:
                with open(f) as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, dict) and "std" in data:
                    seed_variance = data["std"]
            except (yaml.YAMLError, OSError):
                continue

    metrics = [exp.get("metrics", {}).get(primary_metric) for exp in streak_experiments
               if exp.get("metrics", {}).get(primary_metric) is not None]

    if len(metrics) >= 2:
        streak_range = max(metrics) - min(metrics)
        if seed_variance is not None:
            if streak_range < seed_variance * 2:
                score += 0.7
                evidence.append(f"Streak range ({streak_range:.4f}) < 2x seed std ({seed_variance:.4f})")
        else:
            streak_std = float(np.std(metrics))
            if streak_std < 0.005:
                score += 0.3
                evidence.append(f"Streak std ({streak_std:.4f}) very low — may be noise")

    return {"score": round(score, 2), "evidence": evidence or ["No noise floor pattern detected"], "seed_variance": seed_variance}


# --- Main Pipeline ---


def run_postmortem(
    window: int = DEFAULT_WINDOW,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    seed_dir: str = "experiments/seed_studies",
) -> dict:
    """Run failure postmortem analysis.

    Args:
        window: Number of recent experiments to analyze.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        seed_dir: Path to seed study directory.

    Returns:
        Postmortem report with diagnosis, evidence, and recommendations.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(log_path)

    if not experiments:
        return {"error": "No experiments found"}

    # Use last N experiments
    recent = experiments[-window:]

    streak_info = detect_failure_streak(recent, primary_metric, lower_is_better)
    streak_exps = streak_info["streak_experiments"]
    best_metric = streak_info["best_metric"]
    streak_len = streak_info["streak_length"]

    if streak_len < 2:
        return {
            "streak_length": streak_len,
            "message": "No significant failure streak detected",
            "best_metric": best_metric,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # Run all diagnoses
    diagnoses = {
        "search_space_exhaustion": diagnose_search_space_exhaustion(streak_exps, primary_metric),
        "systematic_config_error": diagnose_systematic_config_error(streak_exps, primary_metric, best_metric),
        "data_issue": diagnose_data_issue(streak_exps, primary_metric),
        "metric_ceiling": diagnose_metric_ceiling(streak_exps, primary_metric, best_metric),
        "noise_floor": diagnose_noise_floor(streak_exps, primary_metric, seed_dir),
    }

    # Pick the highest-scoring diagnosis
    primary_diagnosis = max(diagnoses.items(), key=lambda d: d[1]["score"])
    diagnosis_name = primary_diagnosis[0]
    diagnosis_data = primary_diagnosis[1]

    # Generate recommendations
    recommendations = _generate_recommendations(diagnosis_name, diagnosis_data, streak_len)

    return {
        "streak_length": streak_len,
        "window": window,
        "best_metric": best_metric,
        "primary_metric": primary_metric,
        "primary_diagnosis": diagnosis_name,
        "diagnosis_score": diagnosis_data["score"],
        "diagnosis_evidence": diagnosis_data["evidence"],
        "all_diagnoses": {k: {"score": v["score"]} for k, v in diagnoses.items()},
        "recommendations": recommendations,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _generate_recommendations(diagnosis: str, data: dict, streak_len: int) -> list[str]:
    """Generate actionable recommendations based on diagnosis."""
    recs = {
        "search_space_exhaustion": [
            "Stop tuning hyperparameters — switch to `/turing:feature` for feature engineering",
            "Try `/turing:ensemble` — combine existing models instead of building new ones",
            "Run `/turing:scale --axis data` — check if more data would help",
        ],
        "systematic_config_error": [
            "Run `/turing:sensitivity` — identify which params actually matter",
            "Check the common config values against sensitivity analysis",
            "Try resetting to the best experiment's config and vary one param at a time",
        ],
        "data_issue": [
            "Run `/turing:leak` — check for data leakage masking real performance",
            "Run `/turing:sanity` — verify data pipeline integrity",
            "Inspect the raw data for quality issues or distribution shift",
        ],
        "metric_ceiling": [
            "Run `/turing:scale` to confirm you've hit the ceiling",
            "Consider shifting to a different metric or task formulation",
            "Try ensemble methods for marginal gains: `/turing:ensemble`",
        ],
        "noise_floor": [
            "Run `/turing:seed` with more seeds to measure true variance",
            "Increase n_runs for each experiment to reduce noise",
            "Consider whether the current metric resolution is sufficient",
        ],
    }
    return recs.get(diagnosis, [f"Investigate the last {streak_len} experiments manually"])


# --- Report Formatting ---


def save_postmortem_report(report: dict, output_dir: str = "experiments/postmortems") -> Path:
    """Save postmortem report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filepath = out_path / f"postmortem-{ts}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_postmortem_report(report: dict) -> str:
    """Format postmortem report as readable markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    if "message" in report:
        return f"No failure streak: {report['message']} (best {report.get('best_metric', 'N/A')})"

    lines = [
        f"# Failure Postmortem (last {report.get('streak_length', '?')} experiments, 0 improvements)",
        "",
        f"**Diagnosis:** {report.get('primary_diagnosis', 'unknown').upper().replace('_', ' ')}",
        f"**Confidence:** {report.get('diagnosis_score', 0):.0%}",
        "",
        "## Evidence",
        "",
    ]

    for e in report.get("diagnosis_evidence", []):
        lines.append(f"- {e}")

    lines.extend(["", "## All Diagnoses", ""])
    for name, data in report.get("all_diagnoses", {}).items():
        score = data.get("score", 0)
        marker = "◀" if name == report.get("primary_diagnosis") else ""
        lines.append(f"- {name.replace('_', ' ')}: {score:.0%} {marker}")

    lines.extend(["", "## Recommended Actions", ""])
    for i, rec in enumerate(report.get("recommendations", []), 1):
        lines.append(f"{i}. {rec}")

    lines.extend(["", f"*Generated: {report.get('generated_at', 'N/A')}*"])
    return "\n".join(lines)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Failure postmortem — diagnose why experiments stopped improving"
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="Number of recent experiments to analyze")
    parser.add_argument("--auto-trigger", type=int, default=DEFAULT_AUTO_TRIGGER,
                        help="Minimum streak length to trigger postmortem")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    report = run_postmortem(
        window=args.window,
        config_path=args.config,
        log_path=args.log,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_postmortem_report(report))

    if "error" not in report and "message" not in report:
        saved = save_postmortem_report(report)
        if not args.json:
            print(f"\nSaved: {saved}")


if __name__ == "__main__":
    main()
