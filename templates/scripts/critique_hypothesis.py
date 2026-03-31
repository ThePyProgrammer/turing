#!/usr/bin/env python3
"""Hypothesis critique engine for the autoresearch pipeline.

Scores a proposed hypothesis on novelty, feasibility, and expected impact
before committing to an expensive training run. A 30-second critique is
cheaper than a 30-minute wasted experiment.

Integrates with the novelty guard (novelty_guard.py) for duplicate
detection and with experiment history for feasibility assessment.

Usage:
    python scripts/critique_hypothesis.py score "increase max_depth to 12" \\
        --log experiments/log.jsonl --config config.yaml
    python scripts/critique_hypothesis.py score "switch to LightGBM" \\
        --log experiments/log.jsonl --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from scripts.turing_io import load_experiments


def score_novelty(
    description: str,
    experiments: list[dict],
    novelty_result: dict | None = None,
) -> dict:
    """Score how novel this hypothesis is relative to experiment history.

    Returns dict with score (0-10), rationale, and similar experiments.
    """
    desc_lower = description.lower()
    desc_words = set(desc_lower.split())

    # Check for near-duplicates in experiment descriptions
    similar = []
    for exp in experiments:
        exp_desc = exp.get("description", "").lower()
        if not exp_desc:
            continue
        exp_words = set(exp_desc.split())
        if desc_words and exp_words:
            overlap = len(desc_words & exp_words) / max(len(desc_words), 1)
            if overlap > 0.5:
                similar.append({
                    "experiment_id": exp.get("experiment_id", "?"),
                    "description": exp.get("description", ""),
                    "status": exp.get("status", "unknown"),
                    "overlap": round(overlap, 2),
                })

    # If novelty guard was run, use its classification
    if novelty_result:
        classification = novelty_result.get("classification", "novel")
        if classification == "duplicate_run":
            return {"score": 0, "rationale": "Exact duplicate of prior experiment",
                    "similar": similar}
        elif classification == "repeat_failure":
            return {"score": 1, "rationale": f"Repeats a known failed approach",
                    "similar": similar}
        elif classification == "incremental_followup":
            return {"score": 5, "rationale": "Incremental follow-up to existing work",
                    "similar": similar}

    # Score based on similarity
    if not similar:
        return {"score": 9, "rationale": "No similar experiments found — appears novel",
                "similar": []}

    # Check if similar experiments failed
    failed_similar = [s for s in similar if s["status"] == "discarded"]
    if failed_similar:
        max_overlap = max(s["overlap"] for s in failed_similar)
        if max_overlap > 0.7:
            return {"score": 2, "rationale": f"Very similar to {len(failed_similar)} discarded experiment(s)",
                    "similar": similar}
        return {"score": 4, "rationale": f"Partially similar to {len(failed_similar)} discarded experiment(s)",
                "similar": similar}

    # Similar to kept experiments = incremental
    kept_similar = [s for s in similar if s["status"] == "kept"]
    if kept_similar:
        return {"score": 6, "rationale": f"Builds on {len(kept_similar)} successful experiment(s)",
                "similar": similar}

    return {"score": 7, "rationale": "Some overlap but distinct approach",
            "similar": similar}


def score_feasibility(
    description: str,
    experiments: list[dict],
    config: dict,
) -> dict:
    """Score how feasible this hypothesis is given current infrastructure.

    Returns dict with score (0-10), rationale, and concerns.
    """
    desc_lower = description.lower()
    concerns = []

    # Check for model types that might not be available
    exotic_models = ["transformer", "bert", "gpt", "diffusion", "gan",
                     "vae", "autoencoder", "resnet", "efficientnet"]
    current_model = config.get("model", {}).get("type", "").lower()

    for model in exotic_models:
        if model in desc_lower and model not in current_model:
            concerns.append(f"Switching to {model} may require significant code changes and new dependencies")

    # Check for mentions of GPU/hardware requirements
    gpu_terms = ["gpu", "cuda", "multi-gpu", "distributed", "tpu"]
    for term in gpu_terms:
        if term in desc_lower:
            concerns.append(f"Mentions '{term}' — verify hardware availability")

    # Check for mentions of external data
    data_terms = ["download", "external data", "pretrained", "pre-trained",
                  "huggingface", "kaggle", "new dataset"]
    for term in data_terms:
        if term in desc_lower:
            concerns.append(f"May require external resources: '{term}'")

    # Score
    if len(concerns) >= 3:
        score = 3
    elif len(concerns) >= 2:
        score = 5
    elif len(concerns) == 1:
        score = 7
    else:
        score = 9

    rationale = "No feasibility concerns" if not concerns else f"{len(concerns)} concern(s) identified"
    return {"score": score, "rationale": rationale, "concerns": concerns}


def score_expected_impact(
    description: str,
    experiments: list[dict],
    config: dict,
) -> dict:
    """Score the expected impact of this hypothesis.

    Based on what types of changes have historically improved metrics.
    Returns dict with score (0-10), rationale, and evidence.
    """
    if not experiments:
        return {"score": 5, "rationale": "No experiment history — cannot estimate impact",
                "evidence": []}

    desc_lower = description.lower()
    evidence = []

    # Analyze which experiment types have been successful
    type_stats: dict[str, dict] = {}
    for exp in experiments:
        exp_type = exp.get("config", {}).get("experiment_type", "unknown")
        if exp_type not in type_stats:
            type_stats[exp_type] = {"kept": 0, "discarded": 0}
        if exp.get("status") == "kept":
            type_stats[exp_type]["kept"] += 1
        else:
            type_stats[exp_type]["discarded"] += 1

    # Infer experiment type from description
    type_keywords = {
        "hyperparameter": ["learning rate", "lr", "max_depth", "depth", "n_estimators",
                          "estimators", "batch", "epochs", "regularization", "dropout"],
        "architecture": ["switch to", "replace", "lightgbm", "catboost", "random forest",
                        "neural", "transformer", "model type"],
        "feature": ["feature", "polynomial", "interaction", "encoding", "one-hot",
                   "embedding", "normalize", "scale"],
        "data": ["augment", "oversample", "undersample", "smote", "split",
                "validation", "cross-validation"],
    }

    inferred_type = "unknown"
    for exp_type, keywords in type_keywords.items():
        if any(kw in desc_lower for kw in keywords):
            inferred_type = exp_type
            break

    # Check historical success rate for this type
    if inferred_type in type_stats:
        stats = type_stats[inferred_type]
        total = stats["kept"] + stats["discarded"]
        if total > 0:
            success_rate = stats["kept"] / total
            evidence.append(f"{inferred_type} experiments: {stats['kept']}/{total} kept ({success_rate:.0%})")
            if success_rate > 0.5:
                return {"score": 8, "rationale": f"{inferred_type} changes have been successful ({success_rate:.0%} keep rate)",
                        "evidence": evidence}
            elif success_rate > 0.2:
                return {"score": 5, "rationale": f"{inferred_type} changes have mixed results ({success_rate:.0%} keep rate)",
                        "evidence": evidence}
            else:
                return {"score": 3, "rationale": f"{inferred_type} changes have mostly failed ({success_rate:.0%} keep rate)",
                        "evidence": evidence}

    # Check if we're in a diminishing returns zone
    kept_experiments = [e for e in experiments if e.get("status") == "kept"]
    if len(kept_experiments) >= 3:
        metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")
        recent_metrics = []
        for exp in kept_experiments[-5:]:
            val = exp.get("metrics", {}).get(metric_name)
            if isinstance(val, (int, float)):
                recent_metrics.append(val)

        if len(recent_metrics) >= 3:
            improvements = [recent_metrics[i] - recent_metrics[i-1]
                          for i in range(1, len(recent_metrics))]
            avg_improvement = sum(improvements) / len(improvements) if improvements else 0
            if abs(avg_improvement) < 0.001:
                evidence.append(f"Recent improvements are very small (avg {avg_improvement:.4f})")
                return {"score": 4, "rationale": "In diminishing returns zone — incremental changes may not help",
                        "evidence": evidence}

    return {"score": 6, "rationale": "Insufficient history to estimate impact precisely",
            "evidence": evidence}


def critique_hypothesis(
    description: str,
    log_path: str = "experiments/log.jsonl",
    config_path: str = "config.yaml",
    novelty_result: dict | None = None,
) -> dict:
    """Run full critique on a hypothesis.

    Returns dict with:
        overall_score: float (0-10)
        verdict: str ("proceed", "modify", "reject")
        novelty: dict
        feasibility: dict
        impact: dict
    """
    experiments = load_experiments(log_path)

    config = {}
    if Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    novelty = score_novelty(description, experiments, novelty_result)
    feasibility = score_feasibility(description, experiments, config)
    impact = score_expected_impact(description, experiments, config)

    # Weighted average: novelty 30%, feasibility 30%, impact 40%
    overall = (novelty["score"] * 0.3 +
               feasibility["score"] * 0.3 +
               impact["score"] * 0.4)
    overall = round(overall, 1)

    if overall >= 6.0:
        verdict = "proceed"
    elif overall >= 4.0:
        verdict = "modify"
    else:
        verdict = "reject"

    return {
        "overall_score": overall,
        "verdict": verdict,
        "novelty": novelty,
        "feasibility": feasibility,
        "impact": impact,
        "description": description,
    }


def format_critique(result: dict) -> str:
    """Format critique result for display."""
    lines = []
    lines.append(f"Hypothesis Critique: {result['description']}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Overall Score: {result['overall_score']}/10 → {result['verdict'].upper()}")
    lines.append("")

    for dim in ("novelty", "feasibility", "impact"):
        d = result[dim]
        lines.append(f"  {dim.title()}: {d['score']}/10 — {d['rationale']}")
        if d.get("concerns"):
            for c in d["concerns"]:
                lines.append(f"    ⚠ {c}")
        if d.get("evidence"):
            for e in d["evidence"]:
                lines.append(f"    → {e}")
        if d.get("similar"):
            for s in d["similar"][:3]:
                lines.append(f"    ~ {s['experiment_id']} ({s['status']}, {s['overlap']:.0%} overlap)")

    lines.append("")
    if result["verdict"] == "reject":
        lines.append("Recommendation: Reject this hypothesis. Consider a different approach.")
    elif result["verdict"] == "modify":
        lines.append("Recommendation: Modify this hypothesis to address the concerns above.")
    else:
        lines.append("Recommendation: Proceed with this hypothesis.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Critique a hypothesis before execution")
    subparsers = parser.add_subparsers(dest="command")

    score_parser = subparsers.add_parser("score", help="Score a hypothesis")
    score_parser.add_argument("description", help="Hypothesis description")
    score_parser.add_argument("--log", default="experiments/log.jsonl")
    score_parser.add_argument("--config", default="config.yaml")
    score_parser.add_argument("--verbose", action="store_true")
    score_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    result = critique_hypothesis(args.description, args.log, args.config)

    if args.json:
        # Remove non-serializable parts for clean JSON output
        output = {
            "overall_score": result["overall_score"],
            "verdict": result["verdict"],
            "novelty_score": result["novelty"]["score"],
            "feasibility_score": result["feasibility"]["score"],
            "impact_score": result["impact"]["score"],
        }
        if args.verbose:
            output["novelty"] = result["novelty"]
            output["feasibility"] = result["feasibility"]
            output["impact"] = result["impact"]
        print(json.dumps(output, indent=2))
    else:
        print(format_critique(result))


if __name__ == "__main__":
    main()
