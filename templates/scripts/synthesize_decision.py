#!/usr/bin/env python3
"""Decision packet synthesis for the autoresearch pipeline.

After each experiment, synthesizes a structured verdict combining:
- Run outcome and metrics
- Comparison to current champion
- A recommended next action

Actions: promote, branch_followup, replicate, abandon, fix_and_retry,
investigate_crash.

Inspired by pauldebdeep9/autoresearch MemoryLab decision packets.

Usage:
    python scripts/synthesize_decision.py \\
        --experiment exp-005 \\
        --log experiments/log.jsonl \\
        --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROMISING_DELTA = 0.005  # 0.5% relative improvement = promising


def load_experiment(log_path: str, experiment_id: str) -> dict | None:
    """Load a specific experiment from the JSONL log."""
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


def find_champion(log_path: str, metric: str, lower_is_better: bool) -> dict | None:
    """Find the current best (champion) experiment."""
    path = Path(log_path)
    if not path.exists():
        return None
    best = None
    best_val = float("inf") if lower_is_better else float("-inf")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("status") != "kept":
                    continue
                val = entry.get("metrics", {}).get(metric)
                if val is None:
                    continue
                if (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
                    best_val = val
                    best = entry
            except json.JSONDecodeError:
                continue
    return best


def compute_delta(current: float, champion: float, lower_is_better: bool) -> float:
    """Compute relative improvement over champion."""
    if champion == 0:
        return 1.0 if current != 0 else 0.0
    if lower_is_better:
        return (champion - current) / abs(champion)
    else:
        return (current - champion) / abs(champion)


def classify_outcome(
    experiment: dict,
    champion: dict | None,
    metric: str,
    lower_is_better: bool,
) -> tuple[str, float | None]:
    """Classify the experiment outcome.

    Returns (outcome, delta_to_champion).
    Outcomes: new_champion, marginal_improvement, lateral, regression, crash.
    """
    status = experiment.get("status", "")
    if status == "crash":
        return "crash", None

    current_val = experiment.get("metrics", {}).get(metric)
    if current_val is None:
        return "crash", None

    if champion is None:
        if status == "kept":
            return "new_champion", None
        return "regression", None

    champion_val = champion.get("metrics", {}).get(metric)
    if champion_val is None:
        return "new_champion" if status == "kept" else "regression", None

    delta = compute_delta(current_val, champion_val, lower_is_better)

    if status == "kept" and delta > PROMISING_DELTA:
        return "new_champion", delta
    elif status == "kept" and delta > 0:
        return "marginal_improvement", delta
    elif abs(delta) <= PROMISING_DELTA:
        return "lateral", delta
    else:
        return "regression", delta


def recommend_action(
    outcome: str,
    experiment: dict,
) -> tuple[str, str]:
    """Recommend a next action based on the outcome.

    Returns (action, rationale).
    """
    actions = {
        "new_champion": (
            "promote",
            "New best result — update champion, consider replicating to confirm stability",
        ),
        "marginal_improvement": (
            "branch_followup",
            "Slight improvement — explore variations of this approach",
        ),
        "lateral": (
            "abandon",
            "No meaningful change — this direction is not productive",
        ),
        "regression": (
            "abandon",
            "Performance decreased — discard and try a different approach",
        ),
        "crash": (
            "fix_and_retry" if experiment.get("description", "") else "investigate_crash",
            "Experiment crashed — check error logs and retry with fixes" if experiment.get("description") else "Experiment crashed — investigate the cause before retrying",
        ),
    }
    return actions.get(outcome, ("investigate_crash", "Unknown outcome"))


def synthesize_packet(
    experiment: dict,
    champion: dict | None,
    metric: str,
    lower_is_better: bool,
) -> dict:
    """Synthesize a complete decision packet for an experiment.

    Returns dict with: experiment_id, outcome, action, rationale,
    delta, current_metric, champion_metric, champion_id.
    """
    outcome, delta = classify_outcome(experiment, champion, metric, lower_is_better)
    action, rationale = recommend_action(outcome, experiment)

    current_val = experiment.get("metrics", {}).get(metric)
    champion_val = champion.get("metrics", {}).get(metric) if champion else None

    packet = {
        "experiment_id": experiment.get("experiment_id", "?"),
        "outcome": outcome,
        "action": action,
        "rationale": rationale,
        "delta": round(delta, 6) if delta is not None else None,
        "current_metric": current_val,
        "champion_metric": champion_val,
        "champion_id": champion.get("experiment_id", "?") if champion else None,
        "status": experiment.get("status", "?"),
        "description": experiment.get("description", ""),
        "family": experiment.get("family"),
        "hypothesis_id": experiment.get("hypothesis_id"),
    }

    return packet


def format_packet(packet: dict, metric_name: str) -> str:
    """Format a decision packet for display."""
    lines = [
        f"Decision Packet: {packet['experiment_id']}",
        f"  Outcome: {packet['outcome']}",
        f"  Action: {packet['action']}",
        f"  Rationale: {packet['rationale']}",
        f"  {metric_name}: {packet['current_metric']} (champion: {packet['champion_metric']})",
    ]
    if packet["delta"] is not None:
        lines.append(f"  Delta: {packet['delta']:+.4f} relative")
    if packet["family"]:
        lines.append(f"  Family: {packet['family']}")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Synthesize decision packet")
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("--log", default="experiments/log.jsonl")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    eval_cfg = config.get("evaluation", {})
    metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiment = load_experiment(args.log, args.experiment)
    if not experiment:
        print(f"Experiment {args.experiment} not found.", file=sys.stderr)
        sys.exit(1)

    champion = find_champion(args.log, metric, lower_is_better)
    # Don't compare champion to itself
    if champion and champion.get("experiment_id") == experiment.get("experiment_id"):
        champion = None

    packet = synthesize_packet(experiment, champion, metric, lower_is_better)

    if args.json:
        print(json.dumps(packet, indent=2))
    else:
        print(format_packet(packet, metric))


if __name__ == "__main__":
    main()
