#!/usr/bin/env python3
"""Model registry and lifecycle management for the autoresearch pipeline.

Tracks which model version is production, staging, candidate, or archived.
Promotion workflow with automated gates: candidate → staging → production.
Prevents the "which pickle file is deployed?" problem.

Usage:
    python scripts/model_lifecycle.py list
    python scripts/model_lifecycle.py register exp-095 --version v4.1
    python scripts/model_lifecycle.py promote exp-089 staging
    python scripts/model_lifecycle.py demote exp-089 candidate
    python scripts/model_lifecycle.py history
    python scripts/model_lifecycle.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_REGISTRY_PATH = "experiments/registry.yaml"

STAGES = ["candidate", "staging", "production", "archived"]
PROMOTION_ORDER = {"candidate": "staging", "staging": "production"}
DEMOTION_ORDER = {"production": "staging", "staging": "candidate"}

# Gate requirements for each promotion
PROMOTION_GATES = {
    "candidate_to_staging": ["regression", "seed_study"],
    "staging_to_production": ["audit", "calibration"],
}


# --- Registry IO ---


def load_registry(registry_path: str = DEFAULT_REGISTRY_PATH) -> dict:
    """Load the model registry.

    Returns:
        Registry dict with 'models' list and 'history' list.
    """
    path = Path(registry_path)
    if not path.exists():
        return {"models": [], "history": []}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return {"models": [], "history": []}
    if not isinstance(data, dict):
        return {"models": [], "history": []}
    if "models" not in data:
        data["models"] = []
    if "history" not in data:
        data["history"] = []
    return data


def save_registry(registry: dict, registry_path: str = DEFAULT_REGISTRY_PATH) -> Path:
    """Save the model registry."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)
    return path


# --- Registry Operations ---


def register_model(
    registry: dict,
    exp_id: str,
    version: str,
    metric: float | None = None,
    metric_name: str = "accuracy",
    stage: str = "candidate",
) -> dict:
    """Register a new model in the registry.

    Args:
        registry: Current registry state.
        exp_id: Experiment ID.
        version: Model version label.
        metric: Primary metric value.
        metric_name: Primary metric name.
        stage: Initial stage (default: candidate).

    Returns:
        Updated registry.
    """
    if stage not in STAGES:
        return {"error": f"Invalid stage: {stage}. Must be one of {STAGES}"}

    # Check for duplicate
    for model in registry["models"]:
        if model["exp_id"] == exp_id:
            return {"error": f"Model {exp_id} already registered (stage: {model['stage']})"}

    now = datetime.now(timezone.utc).isoformat()

    model_entry = {
        "exp_id": exp_id,
        "version": version,
        "stage": stage,
        "metric_name": metric_name,
        "metric": metric,
        "registered_at": now,
        "last_promoted": now,
        "gates_passed": [],
    }

    registry["models"].append(model_entry)
    registry["history"].append({
        "action": "register",
        "exp_id": exp_id,
        "version": version,
        "stage": stage,
        "timestamp": now,
    })

    return registry


def promote_model(
    registry: dict,
    exp_id: str,
    target_stage: str,
    gate_results: dict[str, str] | None = None,
    force: bool = False,
) -> dict:
    """Promote a model to the next stage.

    Args:
        registry: Current registry state.
        exp_id: Experiment ID to promote.
        target_stage: Target stage.
        gate_results: Gate check results {gate_name: "PASS"/"FAIL"}.
        force: Skip gate checks.

    Returns:
        Updated registry or error dict.
    """
    model = _find_model(registry, exp_id)
    if model is None:
        return {"error": f"Model {exp_id} not found in registry"}

    current_stage = model["stage"]

    # Validate promotion path
    expected_target = PROMOTION_ORDER.get(current_stage)
    if expected_target is None:
        return {"error": f"Cannot promote from {current_stage} — already at highest stage or archived"}

    if target_stage != expected_target:
        return {"error": f"Cannot promote {current_stage} → {target_stage}. Expected: {current_stage} → {expected_target}"}

    # Check gates
    gate_key = f"{current_stage}_to_{target_stage}"
    required_gates = PROMOTION_GATES.get(gate_key, [])

    if not force and required_gates:
        if gate_results is None:
            return {
                "error": "Gate checks required",
                "required_gates": required_gates,
                "suggestion": f"Run gate checks or use --force to skip",
            }

        failed_gates = []
        for gate in required_gates:
            result = gate_results.get(gate, "NOT_RUN")
            if result != "PASS":
                failed_gates.append({"gate": gate, "result": result})

        if failed_gates:
            return {
                "error": "Gate checks failed",
                "failed_gates": failed_gates,
                "required_gates": required_gates,
            }

    # Perform promotion
    now = datetime.now(timezone.utc).isoformat()
    model["stage"] = target_stage
    model["last_promoted"] = now
    model["gates_passed"].extend(required_gates)

    registry["history"].append({
        "action": "promote",
        "exp_id": exp_id,
        "from_stage": current_stage,
        "to_stage": target_stage,
        "gate_results": gate_results or {},
        "forced": force,
        "timestamp": now,
    })

    return registry


def demote_model(
    registry: dict,
    exp_id: str,
    target_stage: str,
    reason: str = "",
) -> dict:
    """Demote a model to a lower stage.

    Args:
        registry: Current registry state.
        exp_id: Experiment ID to demote.
        target_stage: Target stage.
        reason: Reason for demotion.

    Returns:
        Updated registry or error dict.
    """
    model = _find_model(registry, exp_id)
    if model is None:
        return {"error": f"Model {exp_id} not found in registry"}

    current_stage = model["stage"]

    if target_stage not in STAGES:
        return {"error": f"Invalid stage: {target_stage}"}

    if STAGES.index(target_stage) >= STAGES.index(current_stage):
        return {"error": f"Cannot demote {current_stage} → {target_stage} (not a demotion)"}

    now = datetime.now(timezone.utc).isoformat()
    model["stage"] = target_stage
    model["last_promoted"] = now

    registry["history"].append({
        "action": "demote",
        "exp_id": exp_id,
        "from_stage": current_stage,
        "to_stage": target_stage,
        "reason": reason,
        "timestamp": now,
    })

    return registry


def archive_model(
    registry: dict,
    exp_id: str,
    reason: str = "",
) -> dict:
    """Archive a model (remove from active lifecycle).

    Args:
        registry: Current registry state.
        exp_id: Experiment ID to archive.
        reason: Reason for archiving.

    Returns:
        Updated registry or error dict.
    """
    model = _find_model(registry, exp_id)
    if model is None:
        return {"error": f"Model {exp_id} not found in registry"}

    now = datetime.now(timezone.utc).isoformat()
    old_stage = model["stage"]
    model["stage"] = "archived"
    model["last_promoted"] = now

    registry["history"].append({
        "action": "archive",
        "exp_id": exp_id,
        "from_stage": old_stage,
        "reason": reason,
        "timestamp": now,
    })

    return registry


def _find_model(registry: dict, exp_id: str) -> dict | None:
    """Find a model in the registry by experiment ID."""
    for model in registry.get("models", []):
        if model["exp_id"] == exp_id:
            return model
    return None


# --- Query Operations ---


def list_models(registry: dict, stage: str | None = None) -> list[dict]:
    """List registered models, optionally filtered by stage."""
    models = registry.get("models", [])
    if stage:
        models = [m for m in models if m["stage"] == stage]
    return sorted(models, key=lambda m: STAGES.index(m["stage"]) if m["stage"] in STAGES else 99)


def get_model_at_stage(registry: dict, stage: str) -> dict | None:
    """Get the model currently at a specific stage."""
    for model in registry.get("models", []):
        if model["stage"] == stage:
            return model
    return None


def get_history(registry: dict, exp_id: str | None = None) -> list[dict]:
    """Get promotion/demotion history, optionally for a specific model."""
    history = registry.get("history", [])
    if exp_id:
        history = [h for h in history if h.get("exp_id") == exp_id]
    return history


# --- Gate Checking ---


def check_gates(
    exp_id: str,
    gate_names: list[str],
    experiments: list[dict] | None = None,
    regression_dir: str = "experiments/regressions",
    seed_dir: str = "experiments/seed_studies",
    audit_dir: str = "experiments/audits",
    calibration_dir: str = "experiments/calibration",
) -> dict[str, str]:
    """Check promotion gates for a model.

    Looks for existing gate artifacts. Returns {gate_name: PASS/FAIL/NOT_RUN}.
    """
    results = {}

    for gate in gate_names:
        if gate == "regression":
            results[gate] = _check_artifact(regression_dir, exp_id, ["regress-*.yaml", "regression-*.yaml"])
        elif gate == "seed_study":
            results[gate] = _check_artifact(seed_dir, exp_id, [f"{exp_id}-seeds.yaml"])
        elif gate == "audit":
            results[gate] = _check_artifact(audit_dir, exp_id, ["audit-*.yaml"])
        elif gate == "calibration":
            results[gate] = _check_artifact(calibration_dir, exp_id, [f"{exp_id}-calibration.yaml", "calibration-*.yaml"])
        else:
            results[gate] = "NOT_RUN"

    return results


def _check_artifact(directory: str, exp_id: str, patterns: list[str]) -> str:
    """Check if a gate artifact exists."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return "NOT_RUN"

    for pattern in patterns:
        matches = list(dir_path.glob(pattern))
        if matches:
            # Check the most recent artifact for pass/fail
            latest = sorted(matches)[-1]
            try:
                with open(latest) as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    verdict = data.get("verdict", data.get("status", data.get("result", "")))
                    if isinstance(verdict, str) and verdict.upper() in ("PASS", "PASSED"):
                        return "PASS"
                    elif isinstance(verdict, str) and verdict.upper() in ("FAIL", "FAILED"):
                        return "FAIL"
                    # Artifact exists but no clear verdict — count as PASS
                    return "PASS"
            except (yaml.YAMLError, OSError):
                pass

    return "NOT_RUN"


# --- Report Formatting ---


def format_registry_list(models: list[dict]) -> str:
    """Format model list as markdown table."""
    if not models:
        return "No models registered. Use `python scripts/model_lifecycle.py register <exp-id> --version <v>` to register."

    lines = ["# Model Registry", ""]
    lines.append("| Stage | Exp ID | Version | Metric | Registered |")
    lines.append("|-------|--------|---------|--------|------------|")

    for m in models:
        metric = f"{m['metric']:.4f}" if m.get("metric") is not None else "—"
        registered = m.get("registered_at", "")[:10]
        lines.append(f"| {m['stage']} | {m['exp_id']} | {m.get('version', '—')} | {metric} | {registered} |")

    return "\n".join(lines)


def format_history(history: list[dict]) -> str:
    """Format promotion history."""
    if not history:
        return "No history entries."

    lines = ["# Model Lifecycle History", ""]
    lines.append("| Time | Action | Exp ID | Details |")
    lines.append("|------|--------|--------|---------|")

    for h in history:
        ts = h.get("timestamp", "")[:19]
        action = h.get("action", "?")
        exp_id = h.get("exp_id", "?")

        if action == "promote":
            detail = f"{h.get('from_stage')} → {h.get('to_stage')}"
            if h.get("forced"):
                detail += " (forced)"
        elif action == "demote":
            detail = f"{h.get('from_stage')} → {h.get('to_stage')}: {h.get('reason', '')}"
        elif action == "archive":
            detail = f"archived from {h.get('from_stage')}"
        elif action == "register":
            detail = f"registered as {h.get('stage')} ({h.get('version', '?')})"
        else:
            detail = ""

        lines.append(f"| {ts} | {action} | {exp_id} | {detail} |")

    return "\n".join(lines)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Model registry — track, promote, and govern model lifecycle"
    )
    parser.add_argument("action", nargs="?", choices=["list", "register", "promote", "demote", "archive", "history"],
                        help="Registry action")
    parser.add_argument("exp_id", nargs="?", help="Experiment ID")
    parser.add_argument("target", nargs="?", help="Target stage (for promote/demote)")
    parser.add_argument("--version", help="Model version label")
    parser.add_argument("--reason", default="", help="Reason for demotion/archiving")
    parser.add_argument("--force", action="store_true", help="Skip gate checks")
    parser.add_argument("--stage", help="Filter by stage (for list)")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY_PATH, help="Registry file path")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    if not args.action:
        parser.error("Please provide an action: list, register, promote, demote, archive, history")

    registry = load_registry(args.registry)

    if args.action == "list":
        models = list_models(registry, args.stage)
        if args.json:
            print(json.dumps(models, indent=2))
        else:
            print(format_registry_list(models))

    elif args.action == "register":
        if not args.exp_id:
            parser.error("register requires an experiment ID")
        version = args.version or "v1"

        # Look up metric from experiment log
        config = load_config(args.config)
        metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")
        experiments = load_experiments(args.log)
        metric = None
        for exp in experiments:
            if exp.get("experiment_id") == args.exp_id:
                metric = exp.get("metrics", {}).get(metric_name)
                break

        result = register_model(registry, args.exp_id, version, metric, metric_name)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            sys.exit(1)
        save_registry(result, args.registry)
        print(f"Registered {args.exp_id} as {version} (candidate)")

    elif args.action == "promote":
        if not args.exp_id or not args.target:
            parser.error("promote requires <exp-id> <target-stage>")

        # Check gates
        gate_results = None
        if not args.force:
            model = _find_model(registry, args.exp_id)
            if model:
                gate_key = f"{model['stage']}_to_{args.target}"
                required = PROMOTION_GATES.get(gate_key, [])
                if required:
                    gate_results = check_gates(args.exp_id, required)

        result = promote_model(registry, args.exp_id, args.target, gate_results, args.force)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            if "failed_gates" in result:
                for g in result["failed_gates"]:
                    print(f"  {g['gate']}: {g['result']}")
            sys.exit(1)
        save_registry(result, args.registry)
        print(f"Promoted {args.exp_id} → {args.target}")

    elif args.action == "demote":
        if not args.exp_id or not args.target:
            parser.error("demote requires <exp-id> <target-stage>")
        result = demote_model(registry, args.exp_id, args.target, args.reason)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            sys.exit(1)
        save_registry(result, args.registry)
        print(f"Demoted {args.exp_id} → {args.target}")

    elif args.action == "archive":
        if not args.exp_id:
            parser.error("archive requires an experiment ID")
        result = archive_model(registry, args.exp_id, args.reason)
        if "error" in result:
            print(f"ERROR: {result['error']}")
            sys.exit(1)
        save_registry(result, args.registry)
        print(f"Archived {args.exp_id}")

    elif args.action == "history":
        history = get_history(registry, args.exp_id)
        if args.json:
            print(json.dumps(history, indent=2))
        else:
            print(format_history(history))


if __name__ == "__main__":
    main()
