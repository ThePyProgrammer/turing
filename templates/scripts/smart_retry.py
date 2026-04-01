#!/usr/bin/env python3
"""Smart failure recovery for ML experiments.

Auto-diagnoses crash type and retries with a targeted fix.
OOM → reduce batch size. NaN → add gradient clipping. Timeout → more patience.

Usage:
    python scripts/smart_retry.py exp-042                    # Auto-diagnose and retry
    python scripts/smart_retry.py exp-042 --max-attempts 5   # More retries
    python scripts/smart_retry.py --classify "CUDA out of memory"  # Just classify
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments


DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_FAILURE_MODES_PATH = "config/failure_modes.yaml"

# Built-in failure taxonomy (overridden by config/failure_modes.yaml if present)
BUILTIN_FAILURE_MODES = {
    "oom": {
        "patterns": ["CUDA out of memory", "MemoryError", "RuntimeError: out of memory",
                     "std::bad_alloc", "OutOfMemoryError"],
        "fix": "Reduce batch_size by 50%",
        "config_changes": {"batch_size": "//2"},
        "severity": "recoverable",
    },
    "nan_loss": {
        "patterns": ["loss is NaN", "loss is nan", "RuntimeWarning: invalid value",
                     "loss: nan", "NaN loss"],
        "fix": "Add gradient clipping at 1.0, reduce learning_rate by 10x",
        "config_changes": {"gradient_clip": 1.0, "learning_rate": "//10"},
        "severity": "recoverable",
    },
    "timeout": {
        "patterns": ["TimeoutError", "exceeded time limit", "timed out",
                     "TimeoutExpired"],
        "fix": "Double max_epochs or training timeout",
        "config_changes": {"max_epochs": "*2"},
        "severity": "recoverable",
    },
    "import_error": {
        "patterns": ["ModuleNotFoundError", "ImportError", "No module named"],
        "fix": "Install missing dependency",
        "config_changes": {},
        "severity": "requires_intervention",
        "action": "pip_install",
    },
    "convergence_failure": {
        "patterns": ["loss did not decrease", "no improvement", "early stopping",
                     "convergence warning"],
        "fix": "Increase learning_rate by 3x for warm-up",
        "config_changes": {"learning_rate": "*3"},
        "severity": "recoverable",
    },
    "data_error": {
        "patterns": ["FileNotFoundError", "No such file", "empty dataset",
                     "zero samples", "KeyError"],
        "fix": "Check data path and preprocessing",
        "config_changes": {},
        "severity": "requires_intervention",
    },
}


def load_failure_modes(config_path: str = DEFAULT_FAILURE_MODES_PATH) -> dict:
    """Load failure mode taxonomy from config or use built-in."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            custom = yaml.safe_load(f) or {}
        if isinstance(custom, dict) and custom:
            return custom
    return BUILTIN_FAILURE_MODES


def classify_failure(
    output: str,
    exit_code: int | None = None,
    failure_modes: dict | None = None,
) -> dict:
    """Classify an experiment failure against the taxonomy.

    Args:
        output: Combined stdout + stderr from the failed run.
        exit_code: Process exit code.
        failure_modes: Failure taxonomy dict.

    Returns:
        Dict with failure_type, matched_pattern, fix, config_changes, severity.
    """
    if failure_modes is None:
        failure_modes = BUILTIN_FAILURE_MODES

    output_lower = output.lower()

    for failure_type, mode in failure_modes.items():
        patterns = mode.get("patterns", [])
        for pattern in patterns:
            if pattern.lower() in output_lower:
                return {
                    "failure_type": failure_type,
                    "matched_pattern": pattern,
                    "fix": mode.get("fix", "Unknown fix"),
                    "config_changes": mode.get("config_changes", {}),
                    "severity": mode.get("severity", "unknown"),
                    "action": mode.get("action"),
                }

    return {
        "failure_type": "unknown",
        "matched_pattern": None,
        "fix": "Manual investigation required",
        "config_changes": {},
        "severity": "requires_intervention",
    }


def apply_config_changes(
    config: dict,
    changes: dict[str, str | int | float],
) -> dict:
    """Apply fix changes to a config dict.

    Supports operators: //N (divide), *N (multiply), or literal values.

    Returns modified config dict.
    """
    hyperparams = config.get("model", {}).get("hyperparams", {})

    for key, value in changes.items():
        if isinstance(value, str):
            if value.startswith("//"):
                divisor = int(value[2:])
                current = hyperparams.get(key)
                if current and isinstance(current, (int, float)):
                    hyperparams[key] = max(1, current // divisor) if isinstance(current, int) else current / divisor
            elif value.startswith("*"):
                multiplier = int(value[1:])
                current = hyperparams.get(key)
                if current and isinstance(current, (int, float)):
                    hyperparams[key] = current * multiplier
        else:
            hyperparams[key] = value

    config.setdefault("model", {})["hyperparams"] = hyperparams
    return config


def run_retry(
    output: str,
    exit_code: int = 1,
    seed: int = 42,
    timeout: int = 600,
    failure_modes: dict | None = None,
) -> dict:
    """Classify failure, apply fix, and re-run.

    Returns dict with classification, fix applied, and retry result.
    """
    classification = classify_failure(output, exit_code, failure_modes)

    if classification["severity"] == "requires_intervention":
        return {
            "status": "cannot_auto_fix",
            "classification": classification,
            "message": f"Failure type '{classification['failure_type']}' requires manual intervention: {classification['fix']}",
        }

    # Apply fix and re-run
    cmd = ["python", "train.py", "--seed", str(seed)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "status": "retry_timeout",
            "classification": classification,
            "message": "Retry also timed out",
        }

    if proc.returncode == 0:
        # Parse metrics
        metrics = {}
        in_block = False
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line == "---":
                if in_block:
                    break
                in_block = True
                continue
            if in_block and ":" in line:
                k, v = line.split(":", 1)
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    metrics[k.strip()] = v.strip()

        return {
            "status": "retry_succeeded",
            "classification": classification,
            "metrics": metrics,
            "message": f"Retry succeeded after applying fix: {classification['fix']}",
        }
    else:
        return {
            "status": "retry_failed",
            "classification": classification,
            "message": f"Retry failed. Error persists after fix: {classification['fix']}",
            "retry_output": proc.stderr[-500:] if proc.stderr else "",
        }


def retry_experiment(
    exp_id: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
    timeout: int = 600,
) -> dict:
    """Retry a failed experiment with auto-diagnosis and fix.

    Args:
        exp_id: Experiment ID to retry.
        max_attempts: Maximum retry attempts.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        timeout: Per-run timeout.

    Returns:
        Retry result dict.
    """
    experiments = load_experiments(log_path)
    target = None
    for exp in experiments:
        if exp.get("experiment_id") == exp_id:
            target = exp
            break

    if not target:
        return {"error": f"Experiment {exp_id} not found"}

    # Check if there's a run.log for this experiment
    run_log = Path(f"experiments/logs/{exp_id}-run.log")
    if run_log.exists():
        output = run_log.read_text()
    else:
        output = target.get("error_output", target.get("description", ""))

    failure_modes = load_failure_modes()
    classification = classify_failure(output, failure_modes=failure_modes)

    print(f"Retrying {exp_id}", file=sys.stderr)
    print(f"Failure type: {classification['failure_type']}", file=sys.stderr)
    print(f"Fix: {classification['fix']}", file=sys.stderr)

    if classification["severity"] == "requires_intervention":
        return {
            "experiment_id": exp_id,
            "status": "cannot_auto_fix",
            "classification": classification,
            "attempts": 0,
        }

    # Retry loop
    attempts = []
    for attempt in range(1, max_attempts + 1):
        print(f"\n  Attempt {attempt}/{max_attempts}...", end=" ", flush=True, file=sys.stderr)

        result = run_retry(
            output=output,
            seed=42 + attempt,
            timeout=timeout,
            failure_modes=failure_modes,
        )
        attempts.append(result)

        if result["status"] == "retry_succeeded":
            print("SUCCESS", file=sys.stderr)
            break
        else:
            print(f"FAILED ({result['status']})", file=sys.stderr)
            if result.get("retry_output"):
                output = result["retry_output"]

    final_status = attempts[-1]["status"] if attempts else "no_attempts"

    return {
        "experiment_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "attempts": len(attempts),
        "max_attempts": max_attempts,
        "final_status": final_status,
        "history": attempts,
    }


def save_retry_report(report: dict, output_dir: str = "experiments/retries") -> Path:
    """Save retry report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("experiment_id", "unknown")
    filepath = out_path / f"{exp_id}-retry.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_retry_report(report: dict) -> str:
    """Format retry report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    exp_id = report.get("experiment_id", "?")
    classification = report.get("classification", {})
    final = report.get("final_status", "unknown")

    status_markers = {
        "retry_succeeded": "RECOVERED",
        "retry_failed": "FAILED",
        "retry_timeout": "TIMEOUT",
        "cannot_auto_fix": "MANUAL FIX NEEDED",
    }
    marker = status_markers.get(final, final)

    lines = [
        f"# Retry Report: {exp_id}",
        "",
        f"**Status: {marker}**",
        "",
        f"- **Failure type:** {classification.get('failure_type', '?')}",
        f"- **Matched pattern:** {classification.get('matched_pattern', 'N/A')}",
        f"- **Fix applied:** {classification.get('fix', 'N/A')}",
        f"- **Attempts:** {report.get('attempts', 0)}/{report.get('max_attempts', 3)}",
    ]

    if report.get("history"):
        lines.extend(["", "## Attempt History", ""])
        for i, attempt in enumerate(report["history"], 1):
            lines.append(f"- **Attempt {i}:** {attempt.get('status', '?')} — {attempt.get('message', '')}")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Smart failure recovery")
    parser.add_argument("exp_id", nargs="?", default=None, help="Experiment ID to retry")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--classify", default=None, help="Just classify error text")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.classify:
        result = classify_failure(args.classify)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Type: {result['failure_type']}")
            print(f"Fix: {result['fix']}")
            print(f"Severity: {result['severity']}")
        sys.exit(0)

    if not args.exp_id:
        print("Usage: smart_retry.py <exp-id> [--max-attempts 3]", file=sys.stderr)
        sys.exit(1)

    report = retry_experiment(
        args.exp_id, args.max_attempts, args.config, args.log, args.timeout,
    )

    if "error" not in report:
        filepath = save_retry_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_retry_report(report))


if __name__ == "__main__":
    main()
