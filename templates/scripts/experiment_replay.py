#!/usr/bin/env python3
"""Re-run historical experiments with current infrastructure.

Read an old experiment's config from log.jsonl, plan a replay with
current code, data, and preprocessing, then compare original vs
replayed metrics. Answers the question: "would this old experiment
perform better/worse with today's pipeline?"

Usage:
    python scripts/experiment_replay.py exp-042
    python scripts/experiment_replay.py exp-042 --with-current-data
    python scripts/experiment_replay.py exp-042 --with-current-preprocessing
    python scripts/experiment_replay.py exp-042 --dry-run
    python scripts/experiment_replay.py --list
    python scripts/experiment_replay.py --json
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

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_REPLAY_DIR = "experiments/replays"


# --- Replay Planning ---


def find_experiment(experiments: list[dict], experiment_id: str) -> dict | None:
    """Find an experiment by ID in the log."""
    for exp in experiments:
        if exp.get("experiment_id") == experiment_id:
            return exp
    return None


def plan_replay(
    original: dict,
    config: dict,
    with_current_data: bool = False,
    with_current_preprocessing: bool = False,
) -> dict:
    """Plan a replay of an original experiment.

    Determines what changes between original and current infrastructure,
    and constructs a replay configuration.

    Args:
        original: Original experiment dict from log.
        config: Current project config.
        with_current_data: Use current data instead of original data path.
        with_current_preprocessing: Use current preprocessing pipeline.

    Returns:
        Replay plan dict with config, changes, and warnings.
    """
    original_config = original.get("config", {})
    replay_config = dict(original_config)
    changes = []
    warnings = []

    # Data source
    if with_current_data:
        current_data = config.get("data", {}).get("path", "")
        original_data = original_config.get("data_path", "") or original_config.get("data", {}).get("path", "")
        if current_data and current_data != original_data:
            replay_config["data_path"] = current_data
            if isinstance(replay_config.get("data"), dict):
                replay_config["data"]["path"] = current_data
            changes.append({
                "field": "data_path",
                "original": original_data,
                "replay": current_data,
                "reason": "Using current data (--with-current-data)",
            })
        elif not current_data:
            warnings.append("No data path in current config — using original data path")

    # Preprocessing
    if with_current_preprocessing:
        current_preproc = config.get("preprocessing", {})
        original_preproc = original_config.get("preprocessing", {})
        if current_preproc and current_preproc != original_preproc:
            replay_config["preprocessing"] = current_preproc
            changes.append({
                "field": "preprocessing",
                "original": original_preproc,
                "replay": current_preproc,
                "reason": "Using current preprocessing (--with-current-preprocessing)",
            })

    # Check for missing dependencies or features
    model_type = original_config.get("model_type", "")
    if model_type:
        # Check if model type still exists in current codebase
        train_path = Path("train.py")
        if train_path.exists():
            train_content = train_path.read_text()
            if model_type not in train_content:
                warnings.append(
                    f"Model type '{model_type}' not found in current train.py — "
                    f"replay may fail"
                )

    # Seed handling — use same seed for reproducibility
    seed = original_config.get("seed", original.get("seed"))
    if seed is not None:
        replay_config["seed"] = seed
    else:
        replay_config["seed"] = 42
        warnings.append("No seed in original experiment — defaulting to 42")

    return {
        "original_id": original.get("experiment_id"),
        "original_timestamp": original.get("timestamp"),
        "original_metrics": original.get("metrics", {}),
        "replay_config": replay_config,
        "changes": changes,
        "warnings": warnings,
        "with_current_data": with_current_data,
        "with_current_preprocessing": with_current_preprocessing,
    }


# --- Replay Execution ---


def execute_replay(
    plan: dict,
    timeout: int = 600,
) -> dict:
    """Execute a replay by running train.py with the replay config.

    Args:
        plan: Replay plan from plan_replay.
        timeout: Max seconds for training.

    Returns:
        Execution result with replay metrics.
    """
    replay_config = plan.get("replay_config", {})
    started_at = datetime.now(timezone.utc).isoformat()

    # Write temporary config
    tmp_config = Path("experiments/replays/.replay-config.yaml")
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_config, "w") as f:
        yaml.dump(replay_config, f, default_flow_style=False, sort_keys=False)

    # Run training
    cmd = ["python", "train.py", "--config", str(tmp_config)]
    seed = replay_config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "started_at": started_at,
            "error": f"Training exceeded {timeout}s timeout",
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "started_at": started_at,
            "error": "train.py not found",
        }

    completed_at = datetime.now(timezone.utc).isoformat()

    if proc.returncode != 0:
        error_snippet = (proc.stderr + proc.stdout)[-500:]
        return {
            "status": "failed",
            "started_at": started_at,
            "completed_at": completed_at,
            "error": _classify_error(proc.stderr + proc.stdout),
            "stderr_tail": error_snippet,
        }

    # Parse metrics from stdout
    metrics = _parse_metrics(proc.stdout)

    # Clean up temp config
    try:
        tmp_config.unlink()
    except OSError:
        pass

    return {
        "status": "completed",
        "started_at": started_at,
        "completed_at": completed_at,
        "metrics": metrics,
    }


def _parse_metrics(stdout: str) -> dict:
    """Parse metrics from training output."""
    metrics = {}
    in_block = False
    for line in stdout.splitlines():
        line = line.strip()
        if line == "---":
            if in_block:
                break
            in_block = True
            continue
        if in_block and ":" in line:
            key, value = line.split(":", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                metrics[key.strip()] = value.strip()
    return metrics


def _classify_error(output: str) -> str:
    """Classify error from output text."""
    output_lower = output.lower()
    if "cuda out of memory" in output_lower or "memoryerror" in output_lower:
        return "oom"
    if "nan" in output_lower and "loss" in output_lower:
        return "nan_loss"
    if "modulenotfounderror" in output_lower or "importerror" in output_lower:
        return "import_error"
    if "filenotfounderror" in output_lower:
        return "file_not_found"
    return "unknown"


# --- Comparison ---


def compare_metrics(
    original_metrics: dict,
    replay_metrics: dict,
    primary_metric: str = "accuracy",
    lower_is_better: bool = False,
) -> dict:
    """Compare original vs replayed metrics.

    Args:
        original_metrics: Metrics from the original experiment.
        replay_metrics: Metrics from the replay.
        primary_metric: Primary metric name.
        lower_is_better: Whether lower values are better.

    Returns:
        Comparison dict with deltas and verdict.
    """
    comparisons = {}
    all_metrics = set(list(original_metrics.keys()) + list(replay_metrics.keys()))

    for metric in sorted(all_metrics):
        orig = original_metrics.get(metric)
        replay = replay_metrics.get(metric)

        entry: dict = {"original": orig, "replay": replay}

        if orig is not None and replay is not None:
            try:
                orig_f = float(orig)
                replay_f = float(replay)
                delta = replay_f - orig_f
                pct = (delta / abs(orig_f) * 100) if orig_f != 0 else 0
                entry["delta"] = round(delta, 6)
                entry["delta_pct"] = round(pct, 2)

                lib = lower_is_better if metric == primary_metric else (
                    metric in {"loss", "mse", "rmse", "mae", "error_rate",
                               "train_seconds", "latency", "latency_ms"}
                )
                if lib:
                    entry["improved"] = delta < 0
                else:
                    entry["improved"] = delta > 0
            except (ValueError, TypeError):
                pass

        comparisons[metric] = entry

    # Overall verdict
    primary = comparisons.get(primary_metric, {})
    if primary.get("improved") is True:
        verdict = "improved"
    elif primary.get("improved") is False:
        verdict = "regressed"
    else:
        verdict = "inconclusive"

    return {
        "primary_metric": primary_metric,
        "verdict": verdict,
        "comparisons": comparisons,
    }


# --- Report ---


def format_replay_report(report: dict) -> str:
    """Format replay result as a readable markdown report."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    lines = [
        "# Experiment Replay",
        "",
        f"*{report.get('timestamp', '?')[:19]} UTC*",
        "",
    ]

    plan = report.get("plan", {})
    lines.extend([
        "## Original Experiment",
        "",
        f"- **ID:** {plan.get('original_id', '?')}",
        f"- **Timestamp:** {plan.get('original_timestamp', '?')[:19]}",
    ])

    orig_metrics = plan.get("original_metrics", {})
    if orig_metrics:
        lines.append("- **Metrics:** " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in orig_metrics.items()
        ))
    lines.append("")

    # Changes
    changes = plan.get("changes", [])
    if changes:
        lines.extend(["## Changes from Original", ""])
        for ch in changes:
            lines.append(f"- **{ch['field']}**: {ch['reason']}")
        lines.append("")

    # Warnings
    warnings = plan.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", ""])
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    # Execution result
    execution = report.get("execution", {})
    status = execution.get("status", "not_run")
    lines.extend([
        "## Replay Result",
        "",
        f"**Status:** {status}",
    ])

    if status == "completed":
        # Comparison
        comparison = report.get("comparison", {})
        verdict = comparison.get("verdict", "?")
        primary = comparison.get("primary_metric", "?")
        lines.extend([
            f"**Verdict:** {verdict} (primary: {primary})",
            "",
            "| Metric | Original | Replay | Delta | Change |",
            "|--------|----------|--------|-------|--------|",
        ])

        for metric, data in comparison.get("comparisons", {}).items():
            orig = data.get("original")
            replay = data.get("replay")
            orig_str = f"{orig:.4f}" if isinstance(orig, float) else str(orig or "—")
            replay_str = f"{replay:.4f}" if isinstance(replay, float) else str(replay or "—")
            delta = data.get("delta_pct")
            delta_str = f"{delta:+.2f}%" if delta is not None else "—"
            improved = data.get("improved")
            if improved is True:
                change = "improved"
            elif improved is False:
                change = "regressed"
            else:
                change = "—"
            lines.append(f"| {metric} | {orig_str} | {replay_str} | {delta_str} | {change} |")
    elif status in ("failed", "timeout", "error"):
        lines.append(f"**Error:** {execution.get('error', 'unknown')}")

    lines.extend(["", "---"])
    return "\n".join(lines)


def save_replay_report(report: dict, replay_dir: str = DEFAULT_REPLAY_DIR) -> Path:
    """Save replay report to YAML."""
    p = Path(replay_dir)
    p.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("plan", {}).get("original_id", "unknown")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = p / f"{exp_id}-replay-{ts}.yaml"
    with open(out, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return out


def list_replays(replay_dir: str = DEFAULT_REPLAY_DIR) -> list[dict]:
    """List all saved replay reports."""
    p = Path(replay_dir)
    if not p.exists():
        return []

    replays = []
    for path in sorted(p.glob("*-replay-*.yaml")):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            plan = data.get("plan", {})
            execution = data.get("execution", {})
            comparison = data.get("comparison", {})
            replays.append({
                "file": path.name,
                "original_id": plan.get("original_id"),
                "timestamp": data.get("timestamp", ""),
                "status": execution.get("status", "?"),
                "verdict": comparison.get("verdict", "?"),
            })
        except (yaml.YAMLError, OSError):
            continue

    return replays


# --- Orchestration ---


def run_replay(
    experiment_id: str | None = None,
    with_current_data: bool = False,
    with_current_preprocessing: bool = False,
    dry_run: bool = False,
    list_mode: bool = False,
    timeout: int = 600,
    log_path: str = DEFAULT_LOG_PATH,
    config_path: str = "config.yaml",
    replay_dir: str = DEFAULT_REPLAY_DIR,
) -> dict:
    """Run experiment replay workflow.

    Args:
        experiment_id: Experiment to replay.
        with_current_data: Use current data.
        with_current_preprocessing: Use current preprocessing.
        dry_run: Plan only, don't execute.
        list_mode: List previous replays.
        timeout: Training timeout in seconds.
        log_path: Path to experiment log.
        config_path: Path to config.yaml.
        replay_dir: Directory for replay reports.

    Returns:
        Replay result dict.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if list_mode:
        replays = list_replays(replay_dir)
        return {
            "timestamp": timestamp,
            "action": "list",
            "count": len(replays),
            "replays": replays,
        }

    if not experiment_id:
        return {"error": "Experiment ID required. Use --list to see past replays."}

    config = load_config(config_path)
    experiments = load_experiments(log_path)

    if not experiments:
        return {"timestamp": timestamp, "error": "No experiments found"}

    original = find_experiment(experiments, experiment_id)
    if original is None:
        return {"timestamp": timestamp, "error": f"Experiment '{experiment_id}' not found"}

    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    # Plan
    plan = plan_replay(original, config, with_current_data, with_current_preprocessing)

    report: dict = {
        "timestamp": timestamp,
        "plan": plan,
    }

    if dry_run:
        report["execution"] = {"status": "dry_run"}
        saved = save_replay_report(report, replay_dir)
        report["saved_to"] = str(saved)
        return report

    # Execute
    execution = execute_replay(plan, timeout=timeout)
    report["execution"] = execution

    # Compare if completed
    if execution.get("status") == "completed":
        comparison = compare_metrics(
            plan.get("original_metrics", {}),
            execution.get("metrics", {}),
            primary_metric=primary_metric,
            lower_is_better=lower_is_better,
        )
        report["comparison"] = comparison

    # Save
    saved = save_replay_report(report, replay_dir)
    report["saved_to"] = str(saved)

    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Re-run historical experiments")
    parser.add_argument("experiment_id", nargs="?", default=None,
                        help="Experiment ID to replay")
    parser.add_argument("--with-current-data", action="store_true",
                        help="Use current data instead of original")
    parser.add_argument("--with-current-preprocessing", action="store_true",
                        help="Use current preprocessing pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan replay without executing")
    parser.add_argument("--list", dest="list_mode", action="store_true",
                        help="List previous replays")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Training timeout in seconds (default: 600)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--replay-dir", default=DEFAULT_REPLAY_DIR,
                        help="Directory for replay reports")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    report = run_replay(
        experiment_id=args.experiment_id,
        with_current_data=args.with_current_data,
        with_current_preprocessing=args.with_current_preprocessing,
        dry_run=args.dry_run,
        list_mode=args.list_mode,
        timeout=args.timeout,
        log_path=args.log,
        config_path=args.config,
        replay_dir=args.replay_dir,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        if "error" in report:
            print(f"ERROR: {report['error']}", file=sys.stderr)
            sys.exit(1)

        if report.get("action") == "list":
            replays = report.get("replays", [])
            if not replays:
                print("No replays found.")
            else:
                print("# Experiment Replays")
                print()
                print("| Original | Date | Status | Verdict |")
                print("|----------|------|--------|---------|")
                for r in replays:
                    print(f"| {r['original_id']} | {r['timestamp'][:10]} "
                          f"| {r['status']} | {r['verdict']} |")
        else:
            print(format_replay_report(report))


if __name__ == "__main__":
    main()
