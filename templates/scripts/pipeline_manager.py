#!/usr/bin/env python3
"""Pipeline composition manager for the autoresearch pipeline.

Decomposes ML pipelines into swappable stages (preprocess, features,
model, postprocess). Each stage can be independently varied, cached,
and reused across experiments.

Usage:
    python scripts/pipeline_manager.py show
    python scripts/pipeline_manager.py swap model --from exp-031
    python scripts/pipeline_manager.py cache
    python scripts/pipeline_manager.py run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_CACHE_DIR = "experiments/cache"
DEFAULT_STAGES = ["preprocess", "features", "model", "postprocess"]


# --- Stage Definition ---


def define_stages(config: dict) -> list[dict]:
    """Extract pipeline stage definitions from config.

    Args:
        config: Project config dict.

    Returns:
        List of stage dicts with name, type, config_keys, and description.
    """
    pipeline_cfg = config.get("pipeline", {})
    stage_names = pipeline_cfg.get("stages", DEFAULT_STAGES)

    stages = []
    for name in stage_names:
        stage = {
            "name": name,
            "config": pipeline_cfg.get(name, {}),
            "description": _describe_stage(name, config),
        }
        stage["hash"] = compute_stage_hash(stage)
        stages.append(stage)

    return stages


def _describe_stage(name: str, config: dict) -> str:
    """Generate a human-readable description of a pipeline stage."""
    if name == "preprocess":
        model_cfg = config.get("model", {})
        steps = []
        if model_cfg.get("hyperparams", {}).get("handle_missing"):
            steps.append("handle_missing")
        if model_cfg.get("hyperparams", {}).get("scaler"):
            steps.append(model_cfg["hyperparams"]["scaler"])
        return ", ".join(steps) if steps else "default preprocessing"

    elif name == "features":
        model_cfg = config.get("model", {})
        features = model_cfg.get("features", [])
        if features:
            return ", ".join(features[:5])
        return "raw features"

    elif name == "model":
        model_cfg = config.get("model", {})
        model_type = model_cfg.get("type", "unknown")
        hyperparams = model_cfg.get("hyperparams", {})
        key_params = []
        for k in ("max_depth", "n_estimators", "learning_rate", "hidden_size"):
            if k in hyperparams:
                key_params.append(f"{k}={hyperparams[k]}")
        return f"{model_type}({', '.join(key_params)})" if key_params else model_type

    elif name == "postprocess":
        return config.get("postprocess", {}).get("type", "none")

    return "unknown"


def compute_stage_hash(stage: dict) -> str:
    """Compute a content hash for a stage configuration.

    Used to detect when a stage hasn't changed and skip re-computation.
    """
    content = json.dumps(stage.get("config", {}), sort_keys=True)
    content += stage.get("description", "")
    return hashlib.md5(content.encode()).hexdigest()[:8]


# --- Stage Operations ---


def extract_stage_from_experiment(
    experiment: dict,
    stage_name: str,
) -> dict:
    """Extract a specific stage's configuration from an experiment.

    Args:
        experiment: Experiment dict from log.
        stage_name: Name of the stage to extract.

    Returns:
        Stage config dict.
    """
    config = experiment.get("config", {})
    hyperparams = config.get("hyperparams", {})

    if stage_name == "model":
        return {
            "name": "model",
            "config": {
                "type": config.get("model_type", "unknown"),
                "hyperparams": hyperparams,
            },
            "source_experiment": experiment.get("experiment_id"),
        }

    elif stage_name == "preprocess":
        preprocess_keys = {"scaler", "handle_missing", "imputer", "normalize"}
        preprocess_cfg = {k: v for k, v in hyperparams.items() if k in preprocess_keys}
        return {
            "name": "preprocess",
            "config": preprocess_cfg,
            "source_experiment": experiment.get("experiment_id"),
        }

    elif stage_name == "features":
        feature_keys = {"features", "feature_engineering", "polynomial", "interactions"}
        feature_cfg = {k: v for k, v in hyperparams.items() if k in feature_keys}
        return {
            "name": "features",
            "config": feature_cfg,
            "source_experiment": experiment.get("experiment_id"),
        }

    elif stage_name == "postprocess":
        post_keys = {"calibration", "threshold", "postprocess"}
        post_cfg = {k: v for k, v in hyperparams.items() if k in post_keys}
        return {
            "name": "postprocess",
            "config": post_cfg,
            "source_experiment": experiment.get("experiment_id"),
        }

    return {"name": stage_name, "config": {}, "source_experiment": experiment.get("experiment_id")}


def swap_stage(
    current_stages: list[dict],
    stage_name: str,
    new_stage: dict,
) -> list[dict]:
    """Replace a stage in the pipeline with a new one.

    Args:
        current_stages: Current pipeline stages.
        stage_name: Name of stage to replace.
        new_stage: New stage configuration.

    Returns:
        Updated pipeline stages.
    """
    result = []
    for stage in current_stages:
        if stage["name"] == stage_name:
            new_stage["hash"] = compute_stage_hash(new_stage)
            result.append(new_stage)
        else:
            result.append(stage)
    return result


# --- Cache Management ---


def get_cache_path(stage: dict, cache_dir: str = DEFAULT_CACHE_DIR) -> Path:
    """Get the cache file path for a stage."""
    stage_hash = stage.get("hash", compute_stage_hash(stage))
    return Path(cache_dir) / f"{stage['name']}-{stage_hash}"


def check_cache(stage: dict, cache_dir: str = DEFAULT_CACHE_DIR) -> bool:
    """Check if a stage's output is cached."""
    cache_path = get_cache_path(stage, cache_dir)
    return cache_path.exists()


def get_cache_stats(cache_dir: str = DEFAULT_CACHE_DIR) -> dict:
    """Get cache directory statistics."""
    path = Path(cache_dir)
    if not path.exists():
        return {"total_files": 0, "total_size_bytes": 0, "stages": {}}

    total_size = 0
    total_files = 0
    stages = {}

    for entry in path.iterdir():
        if entry.is_dir():
            size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            n_files = sum(1 for f in entry.rglob("*") if f.is_file())
            stage_name = entry.name.rsplit("-", 1)[0] if "-" in entry.name else entry.name
            stages[entry.name] = {"size_bytes": size, "n_files": n_files}
            total_size += size
            total_files += n_files
        elif entry.is_file():
            total_size += entry.stat().st_size
            total_files += 1

    return {
        "total_files": total_files,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "stages": stages,
    }


# --- Pipeline Composition ---


def compose_pipeline(
    action: str,
    stage_name: str | None = None,
    from_exp: str | None = None,
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> dict:
    """Execute a pipeline composition action.

    Args:
        action: One of show, swap, cache, run.
        stage_name: Stage to operate on (for swap).
        from_exp: Experiment ID to take stage from (for swap).
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        cache_dir: Cache directory path.

    Returns:
        Action result dict.
    """
    config = load_config(config_path)

    if action == "show":
        stages = define_stages(config)
        cache_status = {}
        for stage in stages:
            cache_status[stage["name"]] = check_cache(stage, cache_dir)

        return {
            "action": "show",
            "stages": [
                {
                    "name": s["name"],
                    "description": s["description"],
                    "hash": s["hash"],
                    "cached": cache_status.get(s["name"], False),
                }
                for s in stages
            ],
            "cache_stats": get_cache_stats(cache_dir),
        }

    elif action == "swap":
        if not stage_name:
            return {"error": "Stage name required for swap action"}
        if not from_exp:
            return {"error": "Source experiment ID required (--from)"}

        experiments = load_experiments(log_path)
        source = None
        for exp in experiments:
            if exp.get("experiment_id") == from_exp:
                source = exp
                break

        if not source:
            return {"error": f"Experiment {from_exp} not found"}

        current_stages = define_stages(config)
        new_stage = extract_stage_from_experiment(source, stage_name)
        updated = swap_stage(current_stages, stage_name, new_stage)

        return {
            "action": "swap",
            "stage": stage_name,
            "source_experiment": from_exp,
            "old_stage": next((s for s in current_stages if s["name"] == stage_name), None),
            "new_stage": new_stage,
            "updated_pipeline": [
                {"name": s["name"], "description": s.get("description", ""), "hash": s.get("hash", "")}
                for s in updated
            ],
        }

    elif action == "cache":
        stages = define_stages(config)
        return {
            "action": "cache",
            "stages": [
                {
                    "name": s["name"],
                    "hash": s["hash"],
                    "cache_path": str(get_cache_path(s, cache_dir)),
                    "already_cached": check_cache(s, cache_dir),
                }
                for s in stages
            ],
            "cache_dir": cache_dir,
            "stats": get_cache_stats(cache_dir),
        }

    elif action == "run":
        stages = define_stages(config)
        skippable = [s["name"] for s in stages if check_cache(s, cache_dir)]
        return {
            "action": "run",
            "total_stages": len(stages),
            "cached_stages": skippable,
            "stages_to_run": [s["name"] for s in stages if s["name"] not in skippable],
            "message": f"Would skip {len(skippable)} cached stage(s) and run {len(stages) - len(skippable)}",
        }

    return {"error": f"Unknown action: {action}"}


# --- Report Formatting ---


def format_pipeline_report(report: dict) -> str:
    """Format pipeline report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    action = report.get("action", "?")
    lines = [f"# Pipeline: {action.title()}", ""]

    if action == "show":
        lines.extend(["## Pipeline Stages", ""])
        lines.append("| # | Stage | Description | Hash | Cached |")
        lines.append("|---|-------|-------------|------|--------|")
        for i, s in enumerate(report.get("stages", []), 1):
            cached = "yes" if s.get("cached") else "no"
            lines.append(f"| {i} | {s['name']} | {s['description']} | {s['hash']} | {cached} |")

        stats = report.get("cache_stats", {})
        if stats.get("total_files", 0) > 0:
            lines.extend([
                "",
                f"**Cache:** {stats.get('total_files', 0)} files, "
                f"{stats.get('total_size_mb', 0):.1f} MB",
            ])

    elif action == "swap":
        old = report.get("old_stage", {})
        new = report.get("new_stage", {})
        lines.extend([
            f"**Swapped stage:** {report.get('stage')}",
            f"**Source experiment:** {report.get('source_experiment')}",
            "",
            f"Old: {old.get('description', 'N/A')} (hash: {old.get('hash', '?')})",
            f"New: {new.get('config', {})} from {report.get('source_experiment')}",
            "",
            "## Updated Pipeline",
            "",
        ])
        for s in report.get("updated_pipeline", []):
            lines.append(f"- {s['name']}: {s['description']} ({s['hash']})")

    elif action == "cache":
        lines.append("| Stage | Hash | Path | Status |")
        lines.append("|-------|------|------|--------|")
        for s in report.get("stages", []):
            status = "cached" if s.get("already_cached") else "to cache"
            lines.append(f"| {s['name']} | {s['hash']} | {s['cache_path']} | {status} |")

    elif action == "run":
        cached = report.get("cached_stages", [])
        to_run = report.get("stages_to_run", [])
        lines.append(f"**{report.get('message')}**")
        lines.append("")
        if cached:
            lines.append(f"Skip (cached): {', '.join(cached)}")
        if to_run:
            lines.append(f"Run: {', '.join(to_run)}")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pipeline composition manager",
    )
    parser.add_argument(
        "action", choices=["show", "swap", "cache", "run"],
        help="Pipeline action",
    )
    parser.add_argument(
        "stage", nargs="?",
        help="Stage name (for swap)",
    )
    parser.add_argument(
        "--from", dest="from_exp",
        help="Source experiment ID (for swap)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--log", default=DEFAULT_LOG_PATH,
        help="Path to experiment log",
    )
    parser.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    report = compose_pipeline(
        action=args.action,
        stage_name=args.stage,
        from_exp=args.from_exp,
        config_path=args.config,
        log_path=args.log,
        cache_dir=args.cache_dir,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_pipeline_report(report))


if __name__ == "__main__":
    main()
