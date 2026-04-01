#!/usr/bin/env python3
"""Experiment lifecycle cleanup and archival for the autoresearch pipeline.

Identifies archivable experiments (older than threshold, not Pareto-optimal,
not current best), compresses artifacts, and creates a summary index.
Keeps the experiment directory lean without losing institutional knowledge.

Usage:
    python scripts/experiment_archive.py --dry-run
    python scripts/experiment_archive.py --older-than 30
    python scripts/experiment_archive.py --keep-best 5
    python scripts/experiment_archive.py --json
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_OLDER_THAN_DAYS = 30
DEFAULT_KEEP_BEST = 3
ARCHIVE_INDEX_PATH = "experiments/archive/index.yaml"
ARTIFACT_DIRS = ["checkpoints", "predictions", "profiles", "diagnoses"]


# --- Identification ---


def find_current_best(
    experiments: list[dict],
    metric: str,
    lower_is_better: bool,
    keep_best: int = DEFAULT_KEEP_BEST,
) -> set[str]:
    """Find the top-N best kept experiments by primary metric.

    Returns set of experiment IDs that should never be archived.
    """
    kept = []
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
        kept.append((exp.get("experiment_id", "?"), val))

    kept.sort(key=lambda x: x[1], reverse=not lower_is_better)
    return {eid for eid, _ in kept[:keep_best]}


def find_pareto_optimal(
    experiments: list[dict],
    metrics: list[str],
    lower_is_better_map: dict[str, bool],
) -> set[str]:
    """Find Pareto-optimal experiments across all configured metrics.

    Returns set of experiment IDs that should never be archived.
    """
    kept = []
    for exp in experiments:
        if exp.get("status") != "kept":
            continue
        exp_metrics = exp.get("metrics", {})
        values = {}
        complete = True
        for m in metrics:
            v = exp_metrics.get(m)
            if v is None:
                complete = False
                break
            try:
                values[m] = float(v)
            except (ValueError, TypeError):
                complete = False
                break
        if complete:
            kept.append((exp.get("experiment_id", "?"), values))

    if not kept:
        return set()

    pareto_ids = set()
    for i, (eid_i, vals_i) in enumerate(kept):
        dominated = False
        for j, (eid_j, vals_j) in enumerate(kept):
            if i == j:
                continue
            all_ge = True
            strictly_better = False
            for m in metrics:
                lib = lower_is_better_map.get(m, False)
                if lib:
                    if vals_j[m] > vals_i[m]:
                        all_ge = False
                        break
                    if vals_j[m] < vals_i[m]:
                        strictly_better = True
                else:
                    if vals_j[m] < vals_i[m]:
                        all_ge = False
                        break
                    if vals_j[m] > vals_i[m]:
                        strictly_better = True
            if all_ge and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_ids.add(eid_i)

    return pareto_ids


def identify_archivable(
    experiments: list[dict],
    metric: str,
    lower_is_better: bool,
    older_than_days: int = DEFAULT_OLDER_THAN_DAYS,
    keep_best: int = DEFAULT_KEEP_BEST,
    metrics_list: list[str] | None = None,
) -> tuple[list[dict], set[str]]:
    """Identify experiments that can be safely archived.

    An experiment is archivable if ALL of:
    - older than older_than_days
    - not in the top-N best
    - not Pareto-optimal
    - not the most recent experiment

    Returns (archivable_experiments, protected_ids).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    cutoff_str = cutoff.isoformat()

    # Protected sets
    best_ids = find_current_best(experiments, metric, lower_is_better, keep_best)

    lower_metrics = {"train_seconds", "latency", "latency_ms", "n_params",
                     "mse", "rmse", "mae", "loss", "log_loss", "error_rate"}
    if metrics_list and len(metrics_list) >= 2:
        lib_map = {}
        for m in metrics_list:
            if m == metric:
                lib_map[m] = lower_is_better
            else:
                lib_map[m] = m in lower_metrics
        pareto_ids = find_pareto_optimal(experiments, metrics_list, lib_map)
    else:
        pareto_ids = set()

    # Most recent experiment is always protected
    most_recent_id = experiments[-1].get("experiment_id", "") if experiments else ""

    protected = best_ids | pareto_ids | {most_recent_id}

    archivable = []
    for exp in experiments:
        eid = exp.get("experiment_id", "?")
        ts = exp.get("timestamp", "")

        if eid in protected:
            continue
        if ts >= cutoff_str:
            continue

        archivable.append(exp)

    return archivable, protected


# --- Artifact Discovery ---


def find_experiment_artifacts(experiment_id: str) -> list[dict]:
    """Find all artifact files associated with an experiment.

    Scans known artifact directories for files matching the experiment ID.
    """
    artifacts = []
    for dirname in ARTIFACT_DIRS:
        dirpath = Path(f"experiments/{dirname}")
        if not dirpath.exists():
            continue
        for f in dirpath.iterdir():
            if experiment_id in f.name and f.is_file():
                artifacts.append({
                    "path": str(f),
                    "size_bytes": f.stat().st_size,
                    "directory": dirname,
                })

    return artifacts


# --- Archival Operations ---


def compress_artifact(filepath: str) -> dict:
    """Compress a single artifact file with gzip.

    Returns dict with original/compressed sizes and the compressed path.
    """
    src = Path(filepath)
    if not src.exists():
        return {"error": f"File not found: {filepath}"}
    if src.suffix == ".gz":
        return {"skipped": True, "path": filepath, "reason": "Already compressed"}

    dst = Path(f"{filepath}.gz")
    original_size = src.stat().st_size

    with open(src, "rb") as f_in:
        with gzip.open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    compressed_size = dst.stat().st_size
    src.unlink()

    return {
        "original_path": filepath,
        "compressed_path": str(dst),
        "original_size": original_size,
        "compressed_size": compressed_size,
        "ratio": round(compressed_size / original_size, 3) if original_size > 0 else 0,
    }


def create_experiment_summary(exp: dict) -> dict:
    """Create a compact summary of an experiment for the archive index."""
    return {
        "experiment_id": exp.get("experiment_id", "?"),
        "timestamp": exp.get("timestamp", ""),
        "status": exp.get("status", "?"),
        "model_type": exp.get("config", {}).get("model_type", "?"),
        "family": exp.get("family"),
        "description": exp.get("description", ""),
        "metrics": exp.get("metrics", {}),
        "config_summary": {
            "model_type": exp.get("config", {}).get("model_type"),
            "experiment_type": exp.get("config", {}).get("experiment_type"),
        },
    }


def load_archive_index(path: str = ARCHIVE_INDEX_PATH) -> dict:
    """Load existing archive index."""
    p = Path(path)
    if not p.exists():
        return {
            "created": datetime.now(timezone.utc).isoformat(),
            "archived_experiments": [],
            "total_space_saved_bytes": 0,
        }
    try:
        with open(p) as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {
                "created": datetime.now(timezone.utc).isoformat(),
                "archived_experiments": [],
                "total_space_saved_bytes": 0,
            }
    except (yaml.YAMLError, OSError):
        return {
            "created": datetime.now(timezone.utc).isoformat(),
            "archived_experiments": [],
            "total_space_saved_bytes": 0,
        }


def save_archive_index(index: dict, path: str = ARCHIVE_INDEX_PATH) -> Path:
    """Save archive index to YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(index, f, default_flow_style=False, sort_keys=False)
    return p


def archive_experiments(
    archivable: list[dict],
    dry_run: bool = False,
) -> dict:
    """Archive a list of experiments: compress artifacts, update index.

    Args:
        archivable: List of experiment dicts to archive.
        dry_run: If True, report what would happen without changing anything.

    Returns:
        Archive operation result dict.
    """
    results = []
    total_saved = 0

    for exp in archivable:
        eid = exp.get("experiment_id", "?")
        artifacts = find_experiment_artifacts(eid)
        summary = create_experiment_summary(exp)

        entry = {
            "experiment_id": eid,
            "summary": summary,
            "artifacts_found": len(artifacts),
            "artifacts": [],
        }

        for artifact in artifacts:
            if dry_run:
                entry["artifacts"].append({
                    "path": artifact["path"],
                    "size_bytes": artifact["size_bytes"],
                    "action": "would_compress",
                })
            else:
                result = compress_artifact(artifact["path"])
                entry["artifacts"].append(result)
                if "original_size" in result and "compressed_size" in result:
                    total_saved += result["original_size"] - result["compressed_size"]

        results.append(entry)

    # Update index
    if not dry_run and results:
        index = load_archive_index()
        for entry in results:
            index["archived_experiments"].append({
                "experiment_id": entry["experiment_id"],
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "summary": entry["summary"],
                "artifacts_compressed": len(entry["artifacts"]),
            })
        index["total_space_saved_bytes"] = (
            index.get("total_space_saved_bytes", 0) + total_saved
        )
        index["last_archive"] = datetime.now(timezone.utc).isoformat()
        save_archive_index(index)

    return {
        "archived": len(results),
        "total_artifacts": sum(e["artifacts_found"] for e in results),
        "space_saved_bytes": total_saved,
        "dry_run": dry_run,
        "entries": results,
    }


# --- Report ---


def format_archive_report(report: dict) -> str:
    """Format archive operation as markdown report."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    lines = [
        "# Experiment Archive",
        "",
        f"*Generated {report.get('timestamp', '?')[:19]} UTC*",
        "",
    ]

    summary = report.get("summary", {})
    lines.extend([
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total experiments | {summary.get('total_experiments', 0)} |",
        f"| Archivable | {summary.get('archivable', 0)} |",
        f"| Protected | {summary.get('protected', 0)} |",
    ])

    protected_reasons = summary.get("protected_reasons", {})
    if protected_reasons:
        lines.extend([
            "",
            "**Protected experiments:**",
        ])
        for reason, ids in protected_reasons.items():
            lines.append(f"- {reason}: {', '.join(ids)}")
    lines.append("")

    # Archive results
    archive = report.get("archive", {})
    if archive.get("dry_run"):
        lines.extend(["## Dry Run (no changes made)", ""])
    else:
        lines.extend(["## Archived", ""])

    if archive.get("archived", 0) > 0:
        lines.append(f"**{archive['archived']}** experiments, "
                      f"**{archive['total_artifacts']}** artifacts")
        if archive.get("space_saved_bytes", 0) > 0:
            saved_mb = archive["space_saved_bytes"] / (1024 * 1024)
            lines.append(f"**{saved_mb:.1f} MB** space saved by compression")
        lines.append("")

        lines.append("| Experiment | Status | Artifacts |")
        lines.append("|------------|--------|-----------|")
        for entry in archive.get("entries", []):
            eid = entry["experiment_id"]
            n_art = entry["artifacts_found"]
            status = entry.get("summary", {}).get("status", "?")
            lines.append(f"| {eid} | {status} | {n_art} files |")
    else:
        lines.append("No experiments to archive.")

    lines.extend(["", "---"])

    if archive.get("dry_run"):
        lines.append("*Run without `--dry-run` to execute archival.*")
    else:
        lines.append(f"*Archive index saved to `{ARCHIVE_INDEX_PATH}`*")

    return "\n".join(lines)


def run_archive(
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    older_than: int = DEFAULT_OLDER_THAN_DAYS,
    keep_best: int = DEFAULT_KEEP_BEST,
    dry_run: bool = False,
) -> dict:
    """Run the archive workflow.

    Args:
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        older_than: Archive experiments older than this many days.
        keep_best: Never archive the top-N best experiments.
        dry_run: If True, report without making changes.

    Returns:
        Archive result dict.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)
    metrics_list = eval_cfg.get("metrics", [metric])

    experiments = load_experiments(log_path)
    if not experiments:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "No experiments found",
            "log_path": log_path,
        }

    archivable, protected = identify_archivable(
        experiments, metric, lower_is_better,
        older_than_days=older_than,
        keep_best=keep_best,
        metrics_list=metrics_list if len(metrics_list) >= 2 else None,
    )

    # Categorize protected experiments
    best_ids = find_current_best(experiments, metric, lower_is_better, keep_best)
    most_recent = experiments[-1].get("experiment_id", "") if experiments else ""
    protected_reasons: dict[str, list[str]] = {}
    if best_ids:
        protected_reasons[f"top-{keep_best} best"] = sorted(best_ids)
    if most_recent:
        protected_reasons["most recent"] = [most_recent]
    pareto_only = protected - best_ids - {most_recent}
    if pareto_only:
        protected_reasons["Pareto-optimal"] = sorted(pareto_only)

    archive_result = archive_experiments(archivable, dry_run=dry_run)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "older_than_days": older_than,
            "keep_best": keep_best,
            "metric": metric,
            "lower_is_better": lower_is_better,
        },
        "summary": {
            "total_experiments": len(experiments),
            "archivable": len(archivable),
            "protected": len(protected),
            "protected_reasons": protected_reasons,
        },
        "archive": archive_result,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Experiment lifecycle cleanup and archival")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--older-than", type=int, default=DEFAULT_OLDER_THAN_DAYS,
                        help=f"Archive experiments older than N days (default: {DEFAULT_OLDER_THAN_DAYS})")
    parser.add_argument("--keep-best", type=int, default=DEFAULT_KEEP_BEST,
                        help=f"Never archive the top-N best experiments (default: {DEFAULT_KEEP_BEST})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be archived without making changes")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    report = run_archive(
        config_path=args.config,
        log_path=args.log,
        older_than=args.older_than,
        keep_best=args.keep_best,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_archive_report(report))


if __name__ == "__main__":
    main()
