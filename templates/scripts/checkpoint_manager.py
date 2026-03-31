#!/usr/bin/env python3
"""Smart checkpoint manager for ML experiments.

Manages model checkpoints based on Pareto dominance rather than
simple "keep last K". Supports listing, Pareto-based pruning,
checkpoint averaging, and resume from any point.

Usage:
    python scripts/checkpoint_manager.py list                          # List checkpoints
    python scripts/checkpoint_manager.py prune                         # Remove dominated
    python scripts/checkpoint_manager.py average [--top 3]             # Average top-K
    python scripts/checkpoint_manager.py resume exp-042                # Resume from checkpoint
    python scripts/checkpoint_manager.py stats                         # Disk usage summary
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments


DEFAULT_CHECKPOINT_DIR = "experiments/checkpoints"


def scan_checkpoints(checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> list[dict]:
    """Scan checkpoint directory and return metadata for each checkpoint.

    Returns list of dicts with path, experiment_id, metrics, size_bytes, created.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return []

    checkpoints = []
    for entry in sorted(ckpt_path.iterdir()):
        if entry.is_dir():
            # Look for metadata.yaml or metadata.json in the checkpoint dir
            meta_path = entry / "metadata.yaml"
            meta_json = entry / "metadata.json"

            metadata = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = yaml.safe_load(f) or {}
            elif meta_json.exists():
                with open(meta_json) as f:
                    metadata = json.load(f)

            # Compute directory size
            size_bytes = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())

            checkpoints.append({
                "path": str(entry),
                "name": entry.name,
                "experiment_id": metadata.get("experiment_id", entry.name),
                "metrics": metadata.get("metrics", {}),
                "config": metadata.get("config", {}),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / 1024**2, 1),
                "created": metadata.get("timestamp", metadata.get("created_at")),
            })
        elif entry.is_file() and entry.suffix in (".joblib", ".pkl", ".pt", ".pth", ".h5", ".onnx"):
            # Single-file checkpoint
            size_bytes = entry.stat().st_size
            # Try to extract exp ID from filename
            exp_id = entry.stem.replace("-checkpoint", "").replace("_checkpoint", "")

            checkpoints.append({
                "path": str(entry),
                "name": entry.name,
                "experiment_id": exp_id,
                "metrics": {},
                "config": {},
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / 1024**2, 1),
                "created": None,
            })

    return checkpoints


def enrich_checkpoints_from_log(
    checkpoints: list[dict],
    experiments: list[dict],
) -> list[dict]:
    """Enrich checkpoint metadata with metrics from experiment log."""
    exp_map = {e.get("experiment_id"): e for e in experiments}

    for ckpt in checkpoints:
        exp_id = ckpt["experiment_id"]
        if exp_id in exp_map and not ckpt["metrics"]:
            exp = exp_map[exp_id]
            ckpt["metrics"] = exp.get("metrics", {})
            ckpt["config"] = exp.get("config", {})
            if not ckpt["created"]:
                ckpt["created"] = exp.get("timestamp")

    return checkpoints


def compute_pareto_checkpoints(
    checkpoints: list[dict],
    metrics: list[str],
    directions: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Separate checkpoints into Pareto-optimal and dominated sets.

    Returns (pareto_optimal, dominated) tuple.
    """
    if not checkpoints or not metrics:
        return checkpoints, []

    # Filter to checkpoints that have all requested metrics
    complete = []
    incomplete = []
    for ckpt in checkpoints:
        if all(ckpt["metrics"].get(m) is not None for m in metrics):
            complete.append(ckpt)
        else:
            incomplete.append(ckpt)

    if not complete:
        return checkpoints, []

    pareto = []
    dominated = []

    for i, candidate in enumerate(complete):
        is_dominated = False
        for j, other in enumerate(complete):
            if i == j:
                continue
            all_at_least = True
            strictly_better = False
            for m in metrics:
                c_val = float(candidate["metrics"][m])
                o_val = float(other["metrics"][m])
                direction = directions.get(m, "higher")
                if direction == "higher":
                    if o_val < c_val:
                        all_at_least = False
                        break
                    if o_val > c_val:
                        strictly_better = True
                else:
                    if o_val > c_val:
                        all_at_least = False
                        break
                    if o_val < c_val:
                        strictly_better = True
            if all_at_least and strictly_better:
                is_dominated = True
                break
        if is_dominated:
            dominated.append(candidate)
        else:
            pareto.append(candidate)

    # Incomplete checkpoints are kept (can't determine dominance)
    pareto.extend(incomplete)
    return pareto, dominated


def prune_dominated(
    checkpoints: list[dict],
    dominated: list[dict],
    keep_latest: bool = True,
    dry_run: bool = False,
) -> dict:
    """Remove dominated checkpoints from disk.

    Args:
        checkpoints: All checkpoints.
        dominated: Dominated checkpoints to remove.
        keep_latest: Always keep the most recent checkpoint (for resume safety).
        dry_run: If True, report what would be pruned without deleting.

    Returns:
        Dict with pruned count, bytes saved, and details.
    """
    # Protect the latest checkpoint
    if keep_latest and checkpoints:
        latest = max(checkpoints, key=lambda c: c.get("created") or "")
        dominated = [d for d in dominated if d["path"] != latest["path"]]

    pruned = []
    bytes_saved = 0

    for ckpt in dominated:
        path = Path(ckpt["path"])
        if not path.exists():
            continue
        pruned.append({
            "name": ckpt["name"],
            "experiment_id": ckpt["experiment_id"],
            "size_mb": ckpt["size_mb"],
        })
        bytes_saved += ckpt["size_bytes"]
        if not dry_run:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    return {
        "pruned_count": len(pruned),
        "bytes_saved": bytes_saved,
        "mb_saved": round(bytes_saved / 1024**2, 1),
        "pruned": pruned,
        "dry_run": dry_run,
    }


def compute_disk_stats(checkpoints: list[dict]) -> dict:
    """Compute disk usage statistics for checkpoints."""
    if not checkpoints:
        return {"total_count": 0, "total_size_mb": 0, "avg_size_mb": 0}

    total_bytes = sum(c["size_bytes"] for c in checkpoints)
    return {
        "total_count": len(checkpoints),
        "total_size_mb": round(total_bytes / 1024**2, 1),
        "total_size_gb": round(total_bytes / 1024**3, 2),
        "avg_size_mb": round(total_bytes / len(checkpoints) / 1024**2, 1),
        "largest": max(checkpoints, key=lambda c: c["size_bytes"])["name"] if checkpoints else None,
        "largest_mb": max(c["size_mb"] for c in checkpoints) if checkpoints else 0,
    }


def format_checkpoint_list(
    checkpoints: list[dict],
    pareto_ids: set[str],
    primary_metric: str,
) -> str:
    """Format checkpoint list as markdown table."""
    if not checkpoints:
        return "No checkpoints found."

    lines = [
        "# Checkpoints",
        "",
        f"| Name | Experiment | {primary_metric} | Size | Pareto |",
        f"|------|------------|{'---' * max(len(primary_metric) // 3, 1)}--|------|--------|",
    ]

    for ckpt in checkpoints:
        metric_val = ckpt["metrics"].get(primary_metric)
        metric_str = f"{metric_val:.4f}" if isinstance(metric_val, (int, float)) else "N/A"
        pareto_marker = "YES" if ckpt["path"] in pareto_ids or ckpt["name"] in pareto_ids else ""
        lines.append(
            f"| {ckpt['name']} | {ckpt['experiment_id']} "
            f"| {metric_str} | {ckpt['size_mb']:.1f} MB | {pareto_marker} |"
        )

    return "\n".join(lines)


def format_prune_report(prune_result: dict, stats_before: dict, stats_after: dict) -> str:
    """Format pruning report."""
    lines = [
        "# Checkpoint Pruning",
        "",
        f"Before: {stats_before['total_count']} checkpoints, {stats_before.get('total_size_gb', 0):.1f} GB",
    ]

    if prune_result["dry_run"]:
        lines.append(f"Would prune: {prune_result['pruned_count']} dominated checkpoints ({prune_result['mb_saved']:.1f} MB)")
        lines.append("")
        lines.append("*Dry run — no files deleted. Run without --dry-run to prune.*")
    else:
        lines.append(f"Pruned: {prune_result['pruned_count']} dominated checkpoints ({prune_result['mb_saved']:.1f} MB)")
        lines.append(f"After: {stats_after['total_count']} checkpoints, {stats_after.get('total_size_gb', 0):.1f} GB")

    if prune_result["pruned"]:
        lines.extend(["", "## Removed", ""])
        for p in prune_result["pruned"]:
            lines.append(f"- {p['name']} ({p['experiment_id']}, {p['size_mb']:.1f} MB)")

    return "\n".join(lines)


def format_stats_report(stats: dict, checkpoints: list[dict]) -> str:
    """Format disk usage statistics."""
    lines = [
        "# Checkpoint Storage",
        "",
        f"- **Total checkpoints:** {stats['total_count']}",
        f"- **Total size:** {stats.get('total_size_gb', 0):.2f} GB ({stats['total_size_mb']:.1f} MB)",
        f"- **Average size:** {stats['avg_size_mb']:.1f} MB",
    ]
    if stats.get("largest"):
        lines.append(f"- **Largest:** {stats['largest']} ({stats['largest_mb']:.1f} MB)")

    # Size by model type
    by_type: dict[str, list[dict]] = {}
    for ckpt in checkpoints:
        mt = ckpt.get("config", {}).get("model_type", "unknown")
        by_type.setdefault(mt, []).append(ckpt)

    if len(by_type) > 1:
        lines.extend(["", "## By Model Type", ""])
        for mt, ckpts in sorted(by_type.items()):
            total_mb = sum(c["size_mb"] for c in ckpts)
            lines.append(f"- **{mt}:** {len(ckpts)} checkpoints, {total_mb:.1f} MB")

    return "\n".join(lines)


def save_checkpoint_report(report: dict, output_dir: str = "experiments/checkpoints") -> Path:
    """Save checkpoint management report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filepath = out_path / "checkpoint-report.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Smart checkpoint manager")
    parser.add_argument(
        "action",
        choices=["list", "prune", "average", "resume", "stats"],
        help="Action to perform",
    )
    parser.add_argument("exp_id", nargs="?", default=None, help="Experiment ID (for resume)")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl", help="Path to experiment log")
    parser.add_argument("--top", type=int, default=3, help="Top K for averaging")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually delete (prune)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)

    experiments = load_experiments(args.log)
    checkpoints = scan_checkpoints(args.checkpoint_dir)
    checkpoints = enrich_checkpoints_from_log(checkpoints, experiments)

    # Determine metric directions
    lower_metrics = {"train_seconds", "latency", "mse", "rmse", "mae", "loss"}
    metrics_to_check = [primary_metric]
    if "train_seconds" in set().union(*(c["metrics"].keys() for c in checkpoints if c["metrics"])):
        metrics_to_check.append("train_seconds")

    directions = {}
    for m in metrics_to_check:
        if m == primary_metric:
            directions[m] = "lower" if lower_is_better else "higher"
        elif m in lower_metrics:
            directions[m] = "lower"
        else:
            directions[m] = "higher"

    pareto, dominated = compute_pareto_checkpoints(checkpoints, metrics_to_check, directions)
    pareto_paths = {c["path"] for c in pareto}

    if args.action == "list":
        if args.json:
            print(json.dumps(checkpoints, indent=2, default=str))
        else:
            print(format_checkpoint_list(checkpoints, pareto_paths, primary_metric))

    elif args.action == "stats":
        stats = compute_disk_stats(checkpoints)
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(format_stats_report(stats, checkpoints))

    elif args.action == "prune":
        stats_before = compute_disk_stats(checkpoints)
        result = prune_dominated(checkpoints, dominated, dry_run=args.dry_run)

        if not args.dry_run:
            remaining = scan_checkpoints(args.checkpoint_dir)
            stats_after = compute_disk_stats(remaining)
        else:
            stats_after = {
                "total_count": stats_before["total_count"] - result["pruned_count"],
                "total_size_mb": stats_before["total_size_mb"] - result["mb_saved"],
                "total_size_gb": round((stats_before["total_size_mb"] - result["mb_saved"]) / 1024, 2),
            }

        if args.json:
            print(json.dumps({"prune": result, "before": stats_before, "after": stats_after}, indent=2))
        else:
            print(format_prune_report(result, stats_before, stats_after))

    elif args.action == "average":
        # Sort by primary metric and take top K
        with_metric = [c for c in checkpoints if primary_metric in c["metrics"]]
        with_metric.sort(
            key=lambda c: c["metrics"][primary_metric],
            reverse=not lower_is_better,
        )
        top_k = with_metric[:args.top]

        if not top_k:
            print("No checkpoints with metric data found.", file=sys.stderr)
            sys.exit(1)

        print(f"Top {len(top_k)} checkpoints for averaging:", file=sys.stderr)
        for c in top_k:
            print(f"  {c['experiment_id']}: {primary_metric}={c['metrics'][primary_metric]:.4f}", file=sys.stderr)
        print("\nNote: Weight averaging requires model-specific implementation.", file=sys.stderr)
        print("The checkpoint paths are:", file=sys.stderr)
        for c in top_k:
            print(f"  {c['path']}", file=sys.stderr)

    elif args.action == "resume":
        if not args.exp_id:
            print("Specify experiment ID: checkpoint resume <exp-id>", file=sys.stderr)
            sys.exit(1)

        target = None
        for c in checkpoints:
            if c["experiment_id"] == args.exp_id:
                target = c
                break

        if not target:
            print(f"No checkpoint found for {args.exp_id}", file=sys.stderr)
            available = [c["experiment_id"] for c in checkpoints]
            if available:
                print(f"Available: {', '.join(available[:10])}", file=sys.stderr)
            sys.exit(1)

        print(f"Resume checkpoint: {target['path']}", file=sys.stderr)
        print(f"Experiment: {target['experiment_id']}", file=sys.stderr)
        if target["metrics"]:
            print(f"Metrics: {target['metrics']}", file=sys.stderr)


if __name__ == "__main__":
    main()
