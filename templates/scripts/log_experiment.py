"""Experiment logging utility for the autoresearch pipeline.

The ground truth record of all experiments — kept and discarded alike.
Appends structured JSONL entries to experiments/log.jsonl with full
metadata: experiment_id, timestamp, git_commit, status, config, metrics,
model_path, and description.

Also maintains a TSV summary at experiments/results.tsv for quick reference.
Every experiment is logged. No information is lost.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_next_experiment_id(log_path: str) -> str:
    """Get the next sequential experiment ID from the log.

    Returns "exp-001" for empty/nonexistent log, "exp-NNN" otherwise.
    """
    path = Path(log_path)
    if not path.exists() or path.stat().st_size == 0:
        return "exp-001"

    max_id = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                exp_id = entry.get("experiment_id", "")
                if exp_id.startswith("exp-"):
                    num = int(exp_id.split("-")[1])
                    max_id = max(max_id, num)
            except (json.JSONDecodeError, ValueError, IndexError):
                continue

    return f"exp-{max_id + 1:03d}"


def log_tsv_row(
    tsv_path: str,
    experiment_id: str,
    status: str,
    config: dict,
    metrics: dict,
    description: str,
) -> None:
    """Append a TSV row to the results summary file.

    Creates header row if file doesn't exist or is empty.
    Columns: experiment_id, status, model_type, primary_metric, description, timestamp
    """
    path = Path(tsv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0

    # Build metric columns dynamically
    metric_cols = "\t".join(str(v) for v in metrics.values())
    metric_headers = "\t".join(metrics.keys())

    row_data = [
        experiment_id,
        status,
        config.get("model_type", "unknown"),
        metric_cols,
        description.replace("\t", " "),
        datetime.now(timezone.utc).isoformat(),
    ]

    with open(path, "a") as f:
        if write_header:
            f.write(f"experiment_id\tstatus\tmodel_type\t{metric_headers}\tdescription\ttimestamp\n")
        f.write("\t".join(row_data) + "\n")


def log_experiment(
    log_path: str,
    experiment_id: str,
    config: dict,
    metrics: dict,
    model_path: str,
    description: str,
    status: str = "kept",
    git_commit: str | None = None,
    parent_experiment: str | None = None,
    hypothesis_id: str | None = None,
    family: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Append one experiment entry to the JSONL log.

    Args:
        log_path: Path to experiments/log.jsonl.
        experiment_id: e.g. "exp-001".
        config: Dict with model_type, hyperparams, features.
        metrics: Dict with metric values.
        model_path: Path to saved model artifact.
        description: Human-readable experiment description.
        status: "kept" or "discarded".
        git_commit: Optional git commit hash.
        parent_experiment: Optional parent experiment ID (for dependency tree).
        hypothesis_id: Optional hypothesis ID (links to hypotheses.yaml).
        family: Optional experiment family for strategic grouping.
        tags: Optional list of tags for categorization.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load environment snapshot from train_metadata.json if available
    environment = None
    metadata_path = Path("train_metadata.json")
    if metadata_path.exists():
        try:
            with open(metadata_path) as mf:
                meta = json.load(mf)
                environment = meta.get("environment")
        except (json.JSONDecodeError, OSError):
            pass

    entry = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "status": status,
        "parent_experiment": parent_experiment,
        "hypothesis_id": hypothesis_id,
        "family": family,
        "tags": tags or [],
        "config": config,
        "metrics": metrics,
        "model_path": model_path,
        "description": description,
        "environment": environment,
    }

    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Also append TSV summary row
    tsv_path = str(Path(log_path).parent / "results.tsv")
    log_tsv_row(tsv_path, experiment_id, status, config, metrics, description)


def get_best_experiment(log_path: str, primary_metric: str = "accuracy", lower_is_better: bool = False) -> dict | None:
    """Return the experiment with the best primary metric among 'keep' entries.

    Args:
        log_path: Path to experiments/log.jsonl.
        primary_metric: Metric name to optimize.
        lower_is_better: True for metrics like MAE/MSE, False for accuracy/F1.

    Returns:
        Best experiment dict, or None if no 'keep' entries exist.
    """
    path = Path(log_path)
    if not path.exists():
        return None

    best = None
    best_value = float("inf") if lower_is_better else float("-inf")

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("status") != "kept":
                continue

            value = entry.get("metrics", {}).get(primary_metric)
            if value is None:
                continue

            if lower_is_better and value < best_value:
                best_value = value
                best = entry
            elif not lower_is_better and value > best_value:
                best_value = value
                best = entry

    return best


def main() -> None:
    """CLI entry point for logging experiments."""
    parser = argparse.ArgumentParser(
        description="Log an experiment to the append-only JSONL experiment log.",
    )
    parser.add_argument("log_path", help="Path to experiments/log.jsonl")
    parser.add_argument("experiment_id", help='Experiment ID, e.g. "exp-001"')
    parser.add_argument("status", help='"kept" or "discarded"')
    parser.add_argument("metrics_json", help="Metrics as a JSON string")
    parser.add_argument("config_json", help="Config as a JSON string")
    parser.add_argument("model_path", help="Path to saved model artifact")
    parser.add_argument(
        "description",
        nargs="*",
        help="Human-readable experiment description",
    )
    parser.add_argument("--parent", default=None, help="Parent experiment ID")
    parser.add_argument("--hypothesis", default=None, help="Hypothesis ID")
    parser.add_argument("--family", default=None, help="Experiment family name")
    parser.add_argument("--tags", default=None, help="Comma-separated tags")

    args = parser.parse_args()

    metrics = json.loads(args.metrics_json)
    config = json.loads(args.config_json)
    description = " ".join(args.description) if args.description else ""
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None

    log_experiment(
        log_path=args.log_path,
        experiment_id=args.experiment_id,
        config=config,
        metrics=metrics,
        model_path=args.model_path,
        description=description,
        status=args.status,
        parent_experiment=args.parent,
        hypothesis_id=args.hypothesis,
        family=args.family,
        tags=tags,
    )
    print(f"Logged {args.experiment_id} ({args.status})")


if __name__ == "__main__":
    main()
