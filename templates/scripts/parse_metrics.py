#!/usr/bin/env python3
"""Canonical metric parser for the autoresearch pipeline.

Single source of truth for parsing the --- delimited metric format
produced by evaluate.py's format_metrics(). Replaces three independent
parsers (agent grep, bash awk, stop-hook reader) with one testable module.

The format:
    ---
    metric_name:     value
    ...
    model_type:      xgboost
    train_seconds:   2.5
    ---

Metadata keys (model_type, train_seconds) are separated from metric keys.

Usage:
    python scripts/parse_metrics.py <run.log>             # Print parsed JSON
    python scripts/parse_metrics.py <run.log> --metrics    # Metrics only (no metadata)
    python scripts/parse_metrics.py <run.log> --raw        # Full dict including metadata

As a library:
    from scripts.parse_metrics import parse_run_log
    metrics, metadata = parse_run_log("run.log")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METADATA_KEYS = {"model_type", "train_seconds"}


def parse_metrics_block(text: str) -> tuple[dict[str, float], dict[str, str]]:
    """Parse a --- delimited metrics block into metrics and metadata.

    Args:
        text: String containing the --- delimited block (may have surrounding text).

    Returns:
        Tuple of (metrics_dict, metadata_dict).
        metrics_dict has float values. metadata_dict has string values.
    """
    lines = text.strip().split("\n")
    metrics: dict[str, float] = {}
    metadata: dict[str, str] = {}
    in_block = False

    for line in lines:
        line = line.strip()
        if line == "---":
            if in_block:
                break  # end of block
            in_block = True
            continue

        if in_block and ":" in line:
            # Split on first colon only (handles values with colons)
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if not key or not value:
                continue

            if key in METADATA_KEYS:
                metadata[key] = value
            else:
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metadata[key] = value

    return metrics, metadata


def parse_run_log(log_path: str) -> tuple[dict[str, float], dict[str, str]]:
    """Parse metrics from a run.log file.

    Args:
        log_path: Path to the run.log file.

    Returns:
        Tuple of (metrics_dict, metadata_dict).

    Raises:
        FileNotFoundError: If log_path does not exist.
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    text = path.read_text(encoding="utf-8")
    return parse_metrics_block(text)


def metrics_to_json(metrics: dict[str, float], metadata: dict[str, str]) -> tuple[str, str]:
    """Convert parsed metrics and metadata to JSON strings.

    Returns (metrics_json, config_json) suitable for log_experiment.py CLI.
    """
    metrics_json = json.dumps(metrics)
    config_json = json.dumps(metadata)
    return metrics_json, config_json


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Parse metrics from run.log")
    parser.add_argument("log_file", help="Path to run.log")
    parser.add_argument("--metrics", action="store_true", help="Print metrics only (no metadata)")
    parser.add_argument("--raw", action="store_true", help="Print full dict including metadata")
    args = parser.parse_args()

    try:
        metrics, metadata = parse_run_log(args.log_file)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if not metrics and not metadata:
        print("No metrics block found.", file=sys.stderr)
        sys.exit(1)

    if args.metrics:
        print(json.dumps(metrics, indent=2))
    elif args.raw:
        combined = {**metrics, **metadata}
        print(json.dumps(combined, indent=2))
    else:
        print(json.dumps({"metrics": metrics, "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
