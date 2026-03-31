"""Shared data loading functions for the autoresearch pipeline.

Consolidates the duplicated load_experiments, load_config, and
load_hypotheses functions that were copy-pasted across 8+ scripts.
Every script that reads experiment logs, config, or hypothesis
queues should import from here.

Usage:
    from scripts.turing_io import load_experiments, load_config, load_hypotheses
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_experiments(log_path: str) -> list[dict]:
    """Load experiments from JSONL log.

    Args:
        log_path: Path to the experiments JSONL file.

    Returns:
        List of experiment dicts. Empty list if file missing or empty.
        Malformed lines are silently skipped.
    """
    path = Path(log_path)
    if not path.exists():
        return []
    experiments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    experiments.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return experiments


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config with fallback to empty dict.

    Args:
        path: Path to config YAML file. Defaults to "config.yaml".

    Returns:
        Parsed config dict, or empty dict if file missing or invalid.
    """
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def load_hypotheses(queue_path: str) -> list[dict]:
    """Load hypothesis queue from YAML file.

    Args:
        queue_path: Path to the hypotheses YAML file.

    Returns:
        List of hypothesis dicts. Empty list if file missing, empty,
        or not a list.
    """
    path = Path(queue_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []
