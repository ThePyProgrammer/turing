#!/usr/bin/env python3
"""Experiment dependency tree visualizer.

Displays the experiment lineage — which experiments inspired which —
as a text tree. Makes the agent's reasoning chain visible to the human.

Usage:
    python scripts/show_experiment_tree.py [--log experiments/log.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.turing_io import load_experiments


def build_tree(experiments: list[dict]) -> dict[str, list[str]]:
    """Build parent -> children mapping from experiments."""
    children: dict[str, list[str]] = {}
    for exp in experiments:
        eid = exp.get("experiment_id", "")
        parent = exp.get("parent_experiment")
        if parent:
            children.setdefault(parent, []).append(eid)
        # Ensure every experiment appears as a key
        children.setdefault(eid, [])
    return children


def find_roots(experiments: list[dict], children: dict[str, list[str]]) -> list[str]:
    """Find experiments with no parent (tree roots)."""
    all_ids = {e.get("experiment_id", "") for e in experiments}
    child_ids = set()
    for kids in children.values():
        child_ids.update(kids)

    # Roots are experiments that are not children of anything
    # But also need to handle experiments whose parent doesn't exist in the log
    roots = []
    for exp in experiments:
        eid = exp.get("experiment_id", "")
        parent = exp.get("parent_experiment")
        if parent is None or parent not in all_ids:
            roots.append(eid)
    return roots


def format_tree(
    node: str,
    children: dict[str, list[str]],
    experiments_by_id: dict[str, dict],
    prefix: str = "",
    is_last: bool = True,
    metric_name: str = "accuracy",
) -> list[str]:
    """Recursively format a tree node and its children."""
    exp = experiments_by_id.get(node, {})
    status = exp.get("status", "?")
    metric_val = exp.get("metrics", {}).get(metric_name)
    model_type = exp.get("config", {}).get("model_type", "?")
    desc = exp.get("description", "")[:40]

    status_marker = "kept" if status == "kept" else "discarded" if status == "discarded" else status
    metric_str = f"{metric_name}={metric_val:.4f}" if isinstance(metric_val, (int, float)) else ""
    status_icon = "+" if status == "kept" else "-" if status == "discarded" else "?"

    connector = "`-- " if is_last else "|-- "
    line = f"{prefix}{connector}[{status_icon}] {node} ({model_type}, {metric_str}) {desc}"

    lines = [line]

    child_prefix = prefix + ("    " if is_last else "|   ")
    kids = children.get(node, [])
    for i, child in enumerate(kids):
        child_lines = format_tree(
            child, children, experiments_by_id, child_prefix,
            is_last=(i == len(kids) - 1), metric_name=metric_name,
        )
        lines.extend(child_lines)

    return lines


def show_tree(log_path: str, metric_name: str = "accuracy") -> str:
    """Generate the full experiment tree as text."""
    experiments = load_experiments(log_path)
    if not experiments:
        return "No experiments logged yet."

    experiments_by_id = {e.get("experiment_id", ""): e for e in experiments}
    children = build_tree(experiments)
    roots = find_roots(experiments, children)

    if not roots:
        return "No root experiments found."

    lines = ["Experiment Tree", "=" * 60]
    for i, root in enumerate(roots):
        root_lines = format_tree(
            root, children, experiments_by_id,
            prefix="", is_last=(i == len(roots) - 1),
            metric_name=metric_name,
        )
        lines.extend(root_lines)

    # Summary
    total = len(experiments)
    kept = sum(1 for e in experiments if e.get("status") == "kept")
    max_depth = _max_depth(roots, children)
    lines.extend([
        "",
        f"Total: {total} experiments, {kept} kept, depth={max_depth}",
        f"[+] = kept, [-] = discarded, [?] = other",
    ])

    return "\n".join(lines)


def _max_depth(roots: list[str], children: dict[str, list[str]], depth: int = 1) -> int:
    """Compute maximum tree depth."""
    max_d = depth
    for root in roots:
        kids = children.get(root, [])
        if kids:
            max_d = max(max_d, _max_depth(kids, children, depth + 1))
    return max_d


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Show experiment dependency tree")
    parser.add_argument("--log", default="experiments/log.jsonl")
    parser.add_argument("--metric", default="accuracy", help="Metric to display")
    args = parser.parse_args()

    print(show_tree(args.log, args.metric))


if __name__ == "__main__":
    main()
