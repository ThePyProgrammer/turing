#!/usr/bin/env python3
"""Hypothesis queue manager for the autoresearch pipeline.

Manages a structured queue of hypotheses — ideas for experiments
that can be injected by the human (via /turing:try) or generated
by the agent. Human-injected hypotheses take priority.

This is the mechanism by which research taste reaches the agent:
the human selects which coins to flip, the agent flips them.

Usage:
    python scripts/manage_hypotheses.py add "description" [--priority high] [--parent exp-NNN]
    python scripts/manage_hypotheses.py list [--status queued]
    python scripts/manage_hypotheses.py next
    python scripts/manage_hypotheses.py mark <id> <status> [--result exp-NNN]
    python scripts/manage_hypotheses.py count
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

DEFAULT_QUEUE_PATH = "hypotheses.yaml"
DETAIL_DIR = "hypotheses"

VALID_STATUSES = {"queued", "in-progress", "tested", "promising", "dead-end"}
VALID_PRIORITIES = {"high", "medium", "low"}


def load_queue(path: str) -> list[dict]:
    """Load hypothesis queue from YAML file."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def save_queue(path: str, queue: list[dict]) -> None:
    """Save hypothesis queue to YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False)


def get_next_id(queue: list[dict]) -> str:
    """Get the next sequential hypothesis ID."""
    if not queue:
        return "hyp-001"
    max_num = 0
    for entry in queue:
        hid = entry.get("id", "")
        if hid.startswith("hyp-"):
            try:
                num = int(hid.split("-")[1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue
    return f"hyp-{max_num + 1:03d}"


def create_detail_file(
    hid: str,
    description: str,
    source: str = "human",
    priority: str = "high",
    parent_experiment: str | None = None,
    parent_hypothesis: str | None = None,
    family: str | None = None,
    tags: list[str] | None = None,
    architecture: dict | None = None,
    hyperparameters: dict | None = None,
    features: dict | None = None,
    expected_outcome: dict | None = None,
) -> Path:
    """Create a detailed hypothesis file at hypotheses/hyp-NNN.yaml."""
    detail_dir = Path(DETAIL_DIR)
    detail_dir.mkdir(parents=True, exist_ok=True)

    detail = {
        "id": hid,
        "description": description,
        "source": source,
        "status": "queued",
        "priority": priority,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": architecture or {},
        "hyperparameters": hyperparameters or {},
        "features": features or {"add": [], "remove": [], "transform": []},
        "expected_outcome": expected_outcome or {},
        "result": {
            "experiment_id": None,
            "metrics": {},
            "verdict": None,
            "notes": None,
        },
        "parent_experiment": parent_experiment,
        "parent_hypothesis": parent_hypothesis,
        "family": family,
        "tags": tags or [],
    }

    detail_path = detail_dir / f"{hid}.yaml"
    with open(detail_path, "w") as f:
        yaml.dump(detail, f, default_flow_style=False, sort_keys=False)
    return detail_path


def load_detail(hid: str) -> dict | None:
    """Load a detailed hypothesis file."""
    detail_path = Path(DETAIL_DIR) / f"{hid}.yaml"
    if not detail_path.exists():
        return None
    with open(detail_path) as f:
        return yaml.safe_load(f) or None


def update_detail(hid: str, updates: dict) -> bool:
    """Update fields in a detailed hypothesis file."""
    detail = load_detail(hid)
    if detail is None:
        return False

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(detail.get(key), dict):
            detail[key].update(value)
        else:
            detail[key] = value

    detail_path = Path(DETAIL_DIR) / f"{hid}.yaml"
    with open(detail_path, "w") as f:
        yaml.dump(detail, f, default_flow_style=False, sort_keys=False)
    return True


def add_hypothesis(
    queue_path: str,
    description: str,
    source: str = "human",
    priority: str = "high",
    parent_experiment: str | None = None,
    parent_hypothesis: str | None = None,
    family: str | None = None,
    tags: list[str] | None = None,
    architecture: dict | None = None,
    hyperparameters: dict | None = None,
    features: dict | None = None,
    expected_outcome: dict | None = None,
) -> str:
    """Add a hypothesis to the queue and create its detail file.

    Returns the new hypothesis ID.
    """
    queue = load_queue(queue_path)
    hid = get_next_id(queue)

    # Index entry (lightweight)
    entry = {
        "id": hid,
        "description": description,
        "source": source,
        "status": "queued",
        "priority": priority,
        "parent_experiment": parent_experiment,
        "result_experiment": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    queue.append(entry)
    save_queue(queue_path, queue)

    # Detail file (rich)
    create_detail_file(
        hid=hid,
        description=description,
        source=source,
        priority=priority,
        parent_experiment=parent_experiment,
        parent_hypothesis=parent_hypothesis,
        family=family,
        tags=tags,
        architecture=architecture,
        hyperparameters=hyperparameters,
        features=features,
        expected_outcome=expected_outcome,
    )

    return hid


def list_hypotheses(queue_path: str, status_filter: str | None = None) -> list[dict]:
    """List hypotheses, optionally filtered by status."""
    queue = load_queue(queue_path)
    if status_filter:
        queue = [h for h in queue if h.get("status") == status_filter]
    return queue


def get_next_hypothesis(queue_path: str) -> dict | None:
    """Get the next queued hypothesis, prioritizing high > medium > low.

    Within the same priority, human-sourced hypotheses come before agent-sourced.
    Within the same source, earlier hypotheses come first (FIFO).
    """
    queue = load_queue(queue_path)
    queued = [h for h in queue if h.get("status") == "queued"]
    if not queued:
        return None

    priority_order = {"high": 0, "medium": 1, "low": 2}
    source_order = {"human": 0, "literature": 1, "taxonomy": 2, "agent": 3}

    queued.sort(key=lambda h: (
        priority_order.get(h.get("priority", "medium"), 1),
        source_order.get(h.get("source", "agent"), 1),
    ))
    return queued[0]


def mark_hypothesis(
    queue_path: str,
    hypothesis_id: str,
    new_status: str,
    result_experiment: str | None = None,
    result_metrics: dict | None = None,
    result_notes: str | None = None,
) -> bool:
    """Update a hypothesis status in both the index and detail file.

    Returns True if found and updated.
    """
    if new_status not in VALID_STATUSES:
        print(f"Invalid status: {new_status}. Valid: {', '.join(sorted(VALID_STATUSES))}", file=sys.stderr)
        return False

    # Update index
    queue = load_queue(queue_path)
    found = False
    for entry in queue:
        if entry.get("id") == hypothesis_id:
            entry["status"] = new_status
            if result_experiment:
                entry["result_experiment"] = result_experiment
            save_queue(queue_path, queue)
            found = True
            break

    if not found:
        return False

    # Update detail file
    detail_updates = {"status": new_status}
    if result_experiment or result_metrics or result_notes:
        result_update = {}
        if result_experiment:
            result_update["experiment_id"] = result_experiment
        if result_metrics:
            result_update["metrics"] = result_metrics
        if result_notes:
            result_update["notes"] = result_notes
        verdict_map = {"tested": "tested", "promising": "promising", "dead-end": "dead-end"}
        if new_status in verdict_map:
            result_update["verdict"] = verdict_map[new_status]
        detail_updates["result"] = result_update

    update_detail(hypothesis_id, detail_updates)
    return True


def count_by_status(queue_path: str) -> dict[str, int]:
    """Count hypotheses by status."""
    queue = load_queue(queue_path)
    counts: dict[str, int] = {}
    for entry in queue:
        status = entry.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def format_table(hypotheses: list[dict]) -> str:
    """Format hypotheses as a text table."""
    if not hypotheses:
        return "No hypotheses in queue."

    header = f"{'ID':<10} {'Status':<12} {'Priority':<8} {'Source':<8} {'Description'}"
    sep = "-" * 80
    lines = [header, sep]

    for h in hypotheses:
        desc = h.get("description", "")[:45]
        line = f"{h.get('id', '?'):<10} {h.get('status', '?'):<12} {h.get('priority', '?'):<8} {h.get('source', '?'):<8} {desc}"
        if h.get("result_experiment"):
            line += f" -> {h['result_experiment']}"
        lines.append(line)

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage hypothesis queue")
    parser.add_argument("--queue", default=DEFAULT_QUEUE_PATH, help="Path to hypotheses.yaml")
    subparsers = parser.add_subparsers(dest="command")

    # add
    add_parser = subparsers.add_parser("add", help="Add a hypothesis")
    add_parser.add_argument("description", help="What to try and why")
    add_parser.add_argument("--priority", default="high", choices=sorted(VALID_PRIORITIES))
    add_parser.add_argument("--source", default="human", choices=["human", "agent", "literature", "taxonomy"])
    add_parser.add_argument("--parent", default=None, help="Parent experiment ID")
    add_parser.add_argument("--parent-hyp", default=None, help="Parent hypothesis ID")
    add_parser.add_argument("--family", default=None, help="Experiment family (e.g., optimizer-sweep)")
    add_parser.add_argument("--tags", default=None, help="Comma-separated tags")
    add_parser.add_argument("--model-type", default=None, help="Proposed model type")
    add_parser.add_argument("--hyperparams", default=None, help="JSON string of hyperparameters")
    add_parser.add_argument("--expected", default=None, help="Expected outcome description")

    # list
    list_parser = subparsers.add_parser("list", help="List hypotheses")
    list_parser.add_argument("--status", default=None, choices=sorted(VALID_STATUSES))

    # next
    subparsers.add_parser("next", help="Get next queued hypothesis")

    # show
    show_parser = subparsers.add_parser("show", help="Show detailed hypothesis file")
    show_parser.add_argument("id", help="Hypothesis ID")

    # mark
    mark_parser = subparsers.add_parser("mark", help="Update hypothesis status")
    mark_parser.add_argument("id", help="Hypothesis ID")
    mark_parser.add_argument("status", choices=sorted(VALID_STATUSES))
    mark_parser.add_argument("--result", default=None, help="Result experiment ID")
    mark_parser.add_argument("--metrics", default=None, help="JSON string of result metrics")
    mark_parser.add_argument("--notes", default=None, help="Notes about the result")

    # count
    subparsers.add_parser("count", help="Count hypotheses by status")

    args = parser.parse_args()

    if args.command == "add":
        hid = add_hypothesis(args.queue, args.description, args.source, args.priority, args.parent)
        print(f"Added {hid}: {args.description}")

    elif args.command == "list":
        hypotheses = list_hypotheses(args.queue, args.status)
        print(format_table(hypotheses))

    elif args.command == "next":
        h = get_next_hypothesis(args.queue)
        if h:
            import json
            print(json.dumps(h, indent=2))
        else:
            print("No queued hypotheses.", file=sys.stderr)
            sys.exit(1)

    elif args.command == "mark":
        found = mark_hypothesis(args.queue, args.id, args.status, args.result)
        if found:
            print(f"Marked {args.id} as {args.status}")
        else:
            print(f"Hypothesis {args.id} not found.", file=sys.stderr)
            sys.exit(1)

    elif args.command == "count":
        counts = count_by_status(args.queue)
        total = sum(counts.values())
        print(f"Total: {total}")
        for status, count in sorted(counts.items()):
            print(f"  {status}: {count}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
