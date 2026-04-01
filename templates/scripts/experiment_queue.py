#!/usr/bin/env python3
"""Batch experiment scheduler with priority ordering and dependencies.

Queue multiple experiments for unattended execution. The researcher
loads the queue Friday afternoon, reads /turing:brief Monday morning.

Usage:
    python scripts/experiment_queue.py add "try LightGBM" --priority high
    python scripts/experiment_queue.py add "deeper trees" --after q-001
    python scripts/experiment_queue.py list
    python scripts/experiment_queue.py run [--halt-on-error]
    python scripts/experiment_queue.py pause
    python scripts/experiment_queue.py clear
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


DEFAULT_QUEUE_PATH = "experiments/queue.yaml"
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def load_queue(queue_path: str = DEFAULT_QUEUE_PATH) -> list[dict]:
    """Load the experiment queue from YAML."""
    path = Path(queue_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def save_queue(queue: list[dict], queue_path: str = DEFAULT_QUEUE_PATH) -> None:
    """Save the experiment queue to YAML."""
    path = Path(queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False)


def get_next_queue_id(queue: list[dict]) -> str:
    """Generate next sequential queue ID."""
    max_id = 0
    for item in queue:
        qid = item.get("id", "")
        if qid.startswith("q-"):
            try:
                num = int(qid.split("-")[1])
                max_id = max(max_id, num)
            except (ValueError, IndexError):
                pass
    return f"q-{max_id + 1:03d}"


def add_to_queue(
    description: str,
    priority: str = "medium",
    after: str | None = None,
    hypothesis_id: str | None = None,
    queue_path: str = DEFAULT_QUEUE_PATH,
) -> dict:
    """Add an experiment to the queue.

    Args:
        description: What to try.
        priority: critical/high/medium/low.
        after: Queue ID this depends on (runs after that item completes).
        hypothesis_id: Link to hypothesis queue entry.
        queue_path: Path to queue YAML.

    Returns:
        The created queue item dict.
    """
    queue = load_queue(queue_path)
    qid = get_next_queue_id(queue)

    item = {
        "id": qid,
        "description": description,
        "priority": priority,
        "status": "queued",
        "depends_on": after,
        "hypothesis_id": hypothesis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "result_experiment": None,
        "error": None,
        "retries": 0,
    }

    queue.append(item)
    save_queue(queue, queue_path)
    return item


def sort_queue(queue: list[dict]) -> list[dict]:
    """Sort queue by priority then creation time, respecting dependencies.

    Returns items in execution order: dependencies first, then by priority.
    """
    queued = [q for q in queue if q.get("status") == "queued"]

    # Topological sort with priority
    resolved = []
    remaining = list(queued)
    resolved_ids = set()

    max_iterations = len(remaining) * 2
    iteration = 0
    while remaining and iteration < max_iterations:
        iteration += 1
        progress = False
        for item in list(remaining):
            dep = item.get("depends_on")
            if dep is None or dep in resolved_ids:
                resolved.append(item)
                resolved_ids.add(item["id"])
                remaining.remove(item)
                progress = True
        if not progress:
            # Circular dependency — add remaining in priority order
            remaining.sort(key=lambda x: PRIORITY_ORDER.get(x.get("priority", "medium"), 2))
            resolved.extend(remaining)
            break

    # Sort by priority within dependency levels
    resolved.sort(key=lambda x: PRIORITY_ORDER.get(x.get("priority", "medium"), 2))
    return resolved


def estimate_runtime(queue: list[dict], profile_dir: str = "experiments/profiles") -> float:
    """Estimate total runtime for queued items from profile data."""
    path = Path(profile_dir)
    if not path.exists():
        return 0.0

    # Get average runtime from profiles
    runtimes = []
    for f in path.glob("*-profile.yaml"):
        try:
            with open(f) as fh:
                profile = yaml.safe_load(fh)
                if profile and isinstance(profile, dict):
                    p = profile.get("profile", {})
                    rt = p.get("total_time_sec", 0)
                    if rt > 0:
                        runtimes.append(rt)
        except (yaml.YAMLError, OSError):
            continue

    if not runtimes:
        return 0.0

    avg_runtime = sum(runtimes) / len(runtimes)
    n_queued = sum(1 for q in queue if q.get("status") == "queued")
    return avg_runtime * n_queued


def run_queue_item(item: dict, timeout: int = 600) -> dict:
    """Execute a single queue item.

    Returns updated item dict with status, result, timing.
    """
    item["status"] = "running"
    item["started_at"] = datetime.now(timezone.utc).isoformat()

    cmd = ["python", "train.py", "--seed", "42"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        item["status"] = "failed"
        item["error"] = "timeout"
        item["completed_at"] = datetime.now(timezone.utc).isoformat()
        return item

    if proc.returncode != 0:
        item["status"] = "failed"
        item["error"] = _classify_error(proc.stderr + proc.stdout)
        item["completed_at"] = datetime.now(timezone.utc).isoformat()
        return item

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
            key, value = line.split(":", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                metrics[key.strip()] = value.strip()

    item["status"] = "completed"
    item["completed_at"] = datetime.now(timezone.utc).isoformat()
    item["result_metrics"] = metrics
    return item


def _classify_error(output: str) -> str:
    """Classify error from output text."""
    output_lower = output.lower()
    if "cuda out of memory" in output_lower or "memoryerror" in output_lower:
        return "oom"
    if "nan" in output_lower and ("loss" in output_lower or "nan" in output):
        return "nan_loss"
    if "timeouterror" in output_lower:
        return "timeout"
    if "modulenotfounderror" in output_lower or "importerror" in output_lower:
        return "import_error"
    return "unknown"


def run_queue(
    queue_path: str = DEFAULT_QUEUE_PATH,
    halt_on_error: bool = False,
    timeout: int = 600,
) -> dict:
    """Execute all queued experiments in order.

    Args:
        queue_path: Path to queue YAML.
        halt_on_error: Stop on first failure.
        timeout: Per-experiment timeout.

    Returns:
        Batch summary dict.
    """
    queue = load_queue(queue_path)
    execution_order = sort_queue(queue)

    if not execution_order:
        return {"status": "empty", "message": "No queued experiments."}

    summary = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total": len(execution_order),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "results": [],
    }

    print(f"Running {len(execution_order)} queued experiments...", file=sys.stderr)

    for i, item in enumerate(execution_order):
        # Check if paused
        current_queue = load_queue(queue_path)
        paused = any(q.get("_paused") for q in current_queue)
        if paused:
            summary["status"] = "paused"
            summary["skipped"] = len(execution_order) - i
            break

        print(f"\n  [{i+1}/{len(execution_order)}] {item['id']}: {item['description']}", file=sys.stderr)

        result = run_queue_item(item, timeout=timeout)

        # Update queue file
        for q in queue:
            if q["id"] == item["id"]:
                q.update(result)
        save_queue(queue, queue_path)

        if result["status"] == "completed":
            summary["completed"] += 1
            print(f"    ✓ Completed", file=sys.stderr)
        else:
            summary["failed"] += 1
            print(f"    ✗ Failed: {result.get('error', 'unknown')}", file=sys.stderr)
            if halt_on_error:
                summary["status"] = "halted"
                summary["skipped"] = len(execution_order) - i - 1
                break

        summary["results"].append({
            "id": item["id"],
            "description": item["description"],
            "status": result["status"],
            "error": result.get("error"),
        })

    if "status" not in summary:
        summary["status"] = "completed"
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()

    # Save summary
    summary_path = Path(queue_path).parent / "queue-summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    return summary


def pause_queue(queue_path: str = DEFAULT_QUEUE_PATH) -> None:
    """Set pause flag on the queue."""
    queue = load_queue(queue_path)
    queue.append({"_paused": True, "paused_at": datetime.now(timezone.utc).isoformat()})
    save_queue(queue, queue_path)


def clear_queue(queue_path: str = DEFAULT_QUEUE_PATH) -> int:
    """Remove all queued items. Returns count of items cleared."""
    queue = load_queue(queue_path)
    cleared = sum(1 for q in queue if q.get("status") == "queued")
    queue = [q for q in queue if q.get("status") != "queued"]
    save_queue(queue, queue_path)
    return cleared


def format_queue_list(queue: list[dict]) -> str:
    """Format the queue as a readable table."""
    items = [q for q in queue if not q.get("_paused")]
    if not items:
        return "Queue is empty."

    lines = [
        "# Experiment Queue",
        "",
        "| ID | Priority | Status | Description | Depends On |",
        "|----|----------|--------|-------------|------------|",
    ]

    for item in items:
        dep = item.get("depends_on") or "—"
        lines.append(
            f"| {item['id']} | {item.get('priority', 'medium')} "
            f"| {item.get('status', 'queued')} | {item['description'][:50]} | {dep} |"
        )

    # Stats
    queued = sum(1 for q in items if q.get("status") == "queued")
    completed = sum(1 for q in items if q.get("status") == "completed")
    failed = sum(1 for q in items if q.get("status") == "failed")

    lines.extend([
        "",
        f"**Queued:** {queued} | **Completed:** {completed} | **Failed:** {failed}",
    ])

    return "\n".join(lines)


def format_batch_summary(summary: dict) -> str:
    """Format batch execution summary."""
    lines = [
        "# Queue Execution Summary",
        "",
        f"- **Status:** {summary.get('status', 'unknown')}",
        f"- **Total:** {summary.get('total', 0)}",
        f"- **Completed:** {summary.get('completed', 0)}",
        f"- **Failed:** {summary.get('failed', 0)}",
        f"- **Skipped:** {summary.get('skipped', 0)}",
    ]

    results = summary.get("results", [])
    if results:
        lines.extend(["", "## Results", ""])
        for r in results:
            status = "✓" if r["status"] == "completed" else "✗"
            error = f" ({r['error']})" if r.get("error") else ""
            lines.append(f"- {status} {r['id']}: {r['description']}{error}")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Experiment queue scheduler")
    parser.add_argument("action", choices=["add", "list", "run", "pause", "clear"],
                       help="Queue action")
    parser.add_argument("description", nargs="?", default=None,
                       help="Experiment description (for add)")
    parser.add_argument("--priority", default="medium",
                       choices=["critical", "high", "medium", "low"])
    parser.add_argument("--after", default=None, help="Queue ID dependency")
    parser.add_argument("--hypothesis", default=None, help="Hypothesis ID link")
    parser.add_argument("--queue", default=DEFAULT_QUEUE_PATH, help="Queue file path")
    parser.add_argument("--halt-on-error", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.action == "add":
        if not args.description:
            print("Usage: queue add 'description' [--priority high]", file=sys.stderr)
            sys.exit(1)
        item = add_to_queue(args.description, args.priority, args.after,
                           args.hypothesis, args.queue)
        if args.json:
            print(json.dumps(item, indent=2, default=str))
        else:
            print(f"Added {item['id']}: {item['description']} [{item['priority']}]")

    elif args.action == "list":
        queue = load_queue(args.queue)
        if args.json:
            print(json.dumps(queue, indent=2, default=str))
        else:
            print(format_queue_list(queue))

    elif args.action == "run":
        summary = run_queue(args.queue, args.halt_on_error, args.timeout)
        if args.json:
            print(json.dumps(summary, indent=2, default=str))
        else:
            print(format_batch_summary(summary))

    elif args.action == "pause":
        pause_queue(args.queue)
        print("Queue paused. Current experiment will finish, then stop.")

    elif args.action == "clear":
        n = clear_queue(args.queue)
        print(f"Cleared {n} queued experiments.")


if __name__ == "__main__":
    main()
