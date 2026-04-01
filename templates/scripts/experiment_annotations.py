#!/usr/bin/env python3
"""Add human notes and annotations to experiments.

Attach contextual observations, insights, and tagged notes to any
experiment. Annotations are the institutional memory that metrics
alone cannot capture — "this ran during a data migration" or
"suspiciously high accuracy, check for leakage".

Usage:
    python scripts/experiment_annotations.py add exp-042 "Suspiciously high accuracy"
    python scripts/experiment_annotations.py add exp-042 "Check for leakage" --tags leakage,investigate
    python scripts/experiment_annotations.py list
    python scripts/experiment_annotations.py list exp-042
    python scripts/experiment_annotations.py search "leakage"
    python scripts/experiment_annotations.py search --tag investigate
    python scripts/experiment_annotations.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_ANNOTATIONS_PATH = "experiments/annotations.yaml"


# --- Storage ---


def load_annotations(path: str = DEFAULT_ANNOTATIONS_PATH) -> list[dict]:
    """Load annotations from YAML file."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def save_annotations(annotations: list[dict], path: str = DEFAULT_ANNOTATIONS_PATH) -> Path:
    """Save annotations list to YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(annotations, f, default_flow_style=False, sort_keys=False)
    return p


def get_next_annotation_id(annotations: list[dict]) -> str:
    """Generate next sequential annotation ID."""
    max_id = 0
    for ann in annotations:
        aid = ann.get("id", "")
        if aid.startswith("ann-"):
            try:
                num = int(aid.split("-")[1])
                max_id = max(max_id, num)
            except (ValueError, IndexError):
                pass
    return f"ann-{max_id + 1:03d}"


# --- Operations ---


def add_annotation(
    experiment_id: str,
    text: str,
    author: str | None = None,
    tags: list[str] | None = None,
    annotations_path: str = DEFAULT_ANNOTATIONS_PATH,
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Add an annotation to an experiment.

    Args:
        experiment_id: Target experiment ID (e.g., "exp-042").
        text: Annotation text.
        author: Who wrote the annotation. Defaults to $USER.
        tags: Optional list of tags for categorization.
        annotations_path: Path to annotations YAML.
        log_path: Path to experiment log for validation.

    Returns:
        The created annotation dict.
    """
    experiments = load_experiments(log_path)
    known_ids = {e.get("experiment_id") for e in experiments}
    if experiment_id not in known_ids:
        return {"error": f"Experiment '{experiment_id}' not found in log"}

    annotations = load_annotations(annotations_path)
    aid = get_next_annotation_id(annotations)

    annotation = {
        "id": aid,
        "experiment_id": experiment_id,
        "text": text,
        "author": author or os.environ.get("USER", "unknown"),
        "date": datetime.now(timezone.utc).isoformat(),
        "tags": tags or [],
    }

    annotations.append(annotation)
    save_annotations(annotations, annotations_path)
    return annotation


def list_annotations(
    experiment_id: str | None = None,
    annotations_path: str = DEFAULT_ANNOTATIONS_PATH,
) -> list[dict]:
    """List annotations, optionally filtered by experiment.

    Args:
        experiment_id: If given, filter to this experiment only.
        annotations_path: Path to annotations YAML.

    Returns:
        Filtered list of annotations.
    """
    annotations = load_annotations(annotations_path)
    if experiment_id:
        return [a for a in annotations if a.get("experiment_id") == experiment_id]
    return annotations


def search_annotations(
    keyword: str | None = None,
    tag: str | None = None,
    annotations_path: str = DEFAULT_ANNOTATIONS_PATH,
) -> list[dict]:
    """Search annotations by keyword in text or by tag.

    Args:
        keyword: Search string to match against annotation text.
        tag: Tag to filter by.
        annotations_path: Path to annotations YAML.

    Returns:
        List of matching annotations.
    """
    annotations = load_annotations(annotations_path)
    results = []

    for ann in annotations:
        if keyword:
            text_lower = ann.get("text", "").lower()
            if keyword.lower() not in text_lower:
                continue
        if tag:
            ann_tags = [t.lower() for t in ann.get("tags", [])]
            if tag.lower() not in ann_tags:
                continue
        results.append(ann)

    return results


def delete_annotation(
    annotation_id: str,
    annotations_path: str = DEFAULT_ANNOTATIONS_PATH,
) -> dict:
    """Delete an annotation by ID.

    Args:
        annotation_id: Annotation ID to remove.
        annotations_path: Path to annotations YAML.

    Returns:
        Result dict with deleted annotation or error.
    """
    annotations = load_annotations(annotations_path)
    updated = [a for a in annotations if a.get("id") != annotation_id]

    if len(updated) == len(annotations):
        return {"error": f"Annotation '{annotation_id}' not found"}

    save_annotations(updated, annotations_path)
    return {"deleted": annotation_id, "remaining": len(updated)}


# --- Report ---


def format_annotations_report(annotations: list[dict], title: str = "Experiment Annotations") -> str:
    """Format annotations as a readable markdown report."""
    if not annotations:
        return "No annotations found."

    lines = [
        f"# {title}",
        "",
        f"*{len(annotations)} annotation(s)*",
        "",
        "| ID | Experiment | Date | Tags | Text |",
        "|----|-----------|------|------|------|",
    ]

    for ann in annotations:
        aid = ann.get("id", "?")
        eid = ann.get("experiment_id", "?")
        date = ann.get("date", "?")[:10]
        tags = ", ".join(ann.get("tags", [])) or "—"
        text = ann.get("text", "")
        # Truncate long text for table display
        display_text = text[:60] + "..." if len(text) > 60 else text
        lines.append(f"| {aid} | {eid} | {date} | {tags} | {display_text} |")

    # Group summary by experiment
    exp_counts: dict[str, int] = {}
    for ann in annotations:
        eid = ann.get("experiment_id", "?")
        exp_counts[eid] = exp_counts.get(eid, 0) + 1

    if len(exp_counts) > 1:
        lines.extend(["", "## By Experiment", ""])
        for eid, count in sorted(exp_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- **{eid}**: {count} annotation(s)")

    # Tag summary
    tag_counts: dict[str, int] = {}
    for ann in annotations:
        for tag in ann.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if tag_counts:
        lines.extend(["", "## By Tag", ""])
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- `{tag}`: {count}")

    return "\n".join(lines)


def save_annotations_report(report: dict, path: str = "experiments/annotations") -> Path:
    """Save annotations report to YAML."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"report-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.yaml"
    with open(out, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return out


# --- Orchestration ---


def run_annotations(
    action: str,
    experiment_id: str | None = None,
    text: str | None = None,
    author: str | None = None,
    tags: list[str] | None = None,
    keyword: str | None = None,
    tag: str | None = None,
    annotation_id: str | None = None,
    annotations_path: str = DEFAULT_ANNOTATIONS_PATH,
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Run annotation operation.

    Args:
        action: One of add, list, search, delete.
        experiment_id: Target experiment (for add/list).
        text: Annotation text (for add).
        author: Annotation author (for add).
        tags: Tags list (for add).
        keyword: Search keyword (for search).
        tag: Search tag (for search).
        annotation_id: Annotation ID (for delete).
        annotations_path: Path to annotations YAML.
        log_path: Path to experiment log.

    Returns:
        Result dict.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if action == "add":
        if not experiment_id or not text:
            return {"error": "Both experiment_id and text are required for add"}
        result = add_annotation(experiment_id, text, author, tags,
                                annotations_path, log_path)
        if "error" in result:
            return {"timestamp": timestamp, **result}
        return {"timestamp": timestamp, "action": "add", "annotation": result}

    elif action == "list":
        results = list_annotations(experiment_id, annotations_path)
        return {
            "timestamp": timestamp,
            "action": "list",
            "filter": {"experiment_id": experiment_id},
            "count": len(results),
            "annotations": results,
        }

    elif action == "search":
        if not keyword and not tag:
            return {"error": "Provide --search keyword or --tag for search"}
        results = search_annotations(keyword, tag, annotations_path)
        return {
            "timestamp": timestamp,
            "action": "search",
            "filter": {"keyword": keyword, "tag": tag},
            "count": len(results),
            "annotations": results,
        }

    elif action == "delete":
        if not annotation_id:
            return {"error": "Provide annotation ID for delete"}
        result = delete_annotation(annotation_id, annotations_path)
        return {"timestamp": timestamp, "action": "delete", **result}

    return {"error": f"Unknown action: {action}"}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Add human notes to experiments")
    parser.add_argument("action", choices=["add", "list", "search", "delete"],
                        help="Annotation action")
    parser.add_argument("experiment_id", nargs="?", default=None,
                        help="Experiment ID (for add/list)")
    parser.add_argument("text", nargs="?", default=None,
                        help="Annotation text (for add)")
    parser.add_argument("--author", default=None, help="Annotation author")
    parser.add_argument("--tags", default=None,
                        help="Comma-separated tags (e.g., leakage,investigate)")
    parser.add_argument("--search", dest="keyword", default=None,
                        help="Search keyword in annotation text")
    parser.add_argument("--tag", default=None,
                        help="Filter by tag")
    parser.add_argument("--annotation-id", default=None,
                        help="Annotation ID (for delete)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--annotations-path", default=DEFAULT_ANNOTATIONS_PATH,
                        help="Path to annotations YAML")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None

    report = run_annotations(
        action=args.action,
        experiment_id=args.experiment_id,
        text=args.text,
        author=args.author,
        tags=tags,
        keyword=args.keyword,
        tag=args.tag,
        annotation_id=args.annotation_id,
        annotations_path=args.annotations_path,
        log_path=args.log,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        if "error" in report:
            print(f"ERROR: {report['error']}", file=sys.stderr)
            sys.exit(1)
        annotations = report.get("annotations", [])
        if report.get("action") == "add":
            ann = report["annotation"]
            tags_str = f" [{', '.join(ann['tags'])}]" if ann.get("tags") else ""
            print(f"Added {ann['id']} to {ann['experiment_id']}: "
                  f"{ann['text']}{tags_str}")
        elif annotations or report.get("action") in ("list", "search"):
            title = "Experiment Annotations"
            if report.get("action") == "search":
                title = "Search Results"
            print(format_annotations_report(annotations, title))
        elif report.get("action") == "delete":
            print(f"Deleted annotation {report.get('deleted', '?')}")
        else:
            print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
