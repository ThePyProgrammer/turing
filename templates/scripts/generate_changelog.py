#!/usr/bin/env python3
"""Model changelog generator for the autoresearch pipeline.

Reads experiment history, identifies "keep" decisions that improved
the primary metric, groups them into versions by step-change
improvements, and formats a narrative changelog. Designed for both
technical and stakeholder audiences.

Usage:
    python scripts/generate_changelog.py
    python scripts/generate_changelog.py --audience stakeholder
    python scripts/generate_changelog.py --audience technical --since exp-030
    python scripts/generate_changelog.py --since 2026-03-15
    python scripts/generate_changelog.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_OUTPUT_PATH = "paper/CHANGELOG.md"

# Minimum relative improvement to trigger a new version boundary
VERSION_THRESHOLD = 0.02  # 2% relative improvement


# --- Version Detection ---


def compute_trajectory(
    experiments: list[dict],
    metric_name: str,
    lower_is_better: bool = False,
) -> list[dict]:
    """Build improvement trajectory from kept experiments."""
    trajectory = []
    best_val = None

    for exp in experiments:
        if exp.get("status") != "kept":
            continue

        val = exp.get("metrics", {}).get(metric_name)
        if val is None or not isinstance(val, (int, float)):
            continue

        prev_best = best_val
        is_improvement = False

        if best_val is None:
            best_val = val
            is_improvement = True
        elif lower_is_better and val < best_val:
            best_val = val
            is_improvement = True
        elif not lower_is_better and val > best_val:
            best_val = val
            is_improvement = True

        if is_improvement:
            delta = 0.0
            relative_delta = 0.0
            if prev_best is not None:
                delta = val - prev_best
                if prev_best != 0:
                    relative_delta = abs(delta) / abs(prev_best)

            trajectory.append({
                "experiment_id": exp.get("experiment_id", "?"),
                "description": exp.get("description", ""),
                "timestamp": exp.get("timestamp", ""),
                "value": val,
                "delta": delta,
                "relative_delta": relative_delta,
                "config": exp.get("config", {}),
                "family": exp.get("family") or exp.get("config", {}).get("model_type", ""),
            })

    return trajectory


def detect_version_boundaries(
    trajectory: list[dict],
    threshold: float = VERSION_THRESHOLD,
) -> list[int]:
    """Detect version boundaries based on significant metric jumps.

    A new version starts when the relative improvement exceeds the
    threshold, suggesting a qualitative step-change rather than
    incremental tuning.

    Returns indices into the trajectory where new versions begin.
    """
    boundaries = [0]  # First entry always starts v1

    for i, entry in enumerate(trajectory):
        if i == 0:
            continue
        if entry["relative_delta"] >= threshold:
            boundaries.append(i)

    return boundaries


def group_into_versions(
    trajectory: list[dict],
    boundaries: list[int],
) -> list[dict]:
    """Group trajectory entries into version blocks.

    Each version contains the improvements between consecutive
    boundaries, with a summary of the collective improvement.
    """
    versions = []

    for vi, start in enumerate(boundaries):
        end = boundaries[vi + 1] if vi + 1 < len(boundaries) else len(trajectory)
        entries = trajectory[start:end]
        if not entries:
            continue

        first_val = entries[0]["value"]
        last_val = entries[-1]["value"]
        version_delta = last_val - first_val

        # Timestamp range
        timestamps = [e["timestamp"] for e in entries if e.get("timestamp")]
        date_range = ""
        if timestamps:
            first_ts = min(timestamps)[:10]
            last_ts = max(timestamps)[:10]
            date_range = first_ts if first_ts == last_ts else f"{first_ts} to {last_ts}"

        versions.append({
            "version": f"v{vi + 1}.0",
            "improvements": entries,
            "start_value": first_val,
            "end_value": last_val,
            "version_delta": version_delta,
            "n_improvements": len(entries),
            "date_range": date_range,
        })

    return versions


# --- Formatting ---


def format_technical_changelog(
    versions: list[dict],
    metric_name: str,
    lower_is_better: bool,
    task_description: str,
) -> str:
    """Format changelog for technical audience with experiment IDs and deltas."""
    direction = "lower is better" if lower_is_better else "higher is better"
    lines = [
        f"# Model Changelog",
        f"",
        f"**Task:** {task_description}",
        f"**Primary metric:** {metric_name} ({direction})",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"",
        "---",
        "",
    ]

    for version in reversed(versions):
        ver = version["version"]
        end_val = version["end_value"]
        delta = version["version_delta"]
        delta_sign = "+" if delta > 0 else ""
        date_range = version.get("date_range", "")
        n = version["n_improvements"]

        lines.append(f"## {ver} ({date_range})")
        lines.append("")
        lines.append(f"**{metric_name}:** {end_val:.4f} ({delta_sign}{delta:.4f} from version start)")
        lines.append(f"**Improvements:** {n}")
        lines.append("")

        for entry in version["improvements"]:
            exp_id = entry["experiment_id"]
            desc = entry["description"] or "No description"
            val = entry["value"]
            entry_delta = entry["delta"]
            entry_sign = "+" if entry_delta > 0 else ""
            family = entry.get("family", "")
            family_tag = f" [{family}]" if family else ""

            lines.append(f"- `{exp_id}`{family_tag}: {desc}")
            lines.append(f"  {metric_name}: {val:.4f} ({entry_sign}{entry_delta:.4f})")

            # Show key config changes
            cfg = entry.get("config", {})
            if cfg:
                cfg_items = []
                for k, v in cfg.items():
                    if isinstance(v, (int, float, str, bool)):
                        cfg_items.append(f"{k}={v}")
                if cfg_items:
                    lines.append(f"  Config: {', '.join(cfg_items[:5])}")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def format_stakeholder_changelog(
    versions: list[dict],
    metric_name: str,
    lower_is_better: bool,
    task_description: str,
) -> str:
    """Format changelog for non-technical stakeholders.

    No experiment IDs, no configs. Plain English with clear
    performance narratives.
    """
    direction_word = "reduced" if lower_is_better else "improved"
    lines = [
        f"# Model Progress Report",
        f"",
        f"**Task:** {task_description}",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"",
    ]

    # Overall summary
    if versions:
        first_val = versions[0]["start_value"]
        last_val = versions[-1]["end_value"]
        total_delta = last_val - first_val
        total_pct = abs(total_delta / first_val * 100) if first_val != 0 else 0

        lines.append(f"## Overall Progress")
        lines.append("")
        lines.append(f"Performance {direction_word} from **{first_val:.4f}** to "
                      f"**{last_val:.4f}** ({total_pct:.1f}% change) across "
                      f"**{len(versions)} version(s)**.")
        lines.append("")
        lines.append("---")
        lines.append("")

    for version in reversed(versions):
        ver = version["version"]
        date_range = version.get("date_range", "")
        n = version["n_improvements"]

        lines.append(f"## {ver}" + (f" ({date_range})" if date_range else ""))
        lines.append("")

        # Narrative description
        improvements = version["improvements"]
        end_val = version["end_value"]

        # Summarize what changed in plain language
        families_seen = set()
        descriptions = []
        for entry in improvements:
            desc = entry.get("description", "")
            family = entry.get("family", "")
            if family:
                families_seen.add(family)
            if desc:
                # Clean up technical jargon for stakeholders
                clean = desc.replace("_", " ").strip()
                if clean and clean not in descriptions:
                    descriptions.append(clean)

        if families_seen:
            lines.append(f"Explored approaches: {', '.join(sorted(families_seen))}.")
            lines.append("")

        if descriptions:
            lines.append("Key changes:")
            lines.append("")
            for desc in descriptions[:5]:
                lines.append(f"- {desc}")
            lines.append("")

        # Performance summary in plain language
        delta = version["version_delta"]
        if abs(delta) > 0:
            pct = abs(delta / version["start_value"] * 100) if version["start_value"] != 0 else 0
            lines.append(f"Result: {metric_name.replace('_', ' ')} {direction_word} by "
                          f"**{pct:.1f}%** to **{end_val:.4f}** "
                          f"({n} improvement{'s' if n > 1 else ''}).")
        else:
            lines.append(f"Established baseline at **{end_val:.4f}**.")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# --- Report ---


def format_changelog_report(result: dict) -> str:
    """Format the changelog result as readable text."""
    if "changelog" in result:
        return result["changelog"]
    return json.dumps(result, indent=2, default=str)


def save_changelog_report(
    result: dict,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save the changelog to a markdown file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    changelog = result.get("changelog", "")
    with open(p, "w") as f:
        f.write(changelog)
    return p


# --- Orchestration ---


def filter_experiments_since(
    experiments: list[dict],
    since: str,
) -> list[dict]:
    """Filter experiments to those after a given experiment ID or date.

    If `since` looks like an experiment ID (starts with "exp-"), filter
    to experiments after that one. Otherwise treat as a date string.
    """
    if since.startswith("exp-"):
        found = False
        filtered = []
        for exp in experiments:
            if found:
                filtered.append(exp)
            if exp.get("experiment_id") == since:
                found = True
        return filtered
    else:
        return [e for e in experiments if e.get("timestamp", "") >= since]


def run_generate_changelog(
    audience: str = "technical",
    since: str | None = None,
    threshold: float = VERSION_THRESHOLD,
    log_path: str = DEFAULT_LOG_PATH,
    config_path: str = "config.yaml",
    output_path: str = DEFAULT_OUTPUT_PATH,
    save: bool = True,
) -> dict:
    """Generate model changelog from experiment history."""
    timestamp = datetime.now(timezone.utc).isoformat()
    config = load_config(config_path)
    experiments = load_experiments(log_path)

    if not experiments:
        return {"timestamp": timestamp, "error": "No experiments found in log"}

    if since:
        experiments = filter_experiments_since(experiments, since)
        if not experiments:
            return {"timestamp": timestamp, "error": f"No experiments found after '{since}'"}

    metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")
    lower_is_better = config.get("evaluation", {}).get("lower_is_better", False)
    task_desc = config.get("task", {}).get("description", "ML experiment campaign")

    trajectory = compute_trajectory(experiments, metric_name, lower_is_better)

    if not trajectory:
        return {"timestamp": timestamp, "error": "No kept experiments with metrics found"}

    boundaries = detect_version_boundaries(trajectory, threshold)
    versions = group_into_versions(trajectory, boundaries)

    if audience == "stakeholder":
        changelog = format_stakeholder_changelog(
            versions, metric_name, lower_is_better, task_desc,
        )
    else:
        changelog = format_technical_changelog(
            versions, metric_name, lower_is_better, task_desc,
        )

    result = {
        "timestamp": timestamp,
        "audience": audience,
        "metric": metric_name,
        "n_versions": len(versions),
        "n_improvements": len(trajectory),
        "versions": versions,
        "changelog": changelog,
    }

    if save:
        saved_path = save_changelog_report(result, output_path)
        result["saved_to"] = str(saved_path)

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate model changelog from experiment history",
    )
    parser.add_argument("--audience", choices=["technical", "stakeholder"],
                        default="technical",
                        help="Target audience (technical = full detail, stakeholder = plain English)")
    parser.add_argument("--since", default=None,
                        help="Start from experiment ID (exp-NNN) or date (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, default=VERSION_THRESHOLD,
                        help=f"Relative improvement threshold for version boundaries (default: {VERSION_THRESHOLD})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH,
                        help="Output file path")
    parser.add_argument("--no-save", action="store_true",
                        help="Print to stdout instead of saving file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    report = run_generate_changelog(
        audience=args.audience,
        since=args.since,
        threshold=args.threshold,
        log_path=args.log,
        config_path=args.config,
        output_path=args.output,
        save=not args.no_save,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        if "error" in report:
            print(f"ERROR: {report['error']}", file=sys.stderr)
            sys.exit(1)
        print(format_changelog_report(report))
        saved = report.get("saved_to")
        if saved:
            print(f"\nSaved to {saved}", file=sys.stderr)


if __name__ == "__main__":
    main()
