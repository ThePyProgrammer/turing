#!/usr/bin/env python3
"""Natural language experiment search with structured filters.

Parse a query like "accuracy > 0.85 random forest last week" into
semantic keywords + structured filters, then rank experiments by
relevance. Simple keyword matching — full semantic search requires
FAISS infrastructure from Phase 9.1.

Usage:
    python scripts/experiment_search.py "accuracy > 0.85"
    python scripts/experiment_search.py "random forest status:kept"
    python scripts/experiment_search.py "family:baseline date:2025-01"
    python scripts/experiment_search.py "best model last week" --top 5
    python scripts/experiment_search.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"


# --- Query Parsing ---


def parse_query(query: str) -> dict:
    """Parse a natural language query into structured filters + keywords.

    Extracts:
    - Metric comparisons: accuracy>0.85, loss<0.5, f1>=0.9
    - Status filters: status:kept, status:discarded
    - Family filters: family:baseline
    - Tag filters: tag:production
    - Date ranges: date:2025-01, date:last-week, date:last-month
    - Remaining text becomes keyword search terms.

    Returns:
        Dict with 'keywords', 'metric_filters', 'status', 'family',
        'tags', 'date_range'.
    """
    filters: dict = {
        "keywords": [],
        "metric_filters": [],
        "status": None,
        "family": None,
        "tags": [],
        "date_range": None,
    }

    tokens = query.split()
    remaining = []

    for token in tokens:
        # Metric comparisons: accuracy>0.85, loss<=0.5
        metric_match = re.match(r"^(\w+)(>=|<=|>|<|=)([0-9.]+)$", token)
        if metric_match:
            metric, op, val = metric_match.groups()
            try:
                filters["metric_filters"].append({
                    "metric": metric,
                    "operator": op,
                    "value": float(val),
                })
            except ValueError:
                remaining.append(token)
            continue

        # Key:value filters
        kv_match = re.match(r"^(\w+):(.+)$", token)
        if kv_match:
            key, value = kv_match.groups()
            key_lower = key.lower()
            if key_lower == "status":
                filters["status"] = value
            elif key_lower == "family":
                filters["family"] = value
            elif key_lower == "tag":
                filters["tags"].append(value)
            elif key_lower == "date":
                filters["date_range"] = _parse_date_filter(value)
            else:
                # Treat unknown key:value as metric filter with equality
                try:
                    filters["metric_filters"].append({
                        "metric": key,
                        "operator": "=",
                        "value": float(value),
                    })
                except ValueError:
                    remaining.append(token)
            continue

        remaining.append(token)

    # Parse natural language date references from remaining tokens
    remaining_text = " ".join(remaining).lower()
    if not filters["date_range"]:
        if "last week" in remaining_text:
            filters["date_range"] = _parse_date_filter("last-week")
            remaining_text = remaining_text.replace("last week", "").strip()
        elif "last month" in remaining_text:
            filters["date_range"] = _parse_date_filter("last-month")
            remaining_text = remaining_text.replace("last month", "").strip()
        elif "today" in remaining_text:
            filters["date_range"] = _parse_date_filter("today")
            remaining_text = remaining_text.replace("today", "").strip()

    # Clean up keywords — remove stopwords
    stopwords = {"the", "a", "an", "and", "or", "with", "from", "in", "on",
                 "for", "to", "of", "is", "was", "best", "model", "experiment"}
    filters["keywords"] = [
        w for w in remaining_text.split()
        if w and w not in stopwords
    ]

    return filters


def _parse_date_filter(value: str) -> dict | None:
    """Parse date filter value into a start/end range."""
    now = datetime.now(timezone.utc)

    if value == "last-week":
        start = now - timedelta(days=7)
        return {"start": start.isoformat(), "end": now.isoformat()}
    elif value == "last-month":
        start = now - timedelta(days=30)
        return {"start": start.isoformat(), "end": now.isoformat()}
    elif value == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return {"start": start.isoformat(), "end": now.isoformat()}

    # YYYY-MM format
    ym_match = re.match(r"^(\d{4})-(\d{2})$", value)
    if ym_match:
        year, month = int(ym_match.group(1)), int(ym_match.group(2))
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        return {"start": start.isoformat(), "end": end.isoformat()}

    # YYYY-MM-DD format
    ymd_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
    if ymd_match:
        year, month, day = int(ymd_match.group(1)), int(ymd_match.group(2)), int(ymd_match.group(3))
        start = datetime(year, month, day, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        return {"start": start.isoformat(), "end": end.isoformat()}

    return None


# --- Filtering ---


def apply_filters(experiments: list[dict], filters: dict) -> list[dict]:
    """Apply structured filters to experiment list.

    Args:
        experiments: Raw experiment list from log.
        filters: Parsed filter dict from parse_query.

    Returns:
        Filtered experiments.
    """
    results = list(experiments)

    # Status filter
    if filters.get("status"):
        results = [e for e in results
                   if e.get("status", "").lower() == filters["status"].lower()]

    # Family filter
    if filters.get("family"):
        family = filters["family"].lower()
        results = [e for e in results
                   if family in (e.get("family", "") or "").lower()]

    # Date range filter
    date_range = filters.get("date_range")
    if date_range:
        start = date_range.get("start", "")
        end = date_range.get("end", "")
        results = [e for e in results
                   if start <= e.get("timestamp", "") <= end]

    # Tag filter
    for tag in filters.get("tags", []):
        tag_lower = tag.lower()
        results = [e for e in results
                   if tag_lower in [t.lower() for t in e.get("tags", [])]]

    # Metric filters
    for mf in filters.get("metric_filters", []):
        metric = mf["metric"]
        op = mf["operator"]
        threshold = mf["value"]
        filtered = []
        for e in results:
            val = e.get("metrics", {}).get(metric)
            if val is None:
                continue
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue
            if _compare(val, op, threshold):
                filtered.append(e)
        results = filtered

    return results


def _compare(val: float, op: str, threshold: float) -> bool:
    """Apply comparison operator."""
    if op == ">":
        return val > threshold
    elif op == ">=":
        return val >= threshold
    elif op == "<":
        return val < threshold
    elif op == "<=":
        return val <= threshold
    elif op == "=":
        return abs(val - threshold) < 1e-6
    return False


# --- Ranking ---


def rank_by_keywords(experiments: list[dict], keywords: list[str]) -> list[tuple[dict, float]]:
    """Rank experiments by keyword relevance.

    Simple TF-based scoring: count keyword hits across experiment fields.
    Higher score = more relevant.

    Returns:
        List of (experiment, score) tuples sorted by descending score.
    """
    if not keywords:
        return [(e, 1.0) for e in experiments]

    scored = []
    for exp in experiments:
        searchable = _build_searchable_text(exp)
        score = 0.0
        for kw in keywords:
            kw_lower = kw.lower()
            count = searchable.count(kw_lower)
            if count > 0:
                score += 1.0 + 0.5 * (count - 1)
        scored.append((exp, score))

    # Sort by score descending, then by timestamp descending for ties
    scored.sort(key=lambda x: (x[1], x[0].get("timestamp", "")), reverse=True)
    return scored


def _build_searchable_text(exp: dict) -> str:
    """Build a single searchable string from experiment fields."""
    parts = [
        exp.get("experiment_id", ""),
        exp.get("description", ""),
        exp.get("family", ""),
        exp.get("status", ""),
        str(exp.get("config", {}).get("model_type", "")),
        str(exp.get("config", {}).get("experiment_type", "")),
        " ".join(exp.get("tags", [])),
    ]
    return " ".join(parts).lower()


# --- Report ---


def format_search_report(results: list[tuple[dict, float]], filters: dict, query: str) -> str:
    """Format search results as a ranked markdown table."""
    lines = [
        "# Experiment Search Results",
        "",
        f"**Query:** `{query}`",
    ]

    # Show parsed filters
    active_filters = []
    if filters.get("status"):
        active_filters.append(f"status={filters['status']}")
    if filters.get("family"):
        active_filters.append(f"family={filters['family']}")
    for mf in filters.get("metric_filters", []):
        active_filters.append(f"{mf['metric']}{mf['operator']}{mf['value']}")
    if filters.get("date_range"):
        active_filters.append(f"date range applied")
    if filters.get("keywords"):
        active_filters.append(f"keywords: {', '.join(filters['keywords'])}")

    if active_filters:
        lines.append(f"**Filters:** {' | '.join(active_filters)}")
    lines.extend(["", f"**{len(results)} result(s)**", ""])

    if not results:
        lines.append("No experiments matched the query.")
        return "\n".join(lines)

    lines.extend([
        "| Rank | Experiment | Status | Family | Score | Key Metrics |",
        "|------|-----------|--------|--------|-------|-------------|",
    ])

    for i, (exp, score) in enumerate(results, 1):
        eid = exp.get("experiment_id", "?")
        status = exp.get("status", "?")
        family = exp.get("family", "—") or "—"
        metrics = exp.get("metrics", {})
        # Show top 3 metrics
        metric_strs = []
        for k, v in list(metrics.items())[:3]:
            if isinstance(v, float):
                metric_strs.append(f"{k}={v:.4f}")
            else:
                metric_strs.append(f"{k}={v}")
        metrics_display = ", ".join(metric_strs) or "—"
        score_display = f"{score:.1f}" if score > 0 else "—"
        lines.append(f"| {i} | {eid} | {status} | {family} | {score_display} | {metrics_display} |")

    return "\n".join(lines)


def save_search_report(report: dict, path: str = "experiments/searches") -> Path:
    """Save search report to YAML."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"search-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.yaml"
    with open(out, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return out


# --- Orchestration ---


def run_search(
    query: str,
    top: int = 20,
    log_path: str = DEFAULT_LOG_PATH,
    config_path: str = "config.yaml",
) -> dict:
    """Run experiment search.

    Args:
        query: Natural language search query.
        top: Maximum results to return.
        log_path: Path to experiment log.
        config_path: Path to config.yaml.

    Returns:
        Search result dict.
    """
    config = load_config(config_path)
    experiments = load_experiments(log_path)

    if not experiments:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "error": "No experiments found",
        }

    filters = parse_query(query)
    filtered = apply_filters(experiments, filters)
    ranked = rank_by_keywords(filtered, filters.get("keywords", []))

    # Apply top-N limit
    ranked = ranked[:top]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "filters": filters,
        "total_experiments": len(experiments),
        "matched": len(ranked),
        "results": [
            {
                "experiment_id": exp.get("experiment_id"),
                "status": exp.get("status"),
                "family": exp.get("family"),
                "timestamp": exp.get("timestamp"),
                "metrics": exp.get("metrics", {}),
                "score": score,
            }
            for exp, score in ranked
        ],
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Natural language experiment search")
    parser.add_argument("query", nargs="?", default=None,
                        help="Search query (e.g., 'accuracy>0.85 random forest')")
    parser.add_argument("--top", type=int, default=20,
                        help="Maximum number of results (default: 20)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if not args.query:
        print("Usage: experiment_search.py 'query string'", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  experiment_search.py 'accuracy>0.85'", file=sys.stderr)
        print("  experiment_search.py 'status:kept family:baseline'", file=sys.stderr)
        print("  experiment_search.py 'random forest last week'", file=sys.stderr)
        sys.exit(1)

    report = run_search(
        query=args.query,
        top=args.top,
        log_path=args.log,
        config_path=args.config,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        if "error" in report:
            print(f"ERROR: {report['error']}", file=sys.stderr)
            sys.exit(1)
        filters = report.get("filters", {})
        results = [
            (r, r.get("score", 0))
            for r in report.get("results", [])
        ]
        print(format_search_report(results, filters, args.query))


if __name__ == "__main__":
    main()
