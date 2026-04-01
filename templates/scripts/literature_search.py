#!/usr/bin/env python3
"""Literature integration for ML experiments.

Targeted literature search scoped to the current experiment's domain.
Three modes: free query, baseline SOTA comparison, related papers.

Uses Semantic Scholar API (free, no key required for basic search)
with fallback to local-only mode when offline.

Usage:
    python scripts/literature_search.py "gradient boosting tabular"     # Free query
    python scripts/literature_search.py --baseline                       # SOTA comparison
    python scripts/literature_search.py --related exp-042                # Related papers
    python scripts/literature_search.py --auto-queue "query"             # Queue hypotheses
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments


SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
DEFAULT_RESULT_COUNT = 5
REQUEST_TIMEOUT = 15


def search_semantic_scholar(
    query: str,
    limit: int = DEFAULT_RESULT_COUNT,
    fields: str = "title,authors,year,venue,abstract,citationCount,externalIds",
) -> list[dict]:
    """Search Semantic Scholar for papers matching a query.

    Returns list of paper dicts with title, authors, year, venue,
    abstract, citation_count, and URLs.
    """
    params = urllib.parse.urlencode({
        "query": query,
        "limit": limit,
        "fields": fields,
    })
    url = f"{SEMANTIC_SCHOLAR_API}/paper/search?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "turing-ml/2.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        return [{"error": f"Semantic Scholar API failed: {e}"}]

    papers = []
    for item in data.get("data", []):
        authors = [a.get("name", "?") for a in (item.get("authors") or [])]
        ext_ids = item.get("externalIds") or {}

        paper = {
            "title": item.get("title", "Untitled"),
            "authors": authors[:5],
            "year": item.get("year"),
            "venue": item.get("venue") or "N/A",
            "abstract": (item.get("abstract") or "")[:300],
            "citation_count": item.get("citationCount", 0),
            "paper_id": item.get("paperId"),
            "doi": ext_ids.get("DOI"),
            "arxiv_id": ext_ids.get("ArXiv"),
            "url": f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}" if item.get("paperId") else None,
        }
        papers.append(paper)

    return papers


def build_query_from_config(config: dict) -> str:
    """Build a search query from project config."""
    parts = []

    task_desc = config.get("task_description", "")
    if task_desc:
        parts.append(task_desc)

    model_type = config.get("model", {}).get("type", "")
    if model_type:
        parts.append(model_type)

    primary_metric = config.get("evaluation", {}).get("primary_metric", "")
    if primary_metric:
        parts.append(primary_metric)

    data_source = config.get("data", {}).get("source", "")
    if data_source and not data_source.startswith("{"):
        parts.append(data_source)

    return " ".join(parts) if parts else "machine learning"


def build_query_from_experiment(experiment: dict) -> str:
    """Build a search query from experiment metadata."""
    parts = []

    model_type = experiment.get("config", {}).get("model_type", "")
    if model_type:
        parts.append(model_type)

    description = experiment.get("description", "")
    if description:
        parts.append(description[:100])

    return " ".join(parts) if parts else "machine learning experiment"


def search_baseline(
    config: dict,
    experiments: list[dict],
    primary_metric: str,
    lower_is_better: bool,
) -> dict:
    """Search for SOTA baselines and compare against current best.

    Returns dict with SOTA results and gap analysis.
    """
    query = build_query_from_config(config)
    query += " state of the art benchmark"

    papers = search_semantic_scholar(query, limit=10)
    if papers and "error" in papers[0]:
        return {"error": papers[0]["error"], "query": query}

    # Find current best
    best = None
    best_val = float("inf") if lower_is_better else float("-inf")
    for exp in experiments:
        if exp.get("status") != "kept":
            continue
        val = exp.get("metrics", {}).get(primary_metric)
        if val is None:
            continue
        if (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val = val
            best = exp

    result = {
        "query": query,
        "papers": papers,
        "current_best": {
            "experiment_id": best.get("experiment_id") if best else None,
            "metric": primary_metric,
            "value": round(best_val, 4) if best else None,
        },
    }

    return result


def search_related(
    experiment: dict,
    limit: int = DEFAULT_RESULT_COUNT,
) -> dict:
    """Find papers related to a specific experiment."""
    query = build_query_from_experiment(experiment)
    papers = search_semantic_scholar(query, limit=limit)

    return {
        "experiment_id": experiment.get("experiment_id", "?"),
        "query": query,
        "papers": papers,
    }


def generate_literature_hypotheses(papers: list[dict]) -> list[dict]:
    """Generate hypotheses from literature findings.

    Extracts methodological suggestions from paper titles/abstracts.
    """
    hypotheses = []
    for i, paper in enumerate(papers):
        if "error" in paper:
            continue
        title = paper.get("title", "")
        if not title:
            continue

        hypotheses.append({
            "id": f"hyp-lit-{i+1:03d}",
            "description": f"Investigate approach from: {title}",
            "source": "literature",
            "status": "queued",
            "priority": "normal",
            "rationale": f"Paper: {title} ({paper.get('year', '?')}, {paper.get('citation_count', 0)} citations)",
            "paper_url": paper.get("url"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    return hypotheses[:5]


def save_literature_results(results: dict, output_dir: str = "experiments/literature") -> Path:
    """Save literature search results to markdown file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    filepath = out_path / f"query-{timestamp}.md"

    with open(filepath, "w") as f:
        f.write(format_literature_report(results))

    return filepath


def format_literature_report(results: dict) -> str:
    """Format literature search results as markdown."""
    if "error" in results:
        return f"ERROR: {results['error']}"

    mode = results.get("mode", "query")
    query = results.get("query", "")
    papers = results.get("papers", [])

    lines = [
        f"# Literature Search",
        "",
        f"*Query: {query}*",
        f"*Mode: {mode}*",
        "",
    ]

    if not papers:
        lines.append("No papers found.")
        return "\n".join(lines)

    if any("error" in p for p in papers):
        error_paper = next(p for p in papers if "error" in p)
        lines.append(f"**API Error:** {error_paper['error']}")
        lines.append("")
        lines.append("*Search may be offline. Try again later or use a manual search.*")
        return "\n".join(lines)

    # Papers table
    lines.extend([
        "## Results",
        "",
        "| # | Title | Year | Venue | Citations |",
        "|---|-------|------|-------|-----------|",
    ])

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        year = paper.get("year", "?")
        venue = paper.get("venue", "N/A")
        cites = paper.get("citation_count", 0)
        lines.append(f"| {i} | {title} | {year} | {venue} | {cites} |")

    # Paper details
    lines.extend(["", "## Details", ""])
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Untitled")
        authors = ", ".join(paper.get("authors", [])[:3])
        if len(paper.get("authors", [])) > 3:
            authors += " et al."
        abstract = paper.get("abstract", "")
        url = paper.get("url", "")

        lines.append(f"### {i}. {title}")
        lines.append("")
        lines.append(f"**Authors:** {authors}")
        lines.append(f"**Year:** {paper.get('year', '?')} | **Venue:** {paper.get('venue', 'N/A')} | **Citations:** {paper.get('citation_count', 0)}")
        if url:
            lines.append(f"**URL:** {url}")
        if abstract:
            lines.append(f"**Abstract:** {abstract}...")
        lines.append("")

    # Baseline comparison
    baseline = results.get("current_best")
    if baseline and baseline.get("value") is not None:
        lines.extend([
            "## Current Performance",
            "",
            f"- **Best experiment:** {baseline.get('experiment_id', '?')}",
            f"- **{baseline['metric']}:** {baseline['value']:.4f}",
            "",
            "*Compare against reported baselines in the papers above.*",
        ])

    return "\n".join(lines)


def queue_literature_hypotheses(hypotheses: list[dict], queue_path: str = "hypotheses.yaml") -> int:
    """Append literature hypotheses to the queue."""
    path = Path(queue_path)
    existing = []
    if path.exists() and path.stat().st_size > 0:
        with open(path) as f:
            data = yaml.safe_load(f)
            if isinstance(data, list):
                existing = data

    existing_ids = {h.get("id") for h in existing}
    new = [h for h in hypotheses if h["id"] not in existing_ids]

    if new:
        existing.extend(new)
        with open(path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

    return len(new)


def run_literature_search(
    query: str | None = None,
    baseline: bool = False,
    related_exp_id: str | None = None,
    auto_queue: bool = False,
    config_path: str = "config.yaml",
    log_path: str = "experiments/log.jsonl",
    limit: int = DEFAULT_RESULT_COUNT,
) -> dict:
    """Run a literature search in the appropriate mode.

    Args:
        query: Free-text search query.
        baseline: If True, search for SOTA baselines.
        related_exp_id: If set, find papers related to this experiment.
        auto_queue: Auto-queue hypotheses from findings.
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        limit: Maximum number of results.

    Returns:
        Literature search result dict.
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    primary_metric = eval_cfg.get("primary_metric", "accuracy")
    lower_is_better = eval_cfg.get("lower_is_better", False)
    experiments = load_experiments(log_path)

    if baseline:
        result = search_baseline(config, experiments, primary_metric, lower_is_better)
        result["mode"] = "baseline"
    elif related_exp_id:
        target = None
        for exp in experiments:
            if exp.get("experiment_id") == related_exp_id:
                target = exp
                break
        if not target:
            return {"error": f"Experiment {related_exp_id} not found", "mode": "related"}
        result = search_related(target, limit=limit)
        result["mode"] = "related"
    elif query:
        papers = search_semantic_scholar(query, limit=limit)
        result = {"query": query, "papers": papers, "mode": "query"}
    else:
        # Default: search based on config
        query = build_query_from_config(config)
        papers = search_semantic_scholar(query, limit=limit)
        result = {"query": query, "papers": papers, "mode": "query"}

    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Generate and optionally queue hypotheses
    papers = result.get("papers", [])
    if papers and not any("error" in p for p in papers):
        hypotheses = generate_literature_hypotheses(papers)
        result["hypotheses"] = hypotheses

        if auto_queue and hypotheses:
            n_added = queue_literature_hypotheses(hypotheses)
            result["hypotheses_queued"] = n_added
            print(f"Queued {n_added} hypotheses from literature", file=sys.stderr)

    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Literature search for ML experiments")
    parser.add_argument("query", nargs="?", default=None, help="Free-text search query")
    parser.add_argument("--baseline", action="store_true", help="Search for SOTA baselines")
    parser.add_argument("--related", default=None, metavar="EXP_ID", help="Find related papers for experiment")
    parser.add_argument("--auto-queue", action="store_true", help="Auto-queue hypotheses from findings")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl", help="Path to experiment log")
    parser.add_argument("--limit", type=int, default=DEFAULT_RESULT_COUNT, help="Max results")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    result = run_literature_search(
        query=args.query,
        baseline=args.baseline,
        related_exp_id=args.related,
        auto_queue=args.auto_queue,
        config_path=args.config,
        log_path=args.log,
        limit=args.limit,
    )

    if "error" not in result:
        filepath = save_literature_results(result)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_literature_report(result))


if __name__ == "__main__":
    main()
