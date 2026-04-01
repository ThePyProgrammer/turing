#!/usr/bin/env python3
"""Citation and attribution manager for the autoresearch pipeline.

Tracks academic citations associated with experiments. Every method,
dataset, technique, and codebase used in the research campaign should
have a citation. This script manages the citation store, audits for
missing attributions, and generates BibTeX output.

Usage:
    python scripts/citation_manager.py add exp-042 --key Chen2016 --title "XGBoost" --url "..."
    python scripts/citation_manager.py list
    python scripts/citation_manager.py check
    python scripts/citation_manager.py bib
    python scripts/citation_manager.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_CITATIONS_PATH = "experiments/citations.yaml"
VALID_TYPES = ["method", "dataset", "technique", "codebase"]

# Keywords that suggest a method/technique needing citation
METHOD_KEYWORDS = {
    "xgboost": "XGBoost (Chen & Guestrin, 2016)",
    "lightgbm": "LightGBM (Ke et al., 2017)",
    "catboost": "CatBoost (Prokhorenkova et al., 2018)",
    "random_forest": "Random Forest (Breiman, 2001)",
    "gradient_boosting": "Gradient Boosting (Friedman, 2001)",
    "adam": "Adam optimizer (Kingma & Ba, 2015)",
    "sgd": "SGD with momentum (Sutskever et al., 2013)",
    "dropout": "Dropout (Srivastava et al., 2014)",
    "batch_norm": "Batch Normalization (Ioffe & Szegedy, 2015)",
    "resnet": "ResNet (He et al., 2016)",
    "transformer": "Transformer (Vaswani et al., 2017)",
    "bert": "BERT (Devlin et al., 2019)",
    "lstm": "LSTM (Hochreiter & Schmidhuber, 1997)",
    "svm": "SVM (Cortes & Vapnik, 1995)",
    "lasso": "Lasso (Tibshirani, 1996)",
    "ridge": "Ridge Regression (Hoerl & Kennard, 1970)",
    "elastic_net": "Elastic Net (Zou & Hastie, 2005)",
    "pca": "PCA (Pearson, 1901)",
    "tsne": "t-SNE (van der Maaten & Hinton, 2008)",
    "umap": "UMAP (McInnes et al., 2018)",
    "cross_validation": "Cross-validation (Stone, 1974)",
    "smote": "SMOTE (Chawla et al., 2002)",
}


# --- Storage ---


def load_citations(path: str = DEFAULT_CITATIONS_PATH) -> list[dict]:
    """Load citations from YAML file."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def save_citations(citations: list[dict], path: str = DEFAULT_CITATIONS_PATH) -> Path:
    """Save citations list to YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(citations, f, default_flow_style=False, sort_keys=False)
    return p


# --- Operations ---


def add_citation(
    experiment_id: str,
    key: str,
    title: str,
    authors: str | None = None,
    year: int | None = None,
    url: str | None = None,
    doi: str | None = None,
    cite_type: str = "method",
    citations_path: str = DEFAULT_CITATIONS_PATH,
    log_path: str = DEFAULT_LOG_PATH,
) -> dict:
    """Add or update a citation, associating it with an experiment.

    If the citation key already exists, the experiment is appended to
    its experiment list. Otherwise a new citation entry is created.
    """
    experiments = load_experiments(log_path)
    known_ids = {e.get("experiment_id") for e in experiments}
    if experiment_id not in known_ids:
        return {"error": f"Experiment '{experiment_id}' not found in log"}

    if cite_type not in VALID_TYPES:
        return {"error": f"Invalid type '{cite_type}'. Valid: {VALID_TYPES}"}

    citations = load_citations(citations_path)

    # Check if key already exists
    existing = None
    for c in citations:
        if c.get("key") == key:
            existing = c
            break

    if existing:
        if experiment_id not in existing.get("experiments", []):
            existing.setdefault("experiments", []).append(experiment_id)
        # Update fields if provided
        if authors:
            existing["authors"] = authors
        if year:
            existing["year"] = year
        if url:
            existing["url"] = url
        if doi:
            existing["doi"] = doi
        save_citations(citations, citations_path)
        return {"action": "updated", "citation": existing}

    citation = {
        "key": key,
        "title": title,
        "authors": authors or "",
        "year": year or 0,
        "url": url or "",
        "doi": doi or "",
        "type": cite_type,
        "experiments": [experiment_id],
    }
    citations.append(citation)
    save_citations(citations, citations_path)
    return {"action": "added", "citation": citation}


def list_citations(
    citations_path: str = DEFAULT_CITATIONS_PATH,
) -> dict:
    """List all citations grouped by type with experiment associations."""
    citations = load_citations(citations_path)
    grouped: dict[str, list[dict]] = {}
    for c in citations:
        ctype = c.get("type", "unknown")
        grouped.setdefault(ctype, []).append(c)
    return {
        "total": len(citations),
        "by_type": grouped,
        "citations": citations,
    }


def check_citations(
    citations_path: str = DEFAULT_CITATIONS_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    config_path: str = "config.yaml",
) -> dict:
    """Audit for missing citations — methods used without attribution.

    Scans experiment configs and descriptions for known method keywords
    that lack a corresponding citation entry.
    """
    citations = load_citations(citations_path)
    experiments = load_experiments(log_path)
    config = load_config(config_path)

    cited_keys = {c.get("key", "").lower() for c in citations}
    cited_titles = {c.get("title", "").lower() for c in citations}

    missing: list[dict] = []
    covered: list[str] = []

    for keyword, suggestion in METHOD_KEYWORDS.items():
        # Check if this keyword appears in any experiment
        found_in: list[str] = []
        for exp in experiments:
            searchable = json.dumps(exp, default=str).lower()
            if keyword.lower() in searchable:
                found_in.append(exp.get("experiment_id", "?"))

        # Also check config
        config_str = json.dumps(config, default=str).lower()
        if keyword.lower() in config_str:
            found_in.append("config.yaml")

        if not found_in:
            continue

        # Check if cited
        is_cited = (
            keyword.lower() in cited_keys
            or keyword.lower() in cited_titles
            or any(keyword.lower() in c.get("title", "").lower() for c in citations)
        )

        if is_cited:
            covered.append(keyword)
        else:
            missing.append({
                "keyword": keyword,
                "suggestion": suggestion,
                "found_in": found_in,
            })

    return {
        "missing": missing,
        "covered": covered,
        "total_checked": len(METHOD_KEYWORDS),
        "coverage": f"{len(covered)}/{len(covered) + len(missing)}" if (covered or missing) else "N/A",
    }


def generate_bibtex(citations_path: str = DEFAULT_CITATIONS_PATH) -> str:
    """Generate BibTeX output from all citations."""
    citations = load_citations(citations_path)
    if not citations:
        return "% No citations found.\n"

    entries = []
    for c in citations:
        key = c.get("key", "unknown")
        title = c.get("title", "")
        authors = c.get("authors", "")
        year = c.get("year", 0)
        url = c.get("url", "")
        doi = c.get("doi", "")

        # Determine entry type
        entry_type = "misc"
        if doi:
            entry_type = "article"

        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title = {{{title}}},")
        if authors:
            lines.append(f"  author = {{{authors}}},")
        if year:
            lines.append(f"  year = {{{year}}},")
        if url:
            lines.append(f"  url = {{{url}}},")
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        note = f"Type: {c.get('type', 'unknown')}. Used in: {', '.join(c.get('experiments', []))}"
        lines.append(f"  note = {{{note}}},")
        lines.append("}")
        entries.append("\n".join(lines))

    header = f"% Auto-generated by Turing citation manager\n% {len(entries)} citation(s)\n"
    return header + "\n\n".join(entries) + "\n"


# --- Report ---


def format_citations_report(result: dict, action: str) -> str:
    """Format citation results as readable text."""
    lines: list[str] = []

    if action == "list":
        total = result.get("total", 0)
        lines.append(f"# Citations ({total} total)")
        lines.append("")
        by_type = result.get("by_type", {})
        for ctype in VALID_TYPES:
            cites = by_type.get(ctype, [])
            if not cites:
                continue
            lines.append(f"## {ctype.title()} ({len(cites)})")
            lines.append("")
            for c in cites:
                key = c.get("key", "?")
                title = c.get("title", "?")
                authors = c.get("authors", "")
                year = c.get("year", "")
                exps = ", ".join(c.get("experiments", []))
                author_year = f" ({authors}, {year})" if authors and year else ""
                lines.append(f"- **[{key}]** {title}{author_year}")
                if exps:
                    lines.append(f"  Experiments: {exps}")
            lines.append("")

    elif action == "check":
        missing = result.get("missing", [])
        covered = result.get("covered", [])
        coverage = result.get("coverage", "N/A")
        lines.append(f"# Citation Audit (coverage: {coverage})")
        lines.append("")
        if missing:
            lines.append(f"## Missing Citations ({len(missing)})")
            lines.append("")
            for m in missing:
                lines.append(f"- **{m['keyword']}**: {m['suggestion']}")
                lines.append(f"  Found in: {', '.join(m['found_in'])}")
            lines.append("")
        if covered:
            lines.append(f"## Covered ({len(covered)})")
            lines.append("")
            for kw in covered:
                lines.append(f"- {kw}")
            lines.append("")
        if not missing:
            lines.append("All detected methods have citations.")

    elif action == "add":
        cite = result.get("citation", {})
        act = result.get("action", "added")
        lines.append(f"Citation {act}: [{cite.get('key')}] {cite.get('title')}")
        lines.append(f"  Type: {cite.get('type')} | Experiments: {', '.join(cite.get('experiments', []))}")

    return "\n".join(lines)


def save_citations_report(report: dict, path: str = "experiments/citations") -> Path:
    """Save citation report to YAML."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"report-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.yaml"
    with open(out, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return out


# --- Orchestration ---


def run_citation_manager(
    action: str,
    experiment_id: str | None = None,
    key: str | None = None,
    title: str | None = None,
    authors: str | None = None,
    year: int | None = None,
    url: str | None = None,
    doi: str | None = None,
    cite_type: str = "method",
    citations_path: str = DEFAULT_CITATIONS_PATH,
    log_path: str = DEFAULT_LOG_PATH,
    config_path: str = "config.yaml",
) -> dict:
    """Run citation manager operation."""
    timestamp = datetime.now(timezone.utc).isoformat()

    if action == "add":
        if not experiment_id or not key or not title:
            return {"error": "add requires experiment_id, --key, and --title"}
        result = add_citation(
            experiment_id, key, title, authors, year, url, doi,
            cite_type, citations_path, log_path,
        )
        if "error" in result:
            return {"timestamp": timestamp, **result}
        return {"timestamp": timestamp, "action": "add", **result}

    elif action == "list":
        result = list_citations(citations_path)
        return {"timestamp": timestamp, "action": "list", **result}

    elif action == "check":
        result = check_citations(citations_path, log_path, config_path)
        return {"timestamp": timestamp, "action": "check", **result}

    elif action == "bib":
        bibtex = generate_bibtex(citations_path)
        return {"timestamp": timestamp, "action": "bib", "bibtex": bibtex,
                "count": len(load_citations(citations_path))}

    return {"error": f"Unknown action: {action}"}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Citation and attribution manager for ML experiments",
    )
    parser.add_argument("action", choices=["add", "list", "check", "bib"],
                        help="Citation action")
    parser.add_argument("experiment_id", nargs="?", default=None,
                        help="Experiment ID (for add)")
    parser.add_argument("--key", default=None, help="Citation key (e.g., Chen2016)")
    parser.add_argument("--title", default=None, help="Paper/resource title")
    parser.add_argument("--authors", default=None, help="Author list")
    parser.add_argument("--year", type=int, default=None, help="Publication year")
    parser.add_argument("--url", default=None, help="URL to paper/resource")
    parser.add_argument("--doi", default=None, help="DOI identifier")
    parser.add_argument("--type", dest="cite_type", default="method",
                        choices=VALID_TYPES, help="Citation type")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--citations-path", default=DEFAULT_CITATIONS_PATH,
                        help="Path to citations YAML")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    report = run_citation_manager(
        action=args.action,
        experiment_id=args.experiment_id,
        key=args.key,
        title=args.title,
        authors=args.authors,
        year=args.year,
        url=args.url,
        doi=args.doi,
        cite_type=args.cite_type,
        citations_path=args.citations_path,
        log_path=args.log,
        config_path=args.config,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        if "error" in report:
            print(f"ERROR: {report['error']}", file=sys.stderr)
            sys.exit(1)
        action = report.get("action", "")
        if action == "bib":
            print(report["bibtex"])
        else:
            print(format_citations_report(report, action))


if __name__ == "__main__":
    main()
