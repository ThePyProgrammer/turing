#!/usr/bin/env python3
"""Task property classifier for the literature-grounded suggestion pipeline.

Classifies an ML task description against the task taxonomy
(config/task_taxonomy.yaml) using keyword matching. Outputs detected
categories and their search terms for use in literature queries.

No API keys or external services required — pure keyword heuristics.

Usage:
    python scripts/classify_task.py --config config.yaml
    python scripts/classify_task.py --description "time series classification with noisy sensor data"
    python scripts/classify_task.py --config config.yaml --format search-terms
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


def load_taxonomy(taxonomy_path: str = "config/task_taxonomy.yaml") -> dict:
    """Load the task taxonomy from YAML config."""
    path = Path(taxonomy_path)
    # Check multiple locations
    candidates = [
        path,
        Path(__file__).parent.parent.parent / "config" / "task_taxonomy.yaml",
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f)
            return data.get("categories", {})
    return {}


def extract_task_description(config_path: str) -> str:
    """Extract task description from config.yaml.

    Looks for task.description, then falls back to building a description
    from evaluation.primary_metric + model.type + data fields.
    """
    path = Path(config_path)
    if not path.exists():
        return ""
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    # Direct description field
    desc = config.get("task", {}).get("description", "")
    if desc:
        return desc

    # Build from available config
    parts = []
    model_type = config.get("model", {}).get("type", "")
    if model_type:
        parts.append(f"model: {model_type}")

    metric = config.get("evaluation", {}).get("primary_metric", "")
    if metric:
        parts.append(f"metric: {metric}")

    target = config.get("data", {}).get("target_column", "")
    if target:
        parts.append(f"target: {target}")

    source = config.get("data", {}).get("source", "")
    if source:
        parts.append(f"data: {Path(source).name}")

    return ", ".join(parts) if parts else ""


def classify_task(
    description: str,
    taxonomy: dict,
    threshold: float = 0.3,
) -> list[dict]:
    """Classify a task description against the taxonomy.

    Returns a list of matching categories sorted by match confidence,
    each containing: key, label, description, search_terms, score.

    The score is based on keyword overlap between the task description
    and the category's label, description, and search_terms.
    """
    if not description or not taxonomy:
        return []

    # Normalize task description
    desc_lower = description.lower()
    desc_words = set(re.findall(r"[a-z][a-z0-9]+", desc_lower))

    matches = []
    for key, cat in taxonomy.items():
        label = cat.get("label", "")
        cat_desc = cat.get("description", "")
        search_terms = cat.get("search_terms", [])

        # Build category word set
        cat_text = f"{label} {cat_desc} {' '.join(search_terms)} {key.replace('_', ' ')}"
        cat_words = set(re.findall(r"[a-z][a-z0-9]+", cat_text.lower()))

        # Score: weighted combination of different match types
        score = 0.0

        # 1. Direct key match (e.g., "tabular" in description)
        key_words = key.lower().replace("_", " ").split()
        for kw in key_words:
            if kw in desc_lower:
                score += 0.4

        # 2. Search term match (strongest signal)
        for term in search_terms:
            term_lower = term.lower()
            if term_lower in desc_lower:
                score += 0.5
            elif any(w in desc_words for w in term_lower.split() if len(w) > 2):
                score += 0.2

        # 3. Word overlap with category description
        if cat_words and desc_words:
            overlap = len(desc_words & cat_words)
            if overlap > 0:
                score += min(0.3, overlap * 0.08)

        if score >= threshold:
            matches.append({
                "key": key,
                "label": label,
                "description": cat_desc,
                "search_terms": search_terms,
                "score": round(score, 3),
            })

    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches


def generate_search_queries(matches: list[dict], task_description: str) -> list[str]:
    """Generate literature search queries from detected categories.

    Combines category search terms with key task-specific words to
    produce queries suitable for web search or arXiv lookup.
    """
    if not matches:
        return [f"machine learning {task_description}"]

    # Extract task-specific keywords (not stopwords)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "can", "shall",
                 "with", "for", "from", "into", "about", "and", "but", "or",
                 "not", "this", "that", "these", "those", "my", "your", "our",
                 "using", "use", "data", "model", "task", "based", "need"}
    task_words = [w for w in re.findall(r"[a-z][a-z0-9]+", task_description.lower())
                  if w not in stopwords and len(w) > 2]
    task_context = " ".join(task_words[:5])  # top 5 distinctive words

    queries = []
    for match in matches[:3]:  # top 3 categories
        # Primary query: search terms + task context
        for term in match["search_terms"][:2]:
            q = f"{term} {task_context} machine learning"
            queries.append(q.strip())

    # Add a general query from the task description
    if task_context:
        queries.append(f"best model architecture {task_context}")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:7]  # max 7 queries


def format_output(
    matches: list[dict],
    queries: list[str],
    fmt: str = "summary",
) -> str:
    """Format classification output."""
    if fmt == "search-terms":
        return "\n".join(queries)

    if fmt == "json":
        import json
        return json.dumps({"categories": matches, "queries": queries}, indent=2)

    # Default: summary
    lines = []
    if matches:
        lines.append(f"Detected {len(matches)} matching task properties:")
        lines.append("")
        for m in matches:
            lines.append(f"  [{m['score']:.1f}] {m['label']}: {m['description']}")
        lines.append("")
    else:
        lines.append("No strong category matches detected.")
        lines.append("")

    if queries:
        lines.append("Suggested search queries:")
        for i, q in enumerate(queries, 1):
            lines.append(f"  {i}. {q}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify ML task against taxonomy")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--description", default="", help="Task description (overrides config)")
    parser.add_argument("--taxonomy", default="config/task_taxonomy.yaml", help="Taxonomy path")
    parser.add_argument("--format", choices=["summary", "search-terms", "json"], default="summary")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum match score")
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    if not taxonomy:
        print("Error: Could not load task taxonomy", file=sys.stderr)
        sys.exit(1)

    description = args.description or extract_task_description(args.config)
    if not description:
        print("Error: No task description found. Provide --description or add task.description to config.yaml",
              file=sys.stderr)
        sys.exit(1)

    matches = classify_task(description, taxonomy, threshold=args.threshold)
    queries = generate_search_queries(matches, description)
    print(format_output(matches, queries, args.format))


if __name__ == "__main__":
    main()
