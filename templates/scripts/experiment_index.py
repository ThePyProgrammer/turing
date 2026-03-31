#!/usr/bin/env python3
"""Semantic experiment index for the autoresearch pipeline.

Indexes experiment descriptions and results using TF-IDF vectors,
enabling semantic search over experiment history. Complementary to
the novelty guard's keyword matching — this finds experiments by
meaning, not just shared tokens.

Uses sklearn's TfidfVectorizer (already a dependency) rather than
sentence-transformers + FAISS to avoid new heavyweight dependencies.

Usage:
    python scripts/experiment_index.py build --log experiments/log.jsonl
    python scripts/experiment_index.py query "experiments with high learning rate"
    python scripts/experiment_index.py query "overfitting with deep trees" --top 3
    python scripts/experiment_index.py similar exp-005 --top 5
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from scripts.turing_io import load_experiments


def experiment_to_text(exp: dict) -> str:
    """Convert an experiment to a searchable text representation.

    Combines description, config summary, metrics, and status into
    a single string that captures the experiment's essence.
    """
    parts = []

    desc = exp.get("description", "")
    if desc:
        parts.append(desc)

    # Config summary
    config = exp.get("config", {})
    model_type = config.get("model_type", "")
    if model_type:
        parts.append(f"model: {model_type}")

    exp_type = config.get("experiment_type", "")
    if exp_type:
        parts.append(f"type: {exp_type}")

    hyperparams = config.get("hyperparams", {})
    if hyperparams:
        hp_str = " ".join(f"{k}={v}" for k, v in hyperparams.items()
                         if isinstance(v, (int, float, str)))
        if hp_str:
            parts.append(f"hyperparams: {hp_str}")

    # Metrics
    metrics = exp.get("metrics", {})
    if metrics:
        metric_str = " ".join(f"{k}={v}" for k, v in metrics.items()
                             if isinstance(v, (int, float)))
        if metric_str:
            parts.append(f"metrics: {metric_str}")

    # Status
    status = exp.get("status", "")
    if status:
        parts.append(f"status: {status}")

    # Family and tags
    family = exp.get("family", "")
    if family:
        parts.append(f"family: {family}")

    tags = exp.get("tags", [])
    if tags:
        parts.append(f"tags: {' '.join(tags)}")

    return " | ".join(parts) if parts else ""


class ExperimentIndex:
    """TF-IDF based semantic index over experiment history."""

    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.experiment_ids: list[str] = []
        self.experiment_texts: list[str] = []
        self.experiment_data: list[dict] = []

    def build(self, experiments: list[dict]) -> int:
        """Build the index from a list of experiments.

        Returns the number of experiments indexed.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = []
        ids = []
        data = []

        for exp in experiments:
            text = experiment_to_text(exp)
            if text:
                texts.append(text)
                ids.append(exp.get("experiment_id", "?"))
                data.append(exp)

        if not texts:
            return 0

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),  # unigrams and bigrams
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.experiment_ids = ids
        self.experiment_texts = texts
        self.experiment_data = data

        return len(texts)

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Query the index with natural language.

        Returns top-K most similar experiments with scores.
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query_text])

        # Cosine similarity = dot product (TF-IDF vectors are already normalized)
        similarities = (self.tfidf_matrix @ query_vec.T).toarray().flatten()

        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score <= 0:
                continue
            exp = self.experiment_data[idx]
            results.append({
                "experiment_id": self.experiment_ids[idx],
                "similarity": round(score, 4),
                "description": exp.get("description", ""),
                "status": exp.get("status", "unknown"),
                "metrics": exp.get("metrics", {}),
                "config": exp.get("config", {}),
            })

        return results

    def find_similar(self, experiment_id: str, top_k: int = 5) -> list[dict]:
        """Find experiments similar to a given experiment ID."""
        try:
            idx = self.experiment_ids.index(experiment_id)
        except ValueError:
            return []

        return self.query(self.experiment_texts[idx], top_k + 1)

    def save(self, path: str) -> None:
        """Save the index to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "experiment_ids": self.experiment_ids,
                "experiment_texts": self.experiment_texts,
                "experiment_data": self.experiment_data,
            }, f)

    @classmethod
    def load(cls, path: str) -> ExperimentIndex | None:
        """Load an index from disk."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            idx = cls()
            idx.vectorizer = data["vectorizer"]
            idx.tfidf_matrix = data["tfidf_matrix"]
            idx.experiment_ids = data["experiment_ids"]
            idx.experiment_texts = data["experiment_texts"]
            idx.experiment_data = data["experiment_data"]
            return idx
        except Exception:
            return None


def format_results(results: list[dict], metric_name: str = "accuracy") -> str:
    """Format query results for display."""
    if not results:
        return "No matching experiments found."

    lines = [f"Top {len(results)} similar experiments:", ""]
    for i, r in enumerate(results, 1):
        metric_val = r["metrics"].get(metric_name, "?")
        lines.append(f"  {i}. {r['experiment_id']} (similarity: {r['similarity']:.3f})")
        lines.append(f"     {r['description']}")
        lines.append(f"     Status: {r['status']}, {metric_name}: {metric_val}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic experiment index")
    parser.add_argument("--log", default="experiments/log.jsonl")
    parser.add_argument("--index", default="experiments/index.pkl")

    subparsers = parser.add_subparsers(dest="command")

    # build
    subparsers.add_parser("build", help="Build/rebuild the index")

    # query
    query_parser = subparsers.add_parser("query", help="Search experiments by description")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--top", type=int, default=5)

    # similar
    sim_parser = subparsers.add_parser("similar", help="Find experiments similar to a given one")
    sim_parser.add_argument("experiment_id", help="Experiment ID")
    sim_parser.add_argument("--top", type=int, default=5)

    args = parser.parse_args()

    if args.command == "build":
        experiments = load_experiments(args.log)
        if not experiments:
            print("No experiments found.", file=sys.stderr)
            sys.exit(1)

        idx = ExperimentIndex()
        count = idx.build(experiments)
        idx.save(args.index)
        print(f"Indexed {count} experiments → {args.index}")

    elif args.command == "query":
        idx = ExperimentIndex.load(args.index)
        if idx is None:
            # Build on the fly
            experiments = load_experiments(args.log)
            if not experiments:
                print("No experiments to search.", file=sys.stderr)
                sys.exit(1)
            idx = ExperimentIndex()
            idx.build(experiments)

        results = idx.query(args.text, args.top)
        print(format_results(results))

    elif args.command == "similar":
        idx = ExperimentIndex.load(args.index)
        if idx is None:
            experiments = load_experiments(args.log)
            if not experiments:
                print("No experiments to search.", file=sys.stderr)
                sys.exit(1)
            idx = ExperimentIndex()
            idx.build(experiments)

        results = idx.find_similar(args.experiment_id, args.top)
        # Filter out the query experiment itself
        results = [r for r in results if r["experiment_id"] != args.experiment_id]
        print(format_results(results))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
