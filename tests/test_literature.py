"""Tests for literature_search.py.

Phase 14.1: Verifies query building, result parsing, baseline
comparison, hypothesis generation, and report formatting.
"""

from __future__ import annotations

import yaml
import pytest

from scripts.literature_search import (
    build_query_from_config,
    build_query_from_experiment,
    format_literature_report,
    generate_literature_hypotheses,
    queue_literature_hypotheses,
    save_literature_results,
)


# --- build_query_from_config ---


def test_query_from_config_full():
    """Should build query from all config fields."""
    config = {
        "task_description": "sentiment classification",
        "model": {"type": "xgboost"},
        "evaluation": {"primary_metric": "accuracy"},
        "data": {"source": "imdb_reviews.csv"},
    }
    query = build_query_from_config(config)
    assert "sentiment" in query
    assert "xgboost" in query


def test_query_from_config_minimal():
    """Should fall back to default for empty config."""
    query = build_query_from_config({})
    assert "machine learning" in query


def test_query_from_config_skips_placeholders():
    """Should skip template placeholders."""
    config = {"data": {"source": "{{DATA_SOURCE}}"}}
    query = build_query_from_config(config)
    assert "{{" not in query


# --- build_query_from_experiment ---


def test_query_from_experiment():
    """Should build query from experiment metadata."""
    exp = {
        "config": {"model_type": "lightgbm"},
        "description": "Tried gradient boosting with feature engineering",
    }
    query = build_query_from_experiment(exp)
    assert "lightgbm" in query
    assert "gradient" in query


def test_query_from_experiment_minimal():
    """Should fall back for empty experiment."""
    query = build_query_from_experiment({})
    assert "machine learning" in query


# --- generate_literature_hypotheses ---


def test_generate_hypotheses():
    """Should generate hypotheses from papers."""
    papers = [
        {"title": "XGBoost: A Scalable Tree Boosting System", "year": 2016, "citation_count": 15000, "url": "http://example.com"},
        {"title": "LightGBM: A Highly Efficient Gradient Boosting", "year": 2017, "citation_count": 8000, "url": None},
    ]
    hyps = generate_literature_hypotheses(papers)
    assert len(hyps) == 2
    assert hyps[0]["source"] == "literature"
    assert hyps[0]["status"] == "queued"
    assert "XGBoost" in hyps[0]["description"]


def test_generate_hypotheses_skips_errors():
    """Should skip papers with errors."""
    papers = [{"error": "API failed"}, {"title": "Good Paper", "year": 2020, "citation_count": 100}]
    hyps = generate_literature_hypotheses(papers)
    assert len(hyps) == 1


def test_generate_hypotheses_empty():
    """Should return empty for no papers."""
    assert generate_literature_hypotheses([]) == []


def test_generate_hypotheses_max_5():
    """Should cap at 5 hypotheses."""
    papers = [
        {"title": f"Paper {i}", "year": 2020, "citation_count": i * 100}
        for i in range(10)
    ]
    hyps = generate_literature_hypotheses(papers)
    assert len(hyps) <= 5


# --- queue_literature_hypotheses ---


def test_queue_hypotheses(tmp_path):
    """Should write hypotheses to queue file."""
    queue_path = str(tmp_path / "hypotheses.yaml")
    hyps = [{"id": "hyp-lit-001", "description": "test", "source": "literature", "status": "queued"}]
    n = queue_literature_hypotheses(hyps, queue_path)
    assert n == 1


def test_queue_hypotheses_dedup(tmp_path):
    """Should not add duplicate IDs."""
    queue_path = str(tmp_path / "hypotheses.yaml")
    hyps = [{"id": "hyp-lit-001", "description": "test", "source": "literature", "status": "queued"}]
    queue_literature_hypotheses(hyps, queue_path)
    n = queue_literature_hypotheses(hyps, queue_path)
    assert n == 0


# --- save_literature_results ---


def test_save_results(tmp_path):
    """Should save markdown file."""
    results = {"mode": "query", "query": "test", "papers": []}
    filepath = save_literature_results(results, str(tmp_path))
    assert filepath.exists()
    assert filepath.suffix == ".md"
    assert "query-" in filepath.name


# --- format_literature_report ---


def test_format_report_with_papers():
    """Should format papers as markdown table."""
    results = {
        "mode": "query",
        "query": "gradient boosting",
        "papers": [
            {
                "title": "XGBoost Paper",
                "authors": ["Chen", "Guestrin"],
                "year": 2016,
                "venue": "KDD",
                "citation_count": 15000,
                "abstract": "We describe XGBoost...",
                "url": "http://example.com",
            },
        ],
    }
    report = format_literature_report(results)
    assert "XGBoost Paper" in report
    assert "2016" in report
    assert "KDD" in report
    assert "15000" in report


def test_format_report_no_papers():
    """Should handle no results."""
    results = {"mode": "query", "query": "nonexistent topic", "papers": []}
    report = format_literature_report(results)
    assert "No papers found" in report


def test_format_report_api_error():
    """Should show error for API failures."""
    results = {
        "mode": "query",
        "query": "test",
        "papers": [{"error": "API timeout"}],
    }
    report = format_literature_report(results)
    assert "API" in report or "Error" in report


def test_format_report_with_baseline():
    """Should include baseline comparison."""
    results = {
        "mode": "baseline",
        "query": "test baseline",
        "papers": [{"title": "Paper", "authors": ["A"], "year": 2024, "venue": "NeurIPS",
                    "citation_count": 50, "abstract": "test", "url": None}],
        "current_best": {"experiment_id": "exp-042", "metric": "accuracy", "value": 0.872},
    }
    report = format_literature_report(results)
    assert "0.8720" in report
    assert "exp-042" in report


def test_format_report_error():
    """Should handle top-level error."""
    results = {"error": "Config not found"}
    report = format_literature_report(results)
    assert "ERROR" in report


def test_format_report_many_authors():
    """Should truncate long author lists."""
    results = {
        "mode": "query",
        "query": "test",
        "papers": [{
            "title": "Paper",
            "authors": ["A", "B", "C", "D", "E"],
            "year": 2024,
            "venue": "Conf",
            "citation_count": 10,
            "abstract": "test abstract",
            "url": None,
        }],
    }
    report = format_literature_report(results)
    assert "et al" in report
