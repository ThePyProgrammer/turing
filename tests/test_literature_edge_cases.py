"""Edge case tests for literature_search.py."""

from __future__ import annotations

import pytest

from scripts.literature_search import (
    build_query_from_config,
    build_query_from_experiment,
    format_literature_report,
    generate_literature_hypotheses,
    save_literature_results,
)


def test_query_long_description():
    """Should truncate very long experiment descriptions."""
    exp = {"config": {}, "description": "x" * 500}
    query = build_query_from_experiment(exp)
    assert len(query) <= 200


def test_generate_hypotheses_no_url():
    """Should handle papers without URLs."""
    papers = [{"title": "No URL Paper", "year": 2024, "citation_count": 5}]
    hyps = generate_literature_hypotheses(papers)
    assert len(hyps) == 1
    assert hyps[0]["paper_url"] is None


def test_generate_hypotheses_empty_title():
    """Should skip papers with empty titles."""
    papers = [{"title": "", "year": 2024, "citation_count": 0}]
    hyps = generate_literature_hypotheses(papers)
    assert len(hyps) == 0


def test_format_report_multiple_papers():
    """Should format multiple papers correctly."""
    results = {
        "mode": "query",
        "query": "test",
        "papers": [
            {"title": f"Paper {i}", "authors": ["Author"], "year": 2020 + i,
             "venue": "Conf", "citation_count": i * 100, "abstract": "test", "url": None}
            for i in range(5)
        ],
    }
    report = format_literature_report(results)
    for i in range(5):
        assert f"Paper {i}" in report


def test_save_creates_nested_dir(tmp_path):
    """Should create nested output directory."""
    results = {"mode": "query", "query": "test", "papers": []}
    filepath = save_literature_results(results, str(tmp_path / "deep" / "nested"))
    assert filepath.exists()


def test_format_report_no_abstract():
    """Should handle papers without abstracts."""
    results = {
        "mode": "query",
        "query": "test",
        "papers": [{"title": "Paper", "authors": [], "year": 2024,
                    "venue": "Conf", "citation_count": 0, "abstract": "", "url": None}],
    }
    report = format_literature_report(results)
    assert "Paper" in report
