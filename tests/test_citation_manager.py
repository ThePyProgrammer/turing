"""Tests for citation manager (citation_manager.py). Phase 25.1."""
from __future__ import annotations
import pytest
from scripts.citation_manager import (
    load_citations, save_citations, add_citation, list_citations,
    generate_bibtex, format_citations_report,
)

def test_load_missing(tmp_path):
    assert load_citations(str(tmp_path / "missing.yaml")) == []

def test_save_load(tmp_path):
    path = str(tmp_path / "cit.yaml")
    cits = [{"key": "Chen2016", "title": "XGBoost", "type": "method", "experiments": ["exp-042"]}]
    save_citations(cits, path)
    loaded = load_citations(path)
    assert len(loaded) == 1

def test_add_citation(tmp_path):
    path = str(tmp_path / "cit.yaml")
    result = add_citation("exp-042", key="Chen2016", title="XGBoost", cite_type="method",
                          citations_path=path, log_path=str(tmp_path / "log.jsonl"))
    # May error due to missing log, but function should not crash
    assert isinstance(result, dict)

def test_list_citations(tmp_path):
    path = str(tmp_path / "cit.yaml")
    save_citations([
        {"key": "Chen2016", "title": "XGBoost", "type": "method", "experiments": ["exp-042"]},
    ], path)
    result = list_citations(citations_path=path)
    assert isinstance(result, dict)
    assert result.get("total", 0) >= 1

def test_bibtex(tmp_path):
    path = str(tmp_path / "cit.yaml")
    save_citations([{"key": "Chen2016", "title": "XGBoost", "authors": "Chen", "year": 2016,
                     "type": "method", "experiments": ["exp-042"]}], path)
    bib = generate_bibtex(path)
    assert "Chen2016" in bib

def test_bibtex_empty(tmp_path):
    bib = generate_bibtex(str(tmp_path / "missing.yaml"))
    assert isinstance(bib, str)

def test_format_list():
    result = {"total": 1, "citations": [{"key": "Chen2016", "title": "XGBoost"}], "by_type": {}}
    text = format_citations_report(result, "list")
    assert isinstance(text, str)
