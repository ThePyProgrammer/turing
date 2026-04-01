"""Tests for experiment annotations (experiment_annotations.py). Phase 24.4."""
from __future__ import annotations
import pytest
from scripts.experiment_annotations import (
    load_annotations, save_annotations, get_next_annotation_id,
    search_annotations, format_annotations_report,
)

def test_load_missing(tmp_path):
    assert load_annotations(str(tmp_path / "missing.yaml")) == []

def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "ann.yaml")
    anns = [{"id": "ann-001", "experiment_id": "exp-042", "text": "Fragile", "tags": ["fragile"], "date": "2026-04-01"}]
    save_annotations(anns, path)
    loaded = load_annotations(path)
    assert len(loaded) == 1
    assert loaded[0]["experiment_id"] == "exp-042"

def test_next_id():
    anns = [{"id": "ann-001"}, {"id": "ann-002"}]
    next_id = get_next_annotation_id(anns)
    assert "003" in next_id

def test_next_id_empty():
    next_id = get_next_annotation_id([])
    assert "001" in next_id

def test_search(tmp_path):
    path = str(tmp_path / "ann.yaml")
    anns = [
        {"id": "ann-001", "experiment_id": "exp-042", "text": "Fragile preprocessing", "tags": ["fragile"]},
        {"id": "ann-002", "experiment_id": "exp-053", "text": "Good result", "tags": ["solid"]},
    ]
    save_annotations(anns, path)
    results = search_annotations("fragile", annotations_path=path)
    assert len(results) >= 1

def test_search_empty(tmp_path):
    results = search_annotations("anything", annotations_path=str(tmp_path / "missing.yaml"))
    assert results == []

def test_format():
    anns = [{"id": "ann-001", "experiment_id": "exp-042", "text": "Fragile", "tags": ["fragile"], "date": "2026-04-01"}]
    text = format_annotations_report(anns)
    assert "exp-042" in text

def test_format_empty():
    text = format_annotations_report([])
    assert isinstance(text, str)
