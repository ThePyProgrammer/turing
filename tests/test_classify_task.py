"""Tests for task property classifier (classify_task.py).

Phase 7.1: Verifies taxonomy loading, keyword-based classification,
search query generation, and config extraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.classify_task import (
    classify_task,
    extract_task_description,
    generate_search_queries,
    load_taxonomy,
)


# --- Fixtures ---

@pytest.fixture
def taxonomy() -> dict:
    """Minimal taxonomy for testing."""
    return {
        "tabular_data": {
            "label": "Tabular Data",
            "description": "Inputs are structured rows/columns",
            "search_terms": ["tabular", "structured data", "gradient boosting"],
        },
        "temporal_structure": {
            "label": "Temporal Structure",
            "description": "Data has inherent time dependencies or ordering",
            "search_terms": ["time series", "temporal", "forecasting"],
        },
        "classification_objective": {
            "label": "Classification",
            "description": "Task involves predicting discrete labels",
            "search_terms": ["classification", "categorical prediction"],
        },
        "imbalanced_classes": {
            "label": "Imbalanced Classes",
            "description": "Class distribution is heavily skewed",
            "search_terms": ["class imbalance", "oversampling", "SMOTE"],
        },
        "noise_robustness": {
            "label": "Noise Robustness",
            "description": "System must perform well under noisy inputs",
            "search_terms": ["noise robust", "data corruption", "missing data"],
        },
    }


@pytest.fixture
def taxonomy_file(tmp_path: Path, taxonomy: dict) -> Path:
    """Write taxonomy to a temp file."""
    path = tmp_path / "task_taxonomy.yaml"
    with open(path, "w") as f:
        yaml.dump({"categories": taxonomy}, f)
    return path


# --- load_taxonomy ---

def test_load_taxonomy_from_file(taxonomy_file):
    result = load_taxonomy(str(taxonomy_file))
    assert "tabular_data" in result
    assert "temporal_structure" in result


def test_load_taxonomy_missing_file(monkeypatch, tmp_path):
    # Override the fallback path so it doesn't find the real taxonomy
    monkeypatch.chdir(tmp_path)
    result = load_taxonomy("/nonexistent/path.yaml")
    # May find via fallback relative to script location, so just check it returns a dict
    assert isinstance(result, dict)


# --- extract_task_description ---

def test_extract_description_direct(tmp_path):
    config = {"task": {"description": "classify sensor data with noisy labels"}}
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    assert extract_task_description(str(path)) == "classify sensor data with noisy labels"


def test_extract_description_from_fields(tmp_path):
    config = {
        "model": {"type": "xgboost"},
        "evaluation": {"primary_metric": "f1_weighted"},
        "data": {"target_column": "status", "source": "/data/events.csv"},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    desc = extract_task_description(str(path))
    assert "xgboost" in desc
    assert "f1_weighted" in desc
    assert "status" in desc


def test_extract_description_missing_file():
    assert extract_task_description("/nonexistent/config.yaml") == ""


# --- classify_task ---

def test_classify_tabular(taxonomy):
    matches = classify_task("tabular classification with gradient boosting", taxonomy)
    keys = [m["key"] for m in matches]
    assert "tabular_data" in keys
    assert "classification_objective" in keys


def test_classify_time_series(taxonomy):
    matches = classify_task("time series forecasting of sensor readings", taxonomy)
    keys = [m["key"] for m in matches]
    assert "temporal_structure" in keys


def test_classify_imbalanced(taxonomy):
    matches = classify_task("classification on heavily imbalanced dataset with class imbalance", taxonomy)
    keys = [m["key"] for m in matches]
    assert "imbalanced_classes" in keys


def test_classify_empty_description(taxonomy):
    assert classify_task("", taxonomy) == []


def test_classify_empty_taxonomy():
    assert classify_task("some task", {}) == []


def test_classify_no_matches(taxonomy):
    matches = classify_task("quantum entanglement simulation", taxonomy, threshold=0.5)
    # May return empty or low-scoring matches
    for m in matches:
        assert m["score"] >= 0.5


def test_classify_scores_sorted(taxonomy):
    matches = classify_task("tabular classification with noisy time series data", taxonomy)
    scores = [m["score"] for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_classify_returns_search_terms(taxonomy):
    matches = classify_task("tabular data classification", taxonomy)
    for m in matches:
        assert "search_terms" in m
        assert isinstance(m["search_terms"], list)


# --- generate_search_queries ---

def test_generate_queries_basic(taxonomy):
    matches = classify_task("tabular classification", taxonomy)
    queries = generate_search_queries(matches, "tabular classification")
    assert len(queries) > 0
    assert len(queries) <= 7


def test_generate_queries_empty_matches():
    queries = generate_search_queries([], "some task")
    assert len(queries) == 1  # fallback query
    assert "machine learning" in queries[0]


def test_generate_queries_no_duplicates(taxonomy):
    matches = classify_task("tabular time series classification", taxonomy)
    queries = generate_search_queries(matches, "tabular time series classification")
    assert len(queries) == len(set(queries))


def test_generate_queries_include_task_context(taxonomy):
    matches = classify_task("tabular classification", taxonomy)
    queries = generate_search_queries(matches, "tabular classification of customer churn")
    # At least one query should contain task-specific words
    combined = " ".join(queries).lower()
    assert "customer" in combined or "churn" in combined or "tabular" in combined
