"""Tests for automated feature selection (feature_intelligence.py).

Phase 22.1: Verifies importance methods, consensus, redundancy, interactions, reporting.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.feature_intelligence import (
    compute_consensus,
    detect_redundancy,
    feature_analysis,
    format_feature_report,
    generate_interaction_features,
    mutual_information_ranking,
    l1_ranking,
    tree_importance_ranking,
)


# --- mutual_information_ranking ---

def test_mi_ranking():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)  # Feature 0 is most informative
    ranked = mutual_information_ranking(X, y, [f"f{i}" for i in range(5)])
    assert ranked[0]["feature"] == "f0"
    assert ranked[0]["rank"] == 1

def test_mi_regression():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X[:, 0] * 2 + np.random.randn(100) * 0.1
    ranked = mutual_information_ranking(X, y)
    assert ranked[0]["rank"] == 1

# --- l1_ranking ---

def test_l1_ranking():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    ranked = l1_ranking(X, y, [f"f{i}" for i in range(5)])
    assert len(ranked) == 5
    # Top features should be f0 and f1
    top_names = {ranked[0]["feature"], ranked[1]["feature"]}
    assert "f0" in top_names or "f1" in top_names

# --- tree_importance_ranking ---

def test_tree_ranking():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    ranked = tree_importance_ranking(X, y, [f"f{i}" for i in range(5)])
    assert len(ranked) == 5
    assert ranked[0]["rank"] == 1

# --- compute_consensus ---

def test_consensus_unanimous():
    rankings = {
        "mi": [{"feature": "f0", "rank": 1}, {"feature": "f1", "rank": 2}],
        "l1": [{"feature": "f0", "rank": 1}, {"feature": "f1", "rank": 2}],
    }
    result = compute_consensus(rankings, top_k=2)
    top = next(c for c in result if c["feature"] == "f0")
    assert top["consensus"] == 2
    assert "★" in top["consensus_str"]

def test_consensus_disagreement():
    rankings = {
        "mi": [{"feature": "f0", "rank": 1}, {"feature": "f1", "rank": 2}],
        "l1": [{"feature": "f2", "rank": 1}, {"feature": "f3", "rank": 2}],
    }
    result = compute_consensus(rankings, top_k=2)
    # Each feature in only 1 method
    for c in result:
        if c["feature"] in ("f0", "f1", "f2", "f3"):
            assert c["consensus"] <= 1

def test_consensus_drop():
    rankings = {
        "mi": [{"feature": "f0", "rank": 1, "score": 0.5}, {"feature": "f1", "rank": 2, "score": 0.3}],
        "l1": [{"feature": "f0", "rank": 1, "score": 0.5}, {"feature": "f1", "rank": 2, "score": 0.3}],
    }
    result = compute_consensus(rankings, top_k=1)
    drop = [c for c in result if c["consensus"] == 0]
    assert any("DROP" in c.get("consensus_str", "") for c in result if c["consensus"] == 0)

def test_consensus_empty():
    assert compute_consensus({}, top_k=5) == []

# --- detect_redundancy ---

def test_redundancy_detected():
    X = np.column_stack([np.arange(10), np.arange(10) * 2, np.random.randn(10)])
    result = detect_redundancy(X, ["a", "b", "c"], threshold=0.95)
    assert len(result) >= 1
    assert result[0]["feature_a"] == "a"
    assert result[0]["feature_b"] == "b"

def test_redundancy_none():
    np.random.seed(42)
    X = np.random.randn(50, 3)
    result = detect_redundancy(X, threshold=0.95)
    assert len(result) == 0

def test_redundancy_single_feature():
    X = np.array([[1], [2], [3]])
    assert detect_redundancy(X) == []

# --- generate_interaction_features ---

def test_interactions_basic():
    features = ["f0", "f1", "f2"]
    result = generate_interaction_features(features, max_interactions=5)
    assert len(result) <= 5
    assert all("type" in r for r in result)

def test_interactions_empty():
    assert generate_interaction_features([]) == []

def test_interactions_limit():
    features = [f"f{i}" for i in range(10)]
    result = generate_interaction_features(features, max_interactions=3)
    assert len(result) == 3

# --- feature_analysis ---

def test_analysis_basic():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    report = feature_analysis(X, y, [f"f{i}" for i in range(5)])
    assert "consensus" in report
    assert "redundant_pairs" in report
    assert report["n_features"] == 5

def test_analysis_no_data():
    report = feature_analysis()
    assert "error" in report

# --- format_feature_report ---

def test_format_basic():
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "n_features": 10,
        "top_k": 5,
        "consensus": [{"feature": "f0", "methods": {"mi": 1, "l1": 1}, "consensus": 2, "consensus_str": "2/2 ★"}],
        "redundant_pairs": [{"feature_a": "f3", "feature_b": "f4", "correlation": 0.97}],
        "interaction_candidates": [{"name": "f0_x_f1", "type": "product", "features": ["f0", "f1"]}],
        "recommendation": "Drop 3 features",
    }
    text = format_feature_report(report)
    assert "Feature Intelligence" in text
    assert "f0" in text
    assert "★" in text

def test_format_error():
    assert "ERROR" in format_feature_report({"error": "No data"})
