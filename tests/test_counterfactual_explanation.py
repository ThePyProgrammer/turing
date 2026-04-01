"""Tests for counterfactual explanations (counterfactual_explanation.py).

Phase 27.2: Verifies perturbation search, prototype matching, distance, reporting.
"""

from __future__ import annotations

import pytest

from scripts.counterfactual_explanation import (
    greedy_perturbation,
    prototype_counterfactual,
    counterfactual_analysis,
    format_counterfactual_report,
    _numeric_candidates,
    _categorical_candidates,
    _compute_distance,
    _compute_changes,
    _select_best,
)


# --- _numeric_candidates ---

def test_numeric_candidates_basic():
    candidates = _numeric_candidates(0.5, (0.0, 1.0), n_steps=4)
    assert len(candidates) > 0
    assert 0.5 not in candidates  # current value excluded

def test_numeric_candidates_range():
    candidates = _numeric_candidates(5.0, (0.0, 10.0), n_steps=5)
    assert all(0.0 <= c <= 10.0 for c in candidates)


# --- _categorical_candidates ---

def test_categorical_list():
    candidates = _categorical_candidates("color", "red", ("red", "blue", "green"))
    assert "red" not in candidates
    assert "blue" in candidates
    assert "green" in candidates

def test_categorical_none_range():
    assert _categorical_candidates("color", "red", None) == []


# --- _compute_distance ---

def test_distance_zero():
    sample = {"a": 1.0, "b": 2.0}
    dist = _compute_distance(sample, sample, {"a": (0, 10), "b": (0, 10)})
    assert dist == 0.0

def test_distance_positive():
    orig = {"a": 0.0, "b": 0.0}
    cf = {"a": 10.0, "b": 0.0}
    dist = _compute_distance(orig, cf, {"a": (0, 10), "b": (0, 10)})
    assert dist == pytest.approx(1.0)

def test_distance_normalized():
    """Distance should be normalized by feature range."""
    orig = {"a": 0.0}
    cf = {"a": 5.0}
    dist_wide = _compute_distance(orig, cf, {"a": (0, 100)})
    dist_narrow = _compute_distance(orig, cf, {"a": (0, 10)})
    assert dist_narrow > dist_wide


# --- _compute_changes ---

def test_changes_basic():
    orig = {"a": 1.0, "b": 2.0, "c": 3.0}
    cf = {"a": 1.0, "b": 5.0, "c": 3.0}
    changes = _compute_changes(orig, cf, ["a", "b", "c"])
    assert len(changes) == 1
    assert changes[0]["feature"] == "b"
    assert changes[0]["delta"] == 3.0

def test_changes_all_same():
    orig = {"a": 1.0}
    changes = _compute_changes(orig, orig, ["a"])
    assert len(changes) == 0


# --- greedy_perturbation ---

def _simple_predict(sample):
    """Binary classifier: class 1 if feature 'a' > 5, else class 0."""
    return (1 if sample["a"] > 5 else 0, 0.9)

def test_greedy_finds_cf():
    sample = {"a": 8.0, "b": 3.0}
    result = greedy_perturbation(
        sample=sample,
        predict_fn=_simple_predict,
        target_class=0,
        feature_names=["a", "b"],
        feature_ranges={"a": (0, 10), "b": (0, 10)},
    )
    assert result["status"] == "found"
    assert result["counterfactual_prediction"] == 0
    assert result["n_changes"] >= 1

def test_greedy_already_target():
    sample = {"a": 3.0, "b": 3.0}
    result = greedy_perturbation(
        sample=sample,
        predict_fn=_simple_predict,
        target_class=0,
        feature_names=["a", "b"],
        feature_ranges={"a": (0, 10), "b": (0, 10)},
    )
    assert result["status"] == "already_target"

def test_greedy_with_categorical():
    def predict_cat(sample):
        return (1 if sample["a"] > 5 and sample["color"] == "red" else 0, 0.8)

    sample = {"a": 8.0, "color": "red"}
    result = greedy_perturbation(
        sample=sample,
        predict_fn=predict_cat,
        target_class=0,
        feature_names=["a", "color"],
        feature_ranges={"a": (0, 10), "color": ("red", "blue", "green")},
        categorical_features=["color"],
    )
    assert result["status"] == "found"


# --- prototype_counterfactual ---

def test_prototype_finds_nearest():
    sample = {"a": 8.0, "b": 3.0}
    training = [
        {"a": 2.0, "b": 2.0},
        {"a": 4.0, "b": 3.0},  # nearest class-0 sample
        {"a": 9.0, "b": 3.0},
    ]
    labels = [0, 0, 1]
    result = prototype_counterfactual(
        sample=sample,
        training_data=training,
        training_labels=labels,
        target_class=0,
        feature_names=["a", "b"],
        feature_ranges={"a": (0, 10), "b": (0, 10)},
    )
    assert result["status"] == "found"
    assert result["prototype_index"] == 1  # nearest

def test_prototype_no_target_class():
    sample = {"a": 8.0}
    result = prototype_counterfactual(
        sample=sample,
        training_data=[{"a": 2.0}],
        training_labels=[1],
        target_class=0,
        feature_names=["a"],
        feature_ranges={"a": (0, 10)},
    )
    assert result["status"] == "not_found"


# --- counterfactual_analysis ---

def test_analysis_no_sample():
    result = counterfactual_analysis("exp-001")
    assert "error" in result

def test_analysis_no_predict_fn(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    result = counterfactual_analysis(
        "exp-001", sample_data={"a": 5.0},
        config_path=str(config),
    )
    assert "error" in result

def test_analysis_no_features(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("")
    result = counterfactual_analysis(
        "exp-001",
        sample_data={"a": 5.0},
        predict_fn=_simple_predict,
        config_path=str(config),
    )
    assert "error" in result

def test_analysis_full(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    result = counterfactual_analysis(
        "exp-001",
        sample_data={"a": 8.0, "b": 3.0},
        target_class=0,
        predict_fn=_simple_predict,
        feature_names=["a", "b"],
        feature_ranges={"a": (0, 10), "b": (0, 10)},
        config_path=str(config),
    )
    assert "results" in result
    assert result["results"]["greedy"]["status"] == "found"


# --- _select_best ---

def test_select_best_picks_closest():
    candidates = [
        {"status": "found", "distance": 0.5},
        {"status": "found", "distance": 0.3},
    ]
    best = _select_best(candidates)
    assert best["distance"] == 0.3

def test_select_best_ignores_not_found():
    candidates = [
        {"status": "not_found"},
        {"status": "found", "distance": 0.5},
    ]
    best = _select_best(candidates)
    assert best["distance"] == 0.5

def test_select_best_none():
    assert _select_best([{"status": "not_found"}, None]) is None


# --- format_counterfactual_report ---

def test_format_report_found():
    report = {
        "experiment_id": "exp-042",
        "sample_index": 1247,
        "target_class": 0,
        "results": {
            "greedy": {"status": "found", "distance": 0.42, "n_changes": 3,
                       "changes": [{"feature": "amount", "original": 4230, "counterfactual": 1850, "delta": -2380}]},
            "prototype": None,
            "best": {"status": "found", "distance": 0.42, "n_changes": 3,
                     "changes": [{"feature": "amount", "original": 4230, "counterfactual": 1850, "delta": -2380}]},
        },
        "generated_at": "2026-04-01T00:00:00Z",
    }
    text = format_counterfactual_report(report)
    assert "Counterfactual" in text
    assert "amount" in text
    assert "0.42" in text

def test_format_report_not_found():
    report = {
        "experiment_id": "exp-042",
        "sample_index": 1247,
        "target_class": 0,
        "results": {
            "greedy": {"status": "not_found"},
            "prototype": None,
            "best": None,
        },
        "generated_at": "2026-04-01T00:00:00Z",
    }
    text = format_counterfactual_report(report)
    assert "No counterfactual found" in text

def test_format_error():
    text = format_counterfactual_report({"error": "no model"})
    assert "ERROR" in text

def test_format_batch():
    report = {
        "experiment_id": "exp-042",
        "sample_index": "batch",
        "target_class": None,
        "results": [
            {"status": "found", "sample_index": 0},
            {"status": "not_found", "sample_index": 1},
        ],
        "generated_at": "2026-04-01T00:00:00Z",
    }
    text = format_counterfactual_report(report)
    assert "Batch" in text
    assert "1/2" in text
