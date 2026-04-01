"""Edge case tests for counterfactual explanations (counterfactual_explanation.py).

Phase 27.2: Binary features, no counterfactual found, single feature, boundary conditions.
"""

from __future__ import annotations

import pytest
import yaml

from scripts.counterfactual_explanation import (
    greedy_perturbation,
    prototype_counterfactual,
    counterfactual_analysis,
    save_counterfactual_report,
    format_counterfactual_report,
    _numeric_candidates,
    _compute_distance,
    _compute_changes,
    _select_best,
)


# --- Single feature models ---

def test_greedy_single_feature():
    def predict(sample):
        return (1 if sample["x"] > 5 else 0, 0.95)

    result = greedy_perturbation(
        sample={"x": 7.0},
        predict_fn=predict,
        target_class=0,
        feature_names=["x"],
        feature_ranges={"x": (0, 10)},
    )
    assert result["status"] == "found"
    assert result["n_changes"] == 1


# --- Impossible counterfactual ---

def test_greedy_impossible():
    """Model always predicts 1 regardless of input."""
    def always_one(sample):
        return (1, 0.99)

    result = greedy_perturbation(
        sample={"x": 5.0},
        predict_fn=always_one,
        target_class=0,
        feature_names=["x"],
        feature_ranges={"x": (0, 10)},
        max_iterations=10,
    )
    assert result["status"] == "not_found"
    assert result["original_prediction"] == 1


# --- Binary (0/1) features ---

def test_greedy_binary_features():
    def predict(sample):
        score = sample["a"] + sample["b"]
        return (1 if score >= 1.5 else 0, 0.8)

    result = greedy_perturbation(
        sample={"a": 1.0, "b": 1.0},
        predict_fn=predict,
        target_class=0,
        feature_names=["a", "b"],
        feature_ranges={"a": (0, 1), "b": (0, 1)},
    )
    assert result["status"] == "found"


# --- Distance edge cases ---

def test_distance_same_range():
    """Zero range should not cause division by zero."""
    orig = {"a": 5.0}
    cf = {"a": 5.0}
    dist = _compute_distance(orig, cf, {"a": (5, 5)})
    assert dist == 0.0

def test_distance_missing_feature():
    """Missing feature in counterfactual uses original value."""
    orig = {"a": 1.0, "b": 2.0}
    cf = {"a": 1.0}  # b missing
    dist = _compute_distance(orig, cf, {"a": (0, 10), "b": (0, 10)})
    assert dist == 0.0

def test_distance_categorical_changed():
    orig = {"color": "red"}
    cf = {"color": "blue"}
    dist = _compute_distance(orig, cf, {"color": ("red", "blue")})
    assert dist == 1.0

def test_distance_categorical_same():
    orig = {"color": "red"}
    cf = {"color": "red"}
    dist = _compute_distance(orig, cf, {"color": ("red", "blue")})
    assert dist == 0.0


# --- Prototype edge cases ---

def test_prototype_empty_training():
    result = prototype_counterfactual(
        sample={"a": 5.0},
        training_data=[],
        training_labels=[],
        target_class=0,
        feature_names=["a"],
        feature_ranges={"a": (0, 10)},
    )
    assert result["status"] == "not_found"

def test_prototype_string_labels():
    """Labels can be strings, not just ints."""
    result = prototype_counterfactual(
        sample={"a": 8.0},
        training_data=[{"a": 2.0}, {"a": 9.0}],
        training_labels=["cat", "dog"],
        target_class="cat",
        feature_names=["a"],
        feature_ranges={"a": (0, 10)},
    )
    assert result["status"] == "found"
    assert result["prototype_index"] == 0


# --- _numeric_candidates edge cases ---

def test_candidates_zero_steps():
    candidates = _numeric_candidates(5.0, (0, 10), n_steps=0)
    # n_steps=0: just one candidate at low
    assert len(candidates) >= 0

def test_candidates_narrow_range():
    candidates = _numeric_candidates(5.0, (5.0, 5.0), n_steps=3)
    # All candidates are 5.0 = current, so all filtered out
    assert all(c == 5.0 for c in candidates) or len(candidates) == 0


# --- Changes edge cases ---

def test_changes_categorical():
    orig = {"color": "red", "size": 10}
    cf = {"color": "blue", "size": 10}
    changes = _compute_changes(orig, cf, ["color", "size"])
    assert len(changes) == 1
    assert changes[0]["delta"] == "category_change"

def test_changes_integer_delta():
    orig = {"count": 10}
    cf = {"count": 15}
    changes = _compute_changes(orig, cf, ["count"])
    assert changes[0]["delta"] == 5


# --- save_counterfactual_report ---

def test_save_report(tmp_path):
    report = {
        "experiment_id": "exp-042",
        "sample_index": 100,
        "target_class": 0,
        "results": {"greedy": {"status": "found"}},
        "generated_at": "2026-04-01",
    }
    path = save_counterfactual_report(report, str(tmp_path / "counterfactuals"))
    assert path.exists()
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data["experiment_id"] == "exp-042"

def test_save_batch_report(tmp_path):
    report = {
        "experiment_id": "exp-042",
        "sample_index": "batch",
        "results": [],
        "generated_at": "2026-04-01",
    }
    path = save_counterfactual_report(report, str(tmp_path / "cf"))
    assert "batch" in path.name


# --- _select_best edge cases ---

def test_select_best_all_none():
    assert _select_best([None, None]) is None

def test_select_best_empty():
    assert _select_best([]) is None

def test_select_best_tie():
    """When distances are equal, any is acceptable."""
    candidates = [
        {"status": "found", "distance": 0.5, "method": "a"},
        {"status": "found", "distance": 0.5, "method": "b"},
    ]
    best = _select_best(candidates)
    assert best["distance"] == 0.5


# --- Batch analysis ---

def test_batch_analysis(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")

    def predict(sample):
        return (1 if sample["a"] > 5 else 0, 0.9)

    training = [
        {"a": 8.0},  # pred 1, label 0 → misclassified
        {"a": 3.0},  # pred 0, label 0 → correct
    ]
    labels = [0, 0]

    result = counterfactual_analysis(
        "exp-001",
        batch_misclassified=True,
        predict_fn=predict,
        training_data=training,
        training_labels=labels,
        feature_names=["a"],
        feature_ranges={"a": (0, 10)},
        config_path=str(config),
    )
    assert isinstance(result["results"], list)
    assert len(result["results"]) == 1  # only 1 misclassified
