"""Tests for error analysis (diagnose_errors.py).

Phase 11.1: Verifies confusion matrix computation, per-class metrics,
failure mode identification, regression error analysis, and hypothesis
generation.
"""

from __future__ import annotations

import yaml
import pytest

from scripts.diagnose_errors import (
    analyze_regression_errors,
    compute_confusion_matrix,
    format_diagnosis_report,
    generate_hypotheses_from_modes,
    identify_failure_modes,
    queue_hypotheses,
    save_diagnosis,
)


# --- compute_confusion_matrix ---


def test_confusion_matrix_basic():
    """Should compute correct matrix for simple case."""
    y_true = ["cat", "dog", "cat", "dog", "cat"]
    y_pred = ["cat", "dog", "dog", "dog", "cat"]
    result = compute_confusion_matrix(y_true, y_pred)

    assert result["total_samples"] == 5
    assert result["total_errors"] == 1
    assert result["error_rate"] == 0.2
    assert len(result["classes"]) == 2


def test_confusion_matrix_per_class():
    """Should compute correct precision/recall per class."""
    y_true = ["A", "A", "A", "B", "B"]
    y_pred = ["A", "A", "B", "B", "A"]
    result = compute_confusion_matrix(y_true, y_pred)

    # Class A: TP=2, FP=1, FN=1
    assert result["per_class"]["A"]["precision"] == pytest.approx(2 / 3, abs=0.01)
    assert result["per_class"]["A"]["recall"] == pytest.approx(2 / 3, abs=0.01)


def test_confusion_matrix_perfect():
    """Perfect predictions should have zero errors."""
    y_true = ["A", "B", "C"]
    y_pred = ["A", "B", "C"]
    result = compute_confusion_matrix(y_true, y_pred)

    assert result["total_errors"] == 0
    assert result["error_rate"] == 0.0
    assert len(result["most_confused"]) == 0


def test_confusion_matrix_confused_pairs():
    """Should identify the most confused pairs."""
    y_true = ["A", "A", "A", "B", "B", "B"]
    y_pred = ["A", "B", "B", "B", "A", "A"]
    result = compute_confusion_matrix(y_true, y_pred)

    confused = result["most_confused"]
    assert len(confused) > 0
    # Both A->B and B->A should appear
    pairs = [(c["true_class"], c["predicted_class"]) for c in confused]
    assert ("A", "B") in pairs
    assert ("B", "A") in pairs


def test_confusion_matrix_multiclass():
    """Should work with 3+ classes."""
    y_true = ["A", "B", "C", "A", "B", "C"]
    y_pred = ["A", "C", "C", "B", "B", "A"]
    result = compute_confusion_matrix(y_true, y_pred)

    assert len(result["classes"]) == 3
    assert result["total_errors"] == 3


# --- analyze_regression_errors ---


def test_regression_errors_basic():
    """Should compute correct error statistics."""
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.1, 2.2, 2.8, 4.1, 5.3]
    result = analyze_regression_errors(y_true, y_pred)

    assert result["mean_absolute_error"] > 0
    assert result["median_absolute_error"] > 0
    assert "worst_predictions" in result
    assert len(result["worst_predictions"]) > 0


def test_regression_errors_perfect():
    """Perfect predictions should have near-zero error."""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]
    result = analyze_regression_errors(y_true, y_pred)

    assert result["mean_absolute_error"] == 0.0
    assert result["bias"] == 0.0


def test_regression_errors_bias():
    """Should detect systematic over/under prediction."""
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.5, 2.5, 3.5, 4.5, 5.5]  # Consistently over-predicts by 0.5
    result = analyze_regression_errors(y_true, y_pred)

    assert result["bias"] == pytest.approx(0.5, abs=0.01)


def test_regression_errors_worst_predictions():
    """Worst predictions should be ordered by residual."""
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.0, 2.0, 3.0, 4.0, 10.0]  # Last one is way off
    result = analyze_regression_errors(y_true, y_pred, top_n=3)

    worst = result["worst_predictions"]
    assert worst[0]["index"] == 4
    assert worst[0]["residual"] == 5.0


def test_regression_feature_range_bias():
    """Should detect when errors correlate with feature ranges."""
    y_true = [1.0] * 20
    y_pred = [1.0] * 10 + [2.0] * 10  # Errors only in second half
    features = [{"x": i} for i in range(20)]  # x correlates with errors
    result = analyze_regression_errors(y_true, y_pred, features)

    # Should find that high x values have higher error
    assert "feature_range_bias" in result


# --- identify_failure_modes ---


def test_failure_modes_from_confusion():
    """Should identify failure modes from confused pairs."""
    confusion = {
        "most_confused": [
            {"true_class": "cat", "predicted_class": "dog", "count": 42, "pct_of_true": 30},
        ],
        "per_class": {
            "cat": {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 100, "tp": 70, "fp": 20, "fn": 30},
            "dog": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 100, "tp": 80, "fp": 10, "fn": 20},
        },
        "total_errors": 50,
    }
    modes = identify_failure_modes(confusion_data=confusion)
    assert len(modes) > 0
    assert modes[0]["type"] == "class_confusion"
    assert "cat" in modes[0]["description"] and "dog" in modes[0]["description"]


def test_failure_modes_low_recall():
    """Should flag classes with very low recall."""
    confusion = {
        "most_confused": [],
        "per_class": {
            "rare_class": {"precision": 0.8, "recall": 0.3, "f1": 0.44, "support": 20, "tp": 6, "fp": 2, "fn": 14},
        },
        "total_errors": 20,
    }
    modes = identify_failure_modes(confusion_data=confusion)
    low_recall = [m for m in modes if m["type"] == "low_recall"]
    assert len(low_recall) > 0


def test_failure_modes_from_regression():
    """Should identify failure modes from regression analysis."""
    regression = {
        "feature_range_bias": [
            {"feature": "age", "worst_quartile": "Q4", "best_quartile": "Q1",
             "error_ratio": 3.5, "description": "Model error 3.5x higher in Q4"},
        ],
        "bias": 0.5,
        "std_error": 0.2,
        "worst_predictions": [{"index": 42}],
    }
    modes = identify_failure_modes(regression_data=regression)
    assert len(modes) > 0


# --- generate_hypotheses_from_modes ---


def test_hypotheses_generation():
    """Should generate hypothesis entries from failure modes."""
    modes = [
        {
            "id": "fm-001",
            "type": "class_confusion",
            "description": "Model confuses cat and dog",
            "affected_samples": 42,
            "auto_hypothesis": "Add features to distinguish cat from dog",
        },
    ]
    hyps = generate_hypotheses_from_modes(modes)
    assert len(hyps) == 1
    assert hyps[0]["source"] == "diagnose"
    assert hyps[0]["status"] == "queued"
    assert hyps[0]["priority"] == "high"


def test_hypotheses_no_auto():
    """Should skip modes without auto_hypothesis."""
    modes = [{"id": "fm-001", "type": "unknown", "description": "test"}]
    hyps = generate_hypotheses_from_modes(modes)
    assert len(hyps) == 0


# --- queue_hypotheses ---


def test_queue_hypotheses(tmp_path):
    """Should append to hypothesis queue file."""
    queue_path = str(tmp_path / "hypotheses.yaml")
    hyps = [{"id": "hyp-diag-001", "description": "test", "source": "diagnose", "status": "queued"}]
    n = queue_hypotheses(hyps, queue_path)
    assert n == 1

    with open(queue_path) as f:
        data = yaml.safe_load(f)
    assert len(data) == 1


def test_queue_hypotheses_dedup(tmp_path):
    """Should not add duplicate hypothesis IDs."""
    queue_path = str(tmp_path / "hypotheses.yaml")
    hyps = [{"id": "hyp-diag-001", "description": "test", "source": "diagnose", "status": "queued"}]
    queue_hypotheses(hyps, queue_path)
    n = queue_hypotheses(hyps, queue_path)  # Same ID again
    assert n == 0


# --- save_diagnosis ---


def test_save_diagnosis(tmp_path):
    """Should save YAML and return correct path."""
    diag = {"experiment_id": "exp-042", "failure_modes": []}
    filepath = save_diagnosis(diag, str(tmp_path))
    assert filepath.exists()
    assert filepath.name == "exp-042-diagnosis.yaml"


# --- format_diagnosis_report ---


def test_format_report_classification():
    """Should format classification diagnosis."""
    diag = {
        "experiment_id": "exp-042",
        "task_type": "classification",
        "confusion_matrix": {
            "total_samples": 100,
            "total_errors": 15,
            "error_rate": 0.15,
            "per_class": {
                "A": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 50},
                "B": {"precision": 0.85, "recall": 0.9, "f1": 0.87, "support": 50},
            },
            "most_confused": [
                {"true_class": "A", "predicted_class": "B", "count": 8, "pct_of_true": 16},
            ],
        },
        "failure_modes": [
            {"id": "fm-001", "type": "class_confusion",
             "description": "Model confuses A and B",
             "affected_samples": 8, "suggested_fix": "Add features",
             "auto_hypothesis": "Test hypothesis"},
        ],
        "auto_hypotheses": [],
    }
    report = format_diagnosis_report(diag)
    assert "exp-042" in report
    assert "Per-Class Performance" in report
    assert "Failure Modes" in report


def test_format_report_regression():
    """Should format regression diagnosis."""
    diag = {
        "experiment_id": "exp-001",
        "task_type": "regression",
        "regression_analysis": {
            "mean_absolute_error": 0.15,
            "median_absolute_error": 0.10,
            "p90_error": 0.30,
            "p95_error": 0.40,
            "bias": -0.05,
        },
        "failure_modes": [],
        "auto_hypotheses": [],
    }
    report = format_diagnosis_report(diag)
    assert "Error Distribution" in report
    assert "0.15" in report


def test_format_report_error():
    """Should handle error case."""
    diag = {"error": "No experiment found", "experiment_id": "exp-999"}
    report = format_diagnosis_report(diag)
    assert "ERROR" in report
