"""Tests for detailed metric decomposition (evaluate_detailed).

Phase 3.1: Verifies per-class metrics and confusion matrix — the
tools that tell the agent WHERE the model fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import evaluate_detailed


def test_detailed_contains_all_sections():
    """Result should have aggregate, per_class, and confusion_matrix."""
    preds = np.array([0, 1, 0, 1])
    truth = np.array([0, 1, 1, 0])
    result = evaluate_detailed(preds, truth)
    assert "aggregate" in result
    assert "per_class" in result
    assert "confusion_matrix" in result


def test_detailed_aggregate_matches_evaluate():
    """Aggregate section should match evaluate_model output."""
    preds = np.array([0, 1, 0, 1, 0])
    truth = np.array([0, 1, 1, 1, 0])
    config = {"evaluation": {"metrics": ["accuracy"]}}
    result = evaluate_detailed(preds, truth, config)
    assert result["aggregate"]["accuracy"] == 0.8


def test_per_class_has_all_classes():
    """Per-class should have an entry for every unique ground truth class."""
    preds = np.array([0, 1, 2, 0, 1])
    truth = np.array([0, 1, 2, 1, 2])
    result = evaluate_detailed(preds, truth)
    assert "0" in result["per_class"]
    assert "1" in result["per_class"]
    assert "2" in result["per_class"]


def test_per_class_perfect():
    """Perfect predictions should give precision=recall=f1=1.0 for all classes."""
    preds = np.array([0, 1, 0, 1])
    truth = np.array([0, 1, 0, 1])
    result = evaluate_detailed(preds, truth)
    for cls_data in result["per_class"].values():
        assert cls_data["precision"] == 1.0
        assert cls_data["recall"] == 1.0
        assert cls_data["f1"] == 1.0


def test_per_class_support():
    """Support should equal the count of that class in ground truth."""
    preds = np.array([0, 0, 0, 1, 1])
    truth = np.array([0, 0, 1, 1, 1])
    result = evaluate_detailed(preds, truth)
    assert result["per_class"]["0"]["support"] == 2
    assert result["per_class"]["1"]["support"] == 3


def test_confusion_matrix_shape():
    """Confusion matrix should be NxN for N classes."""
    preds = np.array([0, 1, 2, 0, 1, 2])
    truth = np.array([0, 1, 2, 2, 0, 1])
    result = evaluate_detailed(preds, truth)
    cm = result["confusion_matrix"]
    assert len(cm) == 3  # 3 classes
    for row in cm.values():
        assert len(row) == 3


def test_confusion_matrix_diagonal_perfect():
    """Perfect predictions should only have values on the diagonal."""
    preds = np.array([0, 1, 0, 1])
    truth = np.array([0, 1, 0, 1])
    cm = evaluate_detailed(preds, truth)["confusion_matrix"]
    assert cm["0"]["0"] == 2
    assert cm["0"]["1"] == 0
    assert cm["1"]["0"] == 0
    assert cm["1"]["1"] == 2


def test_confusion_matrix_sums_to_total():
    """All confusion matrix entries should sum to total samples."""
    preds = np.array([0, 1, 0, 1, 0])
    truth = np.array([0, 1, 1, 0, 0])
    cm = evaluate_detailed(preds, truth)["confusion_matrix"]
    total = sum(v for row in cm.values() for v in row.values())
    assert total == 5
