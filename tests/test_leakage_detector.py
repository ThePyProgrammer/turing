"""Tests for targeted leakage detection (leakage_detector.py).

Phase 20.3: Verifies correlation checks, single-feature analysis,
overlap detection, and report formatting.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.leakage_detector import (
    check_feature_target_correlation,
    check_single_feature_predictiveness,
    check_train_test_overlap,
    format_leakage_report,
    run_leakage_scan,
    _simple_single_feature_score,
)


# --- check_feature_target_correlation ---

def test_correlation_detects_leakage():
    """Feature identical to target should be flagged."""
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    flags = check_feature_target_correlation(X, y, threshold=0.95)
    assert len(flags) == 1
    assert flags[0]["correlation"] > 0.99

def test_correlation_no_leakage():
    """Random feature should not be flagged."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    flags = check_feature_target_correlation(X, y, threshold=0.95)
    assert len(flags) == 0

def test_correlation_with_names():
    """Should use feature names."""
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([1, 2, 3, 4], dtype=float)
    flags = check_feature_target_correlation(X, y, ["my_feature"], threshold=0.95)
    assert flags[0]["feature"] == "my_feature"

def test_correlation_constant_feature():
    """Constant feature should be skipped."""
    X = np.array([[5], [5], [5], [5]], dtype=float)
    y = np.array([1, 2, 3, 4], dtype=float)
    flags = check_feature_target_correlation(X, y, threshold=0.95)
    assert len(flags) == 0

# --- check_single_feature_predictiveness ---

def test_single_feature_detects_leakage():
    """Perfect feature should be flagged."""
    X = np.array([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=float)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    flags = check_single_feature_predictiveness(X, y, full_model_score=0.85, ratio_threshold=0.80)
    assert len(flags) >= 1

def test_single_feature_no_leakage():
    """Weak feature should not be flagged."""
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = np.random.randint(0, 2, 100)
    flags = check_single_feature_predictiveness(X, y, full_model_score=0.90, ratio_threshold=0.80)
    assert len(flags) == 0

# --- _simple_single_feature_score ---

def test_score_classification_perfect():
    """Perfect feature should score ~1.0."""
    feature = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1])
    score = _simple_single_feature_score(feature.reshape(-1, 1), y, "classification")
    assert score > 0.9

def test_score_regression():
    """Should return R² for regression."""
    feature = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    score = _simple_single_feature_score(feature.reshape(-1, 1), y, "regression")
    assert score > 0.9

def test_score_constant_feature():
    """Constant feature should score 0."""
    feature = np.array([1, 1, 1, 1], dtype=float)
    y = np.array([0, 1, 0, 1])
    score = _simple_single_feature_score(feature.reshape(-1, 1), y, "regression")
    assert score == 0.0

# --- check_train_test_overlap ---

def test_overlap_detected():
    """Identical rows should be detected."""
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[1, 2], [7, 8]])
    result = check_train_test_overlap(X_train, X_test)
    assert result["overlapping_samples"] == 1
    assert result["status"] in ("warn", "fail")

def test_no_overlap():
    """No overlapping rows should pass."""
    X_train = np.array([[1, 2], [3, 4]])
    X_test = np.array([[5, 6], [7, 8]])
    result = check_train_test_overlap(X_train, X_test)
    assert result["overlapping_samples"] == 0
    assert result["status"] == "pass"

def test_full_overlap():
    """Complete overlap should fail."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    result = check_train_test_overlap(X, X)
    assert result["overlapping_samples"] == 3
    assert result["status"] == "fail"

# --- run_leakage_scan ---

def test_scan_with_data():
    """Should run scan with provided data."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    report = run_leakage_scan(X=X, y=y, task_type="classification")
    assert report["verdict"] in ("clean", "suspicious", "leakage_detected")
    assert len(report["checks"]) >= 1

def test_scan_no_data():
    """No data should return error."""
    report = run_leakage_scan()
    assert "error" in report

def test_scan_deep_mode():
    """Deep mode should include single-feature analysis."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = np.random.randint(0, 2, 50)
    report = run_leakage_scan(X=X, y=y, full_model_score=0.85, deep=True)
    check_names = [c.get("check") for c in report.get("checks", [])]
    assert "single_feature_predictiveness" in check_names

def test_scan_with_overlap():
    """Should check overlap when train/test provided."""
    X_train = np.array([[1, 2], [3, 4]])
    X_test = np.array([[5, 6], [7, 8]])
    report = run_leakage_scan(X_train=X_train, X_test=X_test)
    check_names = [c.get("check") for c in report.get("checks", [])]
    assert "train_test_overlap" in check_names

# --- format_leakage_report ---

def test_format_clean():
    report = {
        "scanned_at": "2026-01-01T00:00:00",
        "deep_mode": False,
        "verdict": "clean",
        "checks": [{"check": "correlation", "status": "pass", "reason": "OK", "flags": []}],
    }
    text = format_leakage_report(report)
    assert "CLEAN" in text

def test_format_leakage():
    report = {
        "scanned_at": "2026-01-01T00:00:00",
        "deep_mode": True,
        "verdict": "leakage_detected",
        "checks": [{
            "check": "correlation",
            "status": "fail",
            "reason": "1 feature flagged",
            "flags": [{"feature": "id_col", "reason": "Correlation 0.99 with target"}],
        }],
    }
    text = format_leakage_report(report)
    assert "LEAKAGE" in text
    assert "id_col" in text

def test_format_error():
    text = format_leakage_report({"error": "No data"})
    assert "ERROR" in text
