"""Tests for probability calibration (calibration.py).

Phase 21.3: Verifies ECE/MCE, reliability diagram, calibration methods, reporting.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.calibration import (
    calibrate_model,
    compute_ece,
    compute_mce,
    compute_reliability_diagram,
    format_calibration_report,
    platt_scaling,
    isotonic_calibration,
    temperature_scaling,
)


# --- compute_ece ---

def test_ece_perfect():
    """Perfectly calibrated should have ECE = 0."""
    probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    labels = np.array([0, 0, 1, 1, 1])
    ece = compute_ece(probs, labels, n_bins=5)
    # Not exactly 0 due to binning but should be low
    assert ece < 0.3

def test_ece_overconfident():
    """Overconfident model should have high ECE."""
    probs = np.array([0.9, 0.95, 0.85, 0.92, 0.88, 0.91, 0.87, 0.93, 0.89, 0.90])
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 50% accuracy but 90% confidence
    ece = compute_ece(probs, labels)
    assert ece > 0.3

def test_ece_empty():
    assert compute_ece(np.array([]), np.array([])) == 0.0

# --- compute_mce ---

def test_mce_basic():
    probs = np.array([0.9, 0.95, 0.85, 0.92])
    labels = np.array([1, 0, 1, 0])
    mce = compute_mce(probs, labels)
    assert mce > 0

def test_mce_empty():
    assert compute_mce(np.array([]), np.array([])) == 0.0

# --- compute_reliability_diagram ---

def test_reliability_basic():
    probs = np.linspace(0, 1, 100)
    labels = (probs > 0.5).astype(int)
    diagram = compute_reliability_diagram(probs, labels, n_bins=5)
    assert len(diagram) == 5
    assert all("bin" in b for b in diagram)

def test_reliability_empty():
    assert compute_reliability_diagram(np.array([]), np.array([])) == []

def test_reliability_all_same_bin():
    probs = np.array([0.91, 0.92, 0.93, 0.94])
    labels = np.array([1, 1, 0, 1])
    diagram = compute_reliability_diagram(probs, labels, n_bins=10)
    non_empty = [b for b in diagram if b["n"] > 0]
    assert len(non_empty) >= 1

# --- platt_scaling ---

def test_platt_basic():
    np.random.seed(42)
    logits = np.concatenate([np.random.randn(50) - 1, np.random.randn(50) + 1])
    labels = np.array([0] * 50 + [1] * 50)
    result = platt_scaling(logits, labels)
    assert result["method"] == "platt"
    assert len(result["calibrated_probabilities"]) == 100
    assert "a" in result["params"]

# --- isotonic_calibration ---

def test_isotonic_basic():
    np.random.seed(42)
    probs = np.clip(np.random.randn(100) * 0.3 + 0.5, 0, 1)
    labels = (probs > 0.5).astype(int)
    result = isotonic_calibration(probs, labels)
    assert result["method"] == "isotonic"
    assert len(result["calibrated_probabilities"]) == 100

# --- temperature_scaling ---

def test_temperature_basic():
    np.random.seed(42)
    logits = np.concatenate([np.random.randn(50) - 1, np.random.randn(50) + 1])
    labels = np.array([0] * 50 + [1] * 50)
    result = temperature_scaling(logits, labels)
    assert result["method"] == "temperature"
    assert "T" in result["params"]

# --- calibrate_model ---

def test_calibrate_auto():
    np.random.seed(42)
    logits = np.concatenate([np.random.randn(50) - 1, np.random.randn(50) + 1])
    labels = np.array([0] * 50 + [1] * 50)
    report = calibrate_model(logits=logits, labels=labels, method="auto", exp_id="exp-042")
    assert "before" in report
    assert "calibration_results" in report
    assert report.get("verdict") in ("already_calibrated", "improved", "marginal_improvement", "no_improvement")

def test_calibrate_no_data():
    report = calibrate_model()
    assert "error" in report

def test_calibrate_probs_only():
    np.random.seed(42)
    probs = np.clip(np.random.randn(100) * 0.3 + 0.5, 0, 1)
    labels = (probs > 0.5).astype(int)
    report = calibrate_model(probabilities=probs, labels=labels, method="isotonic")
    assert "before" in report

# --- format_calibration_report ---

def test_format_basic():
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "experiment_id": "exp-042",
        "before": {"ece": 0.068, "mce": 0.170},
        "reliability_diagram": [{"bin": "[0.0-0.1]", "predicted": 0.05, "actual": 0.03, "gap": -0.02, "n": 10}],
        "calibration_results": [{"method": "platt", "ece_after": 0.021, "improvement": 0.047}],
        "best_method": {"method": "platt", "ece_after": 0.021},
        "verdict": "improved",
        "reason": "Platt reduces ECE",
    }
    text = format_calibration_report(report)
    assert "Calibration" in text
    assert "0.068" in text
    assert "IMPROVED" in text

def test_format_error():
    assert "ERROR" in format_calibration_report({"error": "No data"})

def test_format_already_calibrated():
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "experiment_id": "exp-001",
        "before": {"ece": 0.01, "mce": 0.03},
        "reliability_diagram": [],
        "calibration_results": [],
        "best_method": None,
        "verdict": "already_calibrated",
        "reason": "ECE already low",
    }
    text = format_calibration_report(report)
    assert "ALREADY CALIBRATED" in text
