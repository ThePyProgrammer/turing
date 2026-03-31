"""Tests for equivalence_checker.py."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.equivalence_checker import (
    EXACT_TOLERANCE,
    FLOAT32_TOLERANCE,
    compare_outputs,
    format_equivalence_report,
    generate_test_data,
)


def test_compare_identical():
    """Identical outputs should be equivalent."""
    a = [0.85, 0.90, 0.75]
    result = compare_outputs(a, a)
    assert result["verdict"] == "equivalent"
    assert result["max_delta"] == 0.0


def test_compare_within_exact_tolerance():
    """Tiny differences should be equivalent."""
    a = [0.850000000, 0.900000000]
    b = [0.850000000001, 0.900000000001]
    result = compare_outputs(a, b)
    assert result["verdict"] == "equivalent"


def test_compare_approximately_equivalent():
    """Small differences within tolerance should be approx equivalent."""
    a = [0.850, 0.900, 0.750]
    b = [0.850001, 0.900002, 0.750001]
    result = compare_outputs(a, b, tolerance=1e-3)
    assert result["verdict"] in ("equivalent", "approximately_equivalent")


def test_compare_divergent():
    """Large differences should be divergent."""
    a = [0.85, 0.90]
    b = [0.70, 0.75]
    result = compare_outputs(a, b, tolerance=1e-5)
    assert result["verdict"] == "divergent"
    assert result["n_divergent"] == 2


def test_compare_shape_mismatch():
    """Different-length outputs should be divergent."""
    result = compare_outputs([0.85, 0.90], [0.85])
    assert result["verdict"] == "divergent"
    assert "Shape mismatch" in result["reason"]


def test_compare_single_sample():
    """Should work with single sample."""
    result = compare_outputs([0.85], [0.85])
    assert result["verdict"] == "equivalent"
    assert result["n_samples"] == 1


def test_compare_numpy_arrays():
    """Should handle numpy arrays."""
    a = np.array([0.85, 0.90, 0.75])
    b = np.array([0.85, 0.90, 0.75])
    result = compare_outputs(a, b)
    assert result["verdict"] == "equivalent"


def test_compare_2d_arrays():
    """Should flatten and compare 2D arrays."""
    a = np.array([[0.85, 0.15], [0.90, 0.10]])
    b = np.array([[0.85, 0.15], [0.90, 0.10]])
    result = compare_outputs(a, b)
    assert result["verdict"] == "equivalent"


def test_compare_custom_tolerance():
    """Should respect custom tolerance."""
    a = [0.85]
    b = [0.86]
    result_tight = compare_outputs(a, b, tolerance=0.001)
    result_loose = compare_outputs(a, b, tolerance=0.1)
    assert result_tight["verdict"] == "divergent"
    assert result_loose["verdict"] == "approximately_equivalent"


def test_generate_test_data_shape():
    """Should generate correct shape."""
    data = generate_test_data(50, 5)
    assert data.shape == (50, 5)


def test_generate_test_data_reproducible():
    """Same seed should produce same data."""
    a = generate_test_data(10, 3, seed=42)
    b = generate_test_data(10, 3, seed=42)
    assert np.array_equal(a, b)


def test_generate_test_data_dtype():
    """Should be float32."""
    data = generate_test_data(5, 3)
    assert data.dtype == np.float32


def test_format_report_equivalent():
    """Should show PASS for equivalent."""
    result = {"verdict": "equivalent", "reason": "Exact match", "max_delta": 0.0,
              "mean_delta": 0.0, "n_samples": 100, "n_divergent": 0, "tolerance": 1e-5}
    report = format_equivalence_report(result)
    assert "PASS" in report


def test_format_report_divergent():
    """Should show FAIL for divergent."""
    result = {"verdict": "divergent", "reason": "Too different", "max_delta": 0.15,
              "mean_delta": 0.10, "n_samples": 100, "n_divergent": 50, "tolerance": 1e-5}
    report = format_equivalence_report(result)
    assert "FAIL" in report


def test_constants():
    """Tolerance constants should be ordered."""
    assert EXACT_TOLERANCE < FLOAT32_TOLERANCE
