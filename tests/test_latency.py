"""Tests for latency_benchmark.py."""

from __future__ import annotations

import time
import numpy as np
import pytest

from scripts.latency_benchmark import (
    benchmark_inference,
    compare_latency,
    compute_percentiles,
    format_benchmark_report,
)


def _fast_predict(x):
    """A fast predict function for testing."""
    return x * 2


def _slow_predict(x):
    """A slow predict function for testing."""
    time.sleep(0.001)
    return x * 2


# --- benchmark_inference ---


def test_benchmark_basic():
    """Should return timing statistics."""
    result = benchmark_inference(_fast_predict, np.array([1.0]), n_warmup=2, n_iterations=10)
    assert "p50_ms" in result
    assert "p95_ms" in result
    assert "p99_ms" in result
    assert result["n_iterations"] == 10
    assert result["p50_ms"] >= 0


def test_benchmark_with_warmup():
    """Warm-up should not affect results."""
    result = benchmark_inference(_fast_predict, np.array([1.0]), n_warmup=5, n_iterations=20)
    assert result["n_warmup"] == 5
    assert result["n_iterations"] == 20


def test_benchmark_slow_function():
    """Should measure actual latency."""
    result = benchmark_inference(_slow_predict, np.array([1.0]), n_warmup=1, n_iterations=5)
    assert result["p50_ms"] > 0.5  # At least 0.5ms for a 1ms sleep


def test_benchmark_error_handling():
    """Should report error for failing function."""
    def failing_fn(x):
        raise RuntimeError("Model crashed")
    result = benchmark_inference(failing_fn, np.array([1.0]), n_warmup=0, n_iterations=1)
    assert "error" in result


# --- compare_latency ---


def test_compare_faster():
    """Should detect when exported is faster."""
    orig = {"p50_ms": 10.0, "p95_ms": 15.0, "p99_ms": 20.0}
    exported = {"p50_ms": 2.0, "p95_ms": 3.0, "p99_ms": 4.0}
    result = compare_latency(orig, exported)
    assert result["verdict"] == "faster"
    assert result["speedup_ratio"] > 1


def test_compare_slower():
    """Should detect when exported is slower."""
    orig = {"p50_ms": 2.0, "p95_ms": 3.0, "p99_ms": 4.0}
    exported = {"p50_ms": 10.0, "p95_ms": 15.0, "p99_ms": 20.0}
    result = compare_latency(orig, exported)
    assert result["verdict"] == "slower"


def test_compare_similar():
    """Should detect similar latency."""
    orig = {"p50_ms": 5.0, "p95_ms": 6.0, "p99_ms": 7.0}
    exported = {"p50_ms": 5.1, "p95_ms": 6.1, "p99_ms": 7.1}
    result = compare_latency(orig, exported)
    assert result["verdict"] == "similar"


def test_compare_with_error():
    """Should propagate error."""
    orig = {"error": "Failed"}
    exported = {"p50_ms": 5.0, "p95_ms": 6.0, "p99_ms": 7.0}
    result = compare_latency(orig, exported)
    assert result["verdict"] == "error"


def test_compare_zero_exported():
    """Should handle zero exported latency."""
    orig = {"p50_ms": 5.0, "p95_ms": 6.0, "p99_ms": 7.0}
    exported = {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    result = compare_latency(orig, exported)
    assert result["speedup_ratio"] == float("inf") or result["verdict"] == "faster"


# --- compute_percentiles ---


def test_percentiles_basic():
    """Should compute correct percentiles."""
    timings = list(range(100))
    result = compute_percentiles(timings)
    assert result["p50_ms"] == pytest.approx(49.5, abs=1)
    assert result["min_ms"] == 0.0
    assert result["max_ms"] == 99.0


def test_percentiles_empty():
    """Should handle empty list."""
    assert compute_percentiles([]) == {}


def test_percentiles_single():
    """Should handle single value."""
    result = compute_percentiles([5.0])
    assert result["p50_ms"] == 5.0
    assert result["mean_ms"] == 5.0


# --- format_benchmark_report ---


def test_format_report_full():
    """Should format comparison report."""
    orig = {"p50_ms": 10.0, "p95_ms": 15.0, "p99_ms": 20.0, "mean_ms": 11.0, "std_ms": 2.0, "n_iterations": 100}
    exported = {"p50_ms": 2.0, "p95_ms": 3.0, "p99_ms": 4.0, "mean_ms": 2.5, "std_ms": 0.5, "n_iterations": 100}
    comparison = {"verdict": "faster", "description": "5x faster"}
    report = format_benchmark_report(orig, exported, comparison)
    assert "Latency" in report
    assert "faster" in report.lower()


def test_format_report_exported_only():
    """Should format with just exported data."""
    exported = {"p50_ms": 2.0, "p95_ms": 3.0, "p99_ms": 4.0, "mean_ms": 2.5, "std_ms": 0.5, "n_iterations": 100}
    report = format_benchmark_report(None, exported, None)
    assert "2.00" in report
