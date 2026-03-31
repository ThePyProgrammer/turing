"""Edge case tests for model export pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.equivalence_checker import compare_outputs, generate_test_data
from scripts.export_card import generate_export_card, format_export_card
from scripts.export_formats import (
    detect_model_type,
    export_joblib,
    export_model,
    get_supported_formats,
)
from scripts.latency_benchmark import benchmark_inference, compare_latency, compute_percentiles


# --- equivalence edge cases ---


def test_compare_all_zeros():
    """Should handle all-zero outputs."""
    result = compare_outputs([0, 0, 0], [0, 0, 0])
    assert result["verdict"] == "equivalent"


def test_compare_negative_values():
    """Should handle negative values."""
    a = [-0.5, -1.0, -0.3]
    b = [-0.5, -1.0, -0.3]
    result = compare_outputs(a, b)
    assert result["verdict"] == "equivalent"


def test_compare_large_arrays():
    """Should handle large arrays efficiently."""
    n = 10000
    a = np.random.randn(n)
    result = compare_outputs(a, a)
    assert result["verdict"] == "equivalent"
    assert result["n_samples"] == n


def test_compare_quantized_tolerance():
    """Should support quantized tolerance (1e-3)."""
    a = [0.850]
    b = [0.8505]  # Within 1e-3
    result = compare_outputs(a, b, tolerance=1e-3)
    assert result["verdict"] in ("equivalent", "approximately_equivalent")


def test_generate_different_seeds():
    """Different seeds should produce different data."""
    a = generate_test_data(10, 5, seed=42)
    b = generate_test_data(10, 5, seed=123)
    assert not np.array_equal(a, b)


# --- latency edge cases ---


def test_benchmark_single_iteration():
    """Should work with single iteration."""
    result = benchmark_inference(lambda x: x, np.array([1.0]), n_warmup=0, n_iterations=1)
    assert result["n_iterations"] == 1
    assert result["p50_ms"] >= 0


def test_compare_identical_latency():
    """Should report similar for identical timings."""
    bench = {"p50_ms": 5.0, "p95_ms": 6.0, "p99_ms": 7.0}
    result = compare_latency(bench, bench)
    assert result["verdict"] == "similar"
    assert result["speedup_ratio"] == 1.0


def test_percentiles_sorted_input():
    """Should compute correct percentiles for sorted data."""
    timings = [float(i) for i in range(1, 101)]
    result = compute_percentiles(timings)
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 100.0
    assert result["p50_ms"] == pytest.approx(50.5, abs=1)


# --- export_formats edge cases ---


def test_export_joblib_large_file(tmp_path):
    """Should handle large model files."""
    model_file = tmp_path / "large_model.joblib"
    model_file.write_bytes(b"x" * (10 * 1024 * 1024))  # 10 MB
    output_dir = tmp_path / "output"

    result = export_joblib(str(model_file), str(output_dir), "large-model")
    assert result["size_mb"] == pytest.approx(10.0, abs=0.5)


def test_export_joblib_creates_nested_dir(tmp_path):
    """Should create nested output directory."""
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"data")
    output_dir = tmp_path / "deep" / "nested" / "dir"

    result = export_joblib(str(model_file), str(output_dir), "test")
    assert "error" not in result


def test_detect_model_type_fallback():
    """Should use model_type key as fallback."""
    assert detect_model_type({"model_type": "catboost"}) == "catboost"


def test_get_supported_catboost():
    """CatBoost should support joblib and onnx."""
    formats = get_supported_formats("catboost")
    assert "joblib" in formats


# --- export_card edge cases ---


def test_card_minimal():
    """Should generate card with minimal data."""
    experiment = {"experiment_id": "exp-001", "metrics": {}, "config": {}}
    export_result = {"format": "joblib", "dependencies": []}
    card = generate_export_card(experiment, export_result)
    assert card["experiment_id"] == "exp-001"


def test_card_without_config():
    """Should handle missing config."""
    experiment = {"experiment_id": "exp-001", "metrics": {"accuracy": 0.85}, "config": {}}
    export_result = {"format": "joblib", "dependencies": []}
    card = generate_export_card(experiment, export_result, config=None)
    assert card["experiment_id"] == "exp-001"


def test_format_card_with_latency():
    """Should include latency section."""
    card = {
        "name": "test",
        "experiment_id": "exp-001",
        "task": "test",
        "model_type": "xgboost",
        "export_format": "joblib",
        "size_mb": 1.0,
        "dependencies": [],
        "metrics": {},
        "export_date": "2026-04-01",
        "inference_latency": {
            "exported_p50_ms": 1.2,
            "exported_p95_ms": 2.5,
            "original_p50_ms": 5.0,
            "speedup": 4.2,
        },
    }
    report = format_export_card(card)
    assert "Latency" in report
    assert "4.2" in report
