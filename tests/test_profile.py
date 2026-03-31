"""Tests for computational profiling (profile_training.py).

Phase 12.1: Verifies bottleneck detection, recommendation generation,
timing breakdown, throughput computation, and report formatting.
"""

from __future__ import annotations

import pytest

from scripts.profile_training import (
    BOTTLENECK_RECOMMENDATIONS,
    compute_throughput,
    detect_bottleneck,
    estimate_timing_breakdown,
    format_profile_report,
    generate_recommendations,
    save_profile,
)


# --- estimate_timing_breakdown ---


def test_breakdown_with_train_seconds():
    """Should compute breakdown when train_seconds available."""
    profile = {
        "total_time_sec": 100,
        "train_time_sec": 80,
        "overhead_sec": 20,
        "metrics": {"train_seconds": 80},
    }
    breakdown = estimate_timing_breakdown(profile)
    assert breakdown["total_sec"] == 100
    assert breakdown["overhead_sec"] == 20
    assert breakdown["training_sec"] == 80
    assert breakdown["training_pct"] == 80.0


def test_breakdown_no_train_seconds():
    """Should handle missing train_seconds."""
    profile = {"total_time_sec": 50, "train_time_sec": 50, "overhead_sec": 0, "metrics": {}}
    breakdown = estimate_timing_breakdown(profile)
    assert breakdown["total_sec"] == 50
    assert breakdown["overhead_pct"] == 0


def test_breakdown_zero_total():
    """Should handle zero total time."""
    profile = {"total_time_sec": 0, "train_time_sec": 0, "overhead_sec": 0, "metrics": {}}
    breakdown = estimate_timing_breakdown(profile)
    assert breakdown == {}


# --- detect_bottleneck ---


def test_detect_data_loading_bottleneck():
    """Should detect when overhead dominates."""
    profile = {
        "total_time_sec": 100,
        "overhead_sec": 70,
        "memory": {},
        "metrics": {},
    }
    bottleneck = detect_bottleneck(profile, {})
    assert bottleneck["type"] == "data_loading"
    assert bottleneck["severity"] == "high"


def test_detect_memory_bottleneck():
    """Should detect high memory usage."""
    profile = {
        "total_time_sec": 50,
        "overhead_sec": 5,
        "memory": {"peak_rss_mb": 10000},
        "metrics": {},
    }
    bottleneck = detect_bottleneck(profile, {})
    assert bottleneck["type"] == "memory"


def test_detect_no_bottleneck():
    """Should return none_detected when everything is fine."""
    profile = {
        "total_time_sec": 30,
        "overhead_sec": 5,
        "memory": {"peak_rss_mb": 512},
        "metrics": {},
    }
    bottleneck = detect_bottleneck(profile, {})
    assert bottleneck["type"] == "none_detected"


def test_detect_long_training():
    """Should flag very long training time."""
    profile = {
        "total_time_sec": 400,
        "overhead_sec": 10,
        "memory": {},
        "metrics": {"train_seconds": 390},
    }
    bottleneck = detect_bottleneck(profile, {})
    assert bottleneck["type"] == "model_training"


def test_detect_gpu_underutilized():
    """Should flag low GPU memory utilization."""
    profile = {
        "total_time_sec": 50,
        "overhead_sec": 5,
        "memory": {"peak_gpu_mb": 200},
        "gpu": {"type": "cuda", "device": "GTX 1080", "memory_total_mb": 8192},
        "metrics": {},
    }
    bottleneck = detect_bottleneck(profile, {})
    assert bottleneck["type"] == "gpu_underutilized"


# --- generate_recommendations ---


def test_recommendations_for_data_loading():
    """Should generate data loading recommendations."""
    bottleneck = {"type": "data_loading", "severity": "high"}
    recs = generate_recommendations(bottleneck, {"total_time_sec": 100, "memory": {}})
    assert len(recs) > 0
    assert any("cache" in r.lower() or "data" in r.lower() for r in recs)


def test_recommendations_for_memory():
    """Should generate memory recommendations."""
    bottleneck = {"type": "memory", "severity": "high"}
    recs = generate_recommendations(bottleneck, {"total_time_sec": 100, "memory": {"peak_rss_mb": 5000}})
    assert len(recs) > 0


def test_recommendations_no_gpu():
    """Should suggest GPU when not available."""
    bottleneck = {"type": "model_training", "severity": "medium"}
    recs = generate_recommendations(bottleneck, {"total_time_sec": 100, "memory": {}})
    assert any("gpu" in r.lower() for r in recs)


def test_recommendations_fast_training():
    """Should note when training is very fast."""
    bottleneck = {"type": "none_detected"}
    recs = generate_recommendations(bottleneck, {"total_time_sec": 3, "memory": {}})
    assert any("fast" in r.lower() for r in recs)


def test_all_bottleneck_types_have_recommendations():
    """Every bottleneck type in the mapping should have recommendations."""
    for bt_type, recs in BOTTLENECK_RECOMMENDATIONS.items():
        assert len(recs) > 0, f"No recommendations for {bt_type}"


# --- compute_throughput ---


def test_throughput_basic():
    """Should compute samples/sec."""
    profile = {
        "train_time_sec": 10,
        "total_time_sec": 10,
        "metrics": {"n_samples": 1000},
    }
    tp = compute_throughput(profile)
    assert tp["samples_per_sec"] == 100.0


def test_throughput_no_samples():
    """Should return empty when no sample count."""
    profile = {"train_time_sec": 10, "total_time_sec": 10, "metrics": {}}
    tp = compute_throughput(profile)
    assert "samples_per_sec" not in tp


def test_throughput_string_samples():
    """Should parse string sample count."""
    profile = {"train_time_sec": 10, "total_time_sec": 10, "metrics": {"n_samples": "500"}}
    tp = compute_throughput(profile)
    assert tp["samples_per_sec"] == 50.0


# --- format_profile_report ---


def test_format_report_basic():
    """Should produce readable report."""
    data = {
        "experiment_id": "exp-042",
        "profile": {
            "total_time_sec": 142.3,
            "memory": {"peak_rss_mb": 2048, "python_peak_mb": 512},
        },
        "breakdown": {"total_sec": 142.3, "overhead_sec": 85, "overhead_pct": 59.7},
        "bottleneck": {
            "type": "data_loading",
            "severity": "high",
            "description": "Data loading is 60% of total time",
        },
        "recommendations": ["Cache preprocessed data"],
        "throughput": {"samples_per_sec": 1420},
    }
    report = format_profile_report(data)
    assert "exp-042" in report
    assert "142.3" in report
    assert "data_loading" in report
    assert "Cache" in report


def test_format_report_error():
    """Should handle error case."""
    data = {"error": "No experiment found"}
    report = format_profile_report(data)
    assert "ERROR" in report


def test_format_report_with_gpu():
    """Should include GPU section when available."""
    data = {
        "experiment_id": "exp-001",
        "profile": {
            "total_time_sec": 50,
            "memory": {"peak_rss_mb": 1024, "python_peak_mb": 256, "peak_gpu_mb": 4000},
            "gpu": {"type": "cuda", "device": "RTX 3090", "memory_total_mb": 24576},
        },
        "breakdown": {},
        "bottleneck": {"type": "none_detected", "severity": "low", "description": "OK"},
        "recommendations": [],
        "throughput": {},
    }
    report = format_profile_report(data)
    assert "RTX 3090" in report
    assert "GPU" in report


# --- save_profile ---


def test_save_profile(tmp_path):
    """Should save YAML file."""
    data = {"experiment_id": "exp-042", "profile": {}}
    filepath = save_profile(data, str(tmp_path))
    assert filepath.exists()
    assert "exp-042" in filepath.name
