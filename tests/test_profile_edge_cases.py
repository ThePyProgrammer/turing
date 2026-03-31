"""Edge case tests for profile_training.py."""

from __future__ import annotations

import pytest

from scripts.profile_training import (
    compute_throughput,
    detect_bottleneck,
    estimate_timing_breakdown,
    generate_recommendations,
    save_profile,
)


def test_breakdown_string_train_seconds():
    """Should handle train_seconds as string in metrics."""
    profile = {"total_time_sec": 100, "train_time_sec": 80, "overhead_sec": 20, "metrics": {"train_seconds": "80.5"}}
    breakdown = estimate_timing_breakdown(profile)
    assert breakdown["training_sec"] == pytest.approx(80.5, abs=0.1)


def test_detect_multiple_bottlenecks():
    """Should return highest-severity bottleneck."""
    profile = {
        "total_time_sec": 400,
        "overhead_sec": 300,
        "memory": {"peak_rss_mb": 10000},
        "metrics": {"train_seconds": 100},
    }
    bn = detect_bottleneck(profile, {})
    # Both data_loading and memory are high, should return one
    assert bn["severity"] == "high"


def test_detect_zero_total_time():
    """Should handle zero total time."""
    profile = {"total_time_sec": 0, "overhead_sec": 0, "memory": {}, "metrics": {}}
    bn = detect_bottleneck(profile, {})
    assert bn["type"] == "none_detected"


def test_recommendations_with_high_memory():
    """Should add memory warning for non-memory bottleneck."""
    bn = {"type": "data_loading"}
    recs = generate_recommendations(bn, {"total_time_sec": 100, "memory": {"peak_rss_mb": 5000}})
    assert any("memory" in r.lower() or "MB" in r for r in recs)


def test_throughput_zero_time():
    """Should handle zero training time."""
    profile = {"train_time_sec": 0, "total_time_sec": 0, "metrics": {"n_samples": 100}}
    tp = compute_throughput(profile)
    assert "samples_per_sec" not in tp


def test_throughput_train_samples_key():
    """Should use n_train_samples as fallback."""
    profile = {"train_time_sec": 10, "total_time_sec": 10, "metrics": {"n_train_samples": 200}}
    tp = compute_throughput(profile)
    assert tp["samples_per_sec"] == 20.0


def test_save_creates_nested_dir(tmp_path):
    """Should create nested output directory."""
    data = {"experiment_id": "exp-001", "profile": {}}
    filepath = save_profile(data, str(tmp_path / "deep" / "nested"))
    assert filepath.exists()
