"""Tests for post-training quantization (model_quantization.py). Phase 23.2."""

from __future__ import annotations
import pytest
from scripts.model_quantization import (
    compare_precision_levels, compute_quantization_plan, format_quantization_report, analyze_quantization,
)

def test_plan_fp16():
    plan = compute_quantization_plan("fp16", model_size_bytes=48_000_000, latency_ms=12.0)
    assert plan["size_factor"] == 0.5
    assert plan["estimated_size_bytes"] == 24_000_000
    assert plan["speedup"] > 1.0

def test_plan_int8_dynamic():
    plan = compute_quantization_plan("int8_dynamic", model_size_bytes=48_000_000)
    assert plan["size_reduction_pct"] == 75.0
    assert plan["method"] == "dynamic_quantization"

def test_plan_int8_static():
    plan = compute_quantization_plan("int8_static")
    assert plan.get("requires_calibration") is True

def test_plan_fp32():
    plan = compute_quantization_plan("fp32")
    assert plan["size_factor"] == 1.0
    assert plan["method"] == "none"

def test_compare_with_results():
    results = [
        {"precision": "fp32", "accuracy": 0.872},
        {"precision": "fp16", "accuracy": 0.872},
        {"precision": "int8_dynamic", "accuracy": 0.870},
    ]
    comparison = compare_precision_levels(results, model_size_bytes=48_000_000, latency_ms=12.0)
    assert "recommended" in comparison
    assert comparison["recommended"]["precision"] != "fp32"
    for r in comparison["sweep_results"]:
        assert "delta" in r

def test_compare_needs_qat():
    results = [
        {"precision": "fp32", "accuracy": 0.872},
        {"precision": "int8_dynamic", "accuracy": 0.850},  # > 1% drop
    ]
    comparison = compare_precision_levels(results)
    assert comparison["needs_qat"] is True

def test_compare_no_qat():
    results = [
        {"precision": "fp32", "accuracy": 0.872},
        {"precision": "int8_dynamic", "accuracy": 0.870},  # < 1% drop
    ]
    comparison = compare_precision_levels(results)
    assert comparison["needs_qat"] is False

def test_compare_plan_mode():
    comparison = compare_precision_levels(model_size_bytes=10_000_000)
    assert comparison["action"] == "plan"
    assert len(comparison["plans"]) == 4

def test_analyze_plan():
    report = analyze_quantization()
    assert report.get("action") == "plan" or "plans" in report

def test_format_results():
    report = {
        "experiment_id": "exp-042", "primary_metric": "accuracy",
        "sweep_results": [
            {"precision": "fp32", "accuracy": 0.872, "delta": 0.0, "speedup": 1.0, "size_reduction_pct": 0},
            {"precision": "int8_dynamic", "accuracy": 0.870, "delta": -0.002, "speedup": 2.6, "size_reduction_pct": 75},
        ],
        "recommended": {"precision": "int8_dynamic", "delta": -0.002, "speedup": 2.6},
        "needs_qat": False,
    }
    text = format_quantization_report(report)
    assert "Quantization" in text
    assert "int8" in text

def test_format_plan():
    report = {"action": "plan", "plans": [
        {"precision": "fp16", "description": "Half precision", "size_reduction_pct": 50, "speedup": 1.7},
    ]}
    text = format_quantization_report(report)
    assert "Plan" in text

def test_format_error():
    assert "ERROR" in format_quantization_report({"error": "fail"})
