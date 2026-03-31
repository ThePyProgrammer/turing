#!/usr/bin/env python3
"""Inference latency benchmarking for model exports.

Measures p50/p95/p99 inference latency with warm-up phase.
Compares original vs exported model latency.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


DEFAULT_WARMUP = 10
DEFAULT_ITERATIONS = 100


def benchmark_inference(
    predict_fn,
    test_input,
    n_warmup: int = DEFAULT_WARMUP,
    n_iterations: int = DEFAULT_ITERATIONS,
) -> dict:
    """Benchmark inference latency of a prediction function.

    Args:
        predict_fn: Callable that takes input and returns predictions.
        test_input: Input data for prediction.
        n_warmup: Number of warm-up calls (discarded).
        n_iterations: Number of benchmark calls.

    Returns:
        Dict with p50, p95, p99 latency in milliseconds and raw timings.
    """
    # Warm-up phase
    for _ in range(n_warmup):
        try:
            predict_fn(test_input)
        except Exception:
            pass

    # Benchmark phase
    timings_ms = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        try:
            predict_fn(test_input)
        except Exception as e:
            return {"error": f"Prediction failed during benchmark: {e}"}
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed_ms)

    arr = np.array(timings_ms)
    return {
        "n_iterations": n_iterations,
        "n_warmup": n_warmup,
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def compare_latency(
    original_benchmark: dict,
    exported_benchmark: dict,
) -> dict:
    """Compare latency between original and exported model.

    Returns comparison dict with speedup ratios and verdict.
    """
    if "error" in original_benchmark or "error" in exported_benchmark:
        return {
            "verdict": "error",
            "reason": original_benchmark.get("error") or exported_benchmark.get("error"),
        }

    orig_p50 = original_benchmark["p50_ms"]
    exported_p50 = exported_benchmark["p50_ms"]

    if exported_p50 > 0:
        speedup = orig_p50 / exported_p50
    else:
        speedup = float("inf")

    if speedup > 1.1:
        verdict = "faster"
        description = f"Exported model is {speedup:.1f}x faster (p50: {orig_p50:.2f}ms -> {exported_p50:.2f}ms)"
    elif speedup < 0.9:
        verdict = "slower"
        description = f"Exported model is {1/speedup:.1f}x slower (p50: {orig_p50:.2f}ms -> {exported_p50:.2f}ms)"
    else:
        verdict = "similar"
        description = f"Similar latency (p50: {orig_p50:.2f}ms vs {exported_p50:.2f}ms)"

    return {
        "verdict": verdict,
        "description": description,
        "speedup_ratio": round(speedup, 2),
        "original_p50_ms": orig_p50,
        "exported_p50_ms": exported_p50,
        "original_p95_ms": original_benchmark["p95_ms"],
        "exported_p95_ms": exported_benchmark["p95_ms"],
        "original_p99_ms": original_benchmark["p99_ms"],
        "exported_p99_ms": exported_benchmark["p99_ms"],
    }


def compute_percentiles(timings_ms: list[float]) -> dict:
    """Compute percentile statistics from raw timings."""
    if not timings_ms:
        return {}
    arr = np.array(timings_ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def format_benchmark_report(
    original: dict | None,
    exported: dict | None,
    comparison: dict | None = None,
) -> str:
    """Format benchmark results as readable text."""
    lines = ["## Latency Benchmark", ""]

    if exported:
        lines.extend([
            "### Exported Model",
            "",
            f"- **p50:** {exported['p50_ms']:.2f} ms",
            f"- **p95:** {exported['p95_ms']:.2f} ms",
            f"- **p99:** {exported['p99_ms']:.2f} ms",
            f"- **mean:** {exported['mean_ms']:.2f} ms (std: {exported['std_ms']:.2f})",
            f"- **iterations:** {exported.get('n_iterations', 'N/A')}",
        ])

    if original:
        lines.extend([
            "",
            "### Original Model",
            "",
            f"- **p50:** {original['p50_ms']:.2f} ms",
            f"- **p95:** {original['p95_ms']:.2f} ms",
            f"- **p99:** {original['p99_ms']:.2f} ms",
        ])

    if comparison and comparison.get("verdict") != "error":
        lines.extend([
            "",
            "### Comparison",
            "",
            f"**{comparison['verdict'].upper()}** — {comparison['description']}",
        ])

    return "\n".join(lines)
