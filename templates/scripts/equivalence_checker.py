#!/usr/bin/env python3
"""Inference equivalence verification for model exports.

Compares outputs between the original model and the exported model
to verify they produce identical (or near-identical) results.

Verdicts:
    equivalent                 — max delta < 1e-6 (float precision)
    approximately_equivalent   — max delta < tolerance (default 1e-5)
    divergent                  — max delta >= tolerance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


FLOAT32_TOLERANCE = 1e-5
QUANTIZED_TOLERANCE = 1e-3
EXACT_TOLERANCE = 1e-6


def compare_outputs(
    original_outputs: list | np.ndarray,
    exported_outputs: list | np.ndarray,
    tolerance: float = FLOAT32_TOLERANCE,
) -> dict:
    """Compare outputs from original and exported models.

    Args:
        original_outputs: Predictions from original model.
        exported_outputs: Predictions from exported model.
        tolerance: Maximum allowed difference.

    Returns:
        Dict with verdict, max_delta, mean_delta, and per-sample details.
    """
    orig = np.array(original_outputs, dtype=np.float64).flatten()
    exported = np.array(exported_outputs, dtype=np.float64).flatten()

    if orig.shape != exported.shape:
        return {
            "verdict": "divergent",
            "reason": f"Shape mismatch: original {orig.shape} vs exported {exported.shape}",
            "max_delta": float("inf"),
            "mean_delta": float("inf"),
            "n_samples": 0,
            "n_divergent": 0,
        }

    deltas = np.abs(orig - exported)
    max_delta = float(np.max(deltas))
    mean_delta = float(np.mean(deltas))
    n_divergent = int(np.sum(deltas >= tolerance))

    if max_delta < EXACT_TOLERANCE:
        verdict = "equivalent"
        reason = f"Exact match (max delta {max_delta:.2e} < {EXACT_TOLERANCE})"
    elif max_delta < tolerance:
        verdict = "approximately_equivalent"
        reason = f"Within tolerance (max delta {max_delta:.2e} < {tolerance})"
    else:
        verdict = "divergent"
        reason = (
            f"Max delta {max_delta:.2e} exceeds tolerance {tolerance}. "
            f"{n_divergent} of {len(orig)} samples diverge."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "max_delta": round(max_delta, 10),
        "mean_delta": round(mean_delta, 10),
        "n_samples": len(orig),
        "n_divergent": n_divergent,
        "tolerance": tolerance,
    }


def run_equivalence_check(
    original_predict_fn,
    exported_predict_fn,
    test_data: np.ndarray | list,
    tolerance: float = FLOAT32_TOLERANCE,
) -> dict:
    """Run equivalence check by predicting on test data with both models.

    Args:
        original_predict_fn: Callable that takes input data and returns predictions.
        exported_predict_fn: Callable for the exported model.
        test_data: Input data to predict on.
        tolerance: Maximum allowed difference.

    Returns:
        Equivalence check result dict.
    """
    try:
        original_preds = original_predict_fn(test_data)
    except Exception as e:
        return {"verdict": "error", "reason": f"Original model prediction failed: {e}"}

    try:
        exported_preds = exported_predict_fn(test_data)
    except Exception as e:
        return {"verdict": "error", "reason": f"Exported model prediction failed: {e}"}

    return compare_outputs(original_preds, exported_preds, tolerance)


def generate_test_data(n_samples: int = 100, n_features: int = 10, seed: int = 42) -> np.ndarray:
    """Generate random test data for equivalence checking.

    Args:
        n_samples: Number of test samples.
        n_features: Number of features per sample.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, n_features).
    """
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, n_features).astype(np.float32)


def format_equivalence_report(result: dict) -> str:
    """Format equivalence check result as readable text."""
    verdict = result.get("verdict", "unknown")
    reason = result.get("reason", "")

    verdict_markers = {
        "equivalent": "PASS (exact)",
        "approximately_equivalent": "PASS (approx)",
        "divergent": "FAIL",
        "error": "ERROR",
    }
    marker = verdict_markers.get(verdict, verdict)

    lines = [
        f"## Equivalence Check: {marker}",
        "",
        f"*{reason}*",
        "",
    ]

    if result.get("n_samples"):
        lines.extend([
            f"- **Samples tested:** {result['n_samples']}",
            f"- **Max delta:** {result['max_delta']:.2e}",
            f"- **Mean delta:** {result['mean_delta']:.2e}",
            f"- **Divergent samples:** {result['n_divergent']}",
            f"- **Tolerance:** {result['tolerance']:.0e}",
        ])

    return "\n".join(lines)
