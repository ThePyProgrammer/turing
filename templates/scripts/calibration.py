#!/usr/bin/env python3
"""Probability calibration for the autoresearch pipeline.

Measures whether model probabilities are well-calibrated, computes ECE/MCE,
generates reliability diagrams, and applies post-hoc calibration (Platt
scaling, isotonic regression, temperature scaling).

Usage:
    python scripts/calibration.py exp-042
    python scripts/calibration.py exp-042 --method platt
    python scripts/calibration.py exp-042 --method auto
    python scripts/calibration.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_N_BINS = 10
CALIBRATION_METHODS = ["platt", "isotonic", "temperature"]


# --- Calibration Metrics ---


def compute_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Compute Expected Calibration Error.

    ECE = sum(|bin_accuracy - bin_confidence| * bin_size / total)
    """
    if len(probabilities) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probabilities >= bin_boundaries[i]) & (probabilities < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (probabilities >= bin_boundaries[i]) & (probabilities <= bin_boundaries[i + 1])

        bin_size = np.sum(mask)
        if bin_size == 0:
            continue

        bin_accuracy = np.mean(labels[mask])
        bin_confidence = np.mean(probabilities[mask])
        ece += abs(bin_accuracy - bin_confidence) * bin_size / len(probabilities)

    return round(float(ece), 6)


def compute_mce(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> float:
    """Compute Maximum Calibration Error."""
    if len(probabilities) == 0:
        return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_gap = 0.0

    for i in range(n_bins):
        mask = (probabilities >= bin_boundaries[i]) & (probabilities < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (probabilities >= bin_boundaries[i]) & (probabilities <= bin_boundaries[i + 1])

        if np.sum(mask) == 0:
            continue

        bin_accuracy = np.mean(labels[mask])
        bin_confidence = np.mean(probabilities[mask])
        max_gap = max(max_gap, abs(bin_accuracy - bin_confidence))

    return round(float(max_gap), 6)


def compute_reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> list[dict]:
    """Compute reliability diagram data."""
    if len(probabilities) == 0:
        return []

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        mask = (probabilities >= lo) & (probabilities < hi)
        if i == n_bins - 1:
            mask = (probabilities >= lo) & (probabilities <= hi)

        bin_size = int(np.sum(mask))
        if bin_size == 0:
            bins.append({"bin": f"[{lo:.1f}-{hi:.1f}]", "predicted": None,
                        "actual": None, "gap": None, "n": 0})
            continue

        predicted = float(np.mean(probabilities[mask]))
        actual = float(np.mean(labels[mask]))
        gap = actual - predicted

        bins.append({
            "bin": f"[{lo:.1f}-{hi:.1f}]",
            "predicted": round(predicted, 4),
            "actual": round(actual, 4),
            "gap": round(gap, 4),
            "n": bin_size,
        })

    return bins


# --- Calibration Methods ---


def platt_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Apply Platt scaling (logistic regression on logits)."""
    from scipy.special import expit

    # Fit logistic regression: P(y=1|f) = sigmoid(a*f + b)
    # Simple gradient descent for a, b
    a, b = 1.0, 0.0
    lr = 0.01
    for _ in range(1000):
        pred = expit(a * logits + b)
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        grad_a = np.mean((pred - labels) * logits)
        grad_b = np.mean(pred - labels)
        a -= lr * grad_a
        b -= lr * grad_b

    calibrated = expit(a * logits + b)
    return {"method": "platt", "params": {"a": round(float(a), 6), "b": round(float(b), 6)},
            "calibrated_probabilities": calibrated}


def isotonic_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Apply isotonic regression calibration."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    calibrated = iso.fit_transform(probabilities, labels)
    return {"method": "isotonic", "params": {},
            "calibrated_probabilities": np.clip(calibrated, 0, 1)}


def temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Apply temperature scaling (single parameter T)."""
    from scipy.special import expit

    best_t = 1.0
    best_ece = float("inf")

    for t in np.arange(0.5, 5.0, 0.1):
        scaled = expit(logits / t)
        ece = compute_ece(scaled, labels)
        if ece < best_ece:
            best_ece = ece
            best_t = t

    calibrated = expit(logits / best_t)
    return {"method": "temperature", "params": {"T": round(float(best_t), 2)},
            "calibrated_probabilities": calibrated}


# --- Full Pipeline ---


def calibrate_model(
    probabilities: np.ndarray | None = None,
    logits: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    method: str = "auto",
    exp_id: str | None = None,
    config_path: str = "config.yaml",
) -> dict:
    """Run calibration analysis and optionally apply post-hoc calibration."""
    if (probabilities is None and logits is None) or labels is None:
        return {"error": "Provide probabilities (or logits) and labels for calibration"}

    if probabilities is None and logits is not None:
        from scipy.special import expit
        probabilities = expit(logits)

    # Before calibration
    ece_before = compute_ece(probabilities, labels)
    mce_before = compute_mce(probabilities, labels)
    reliability = compute_reliability_diagram(probabilities, labels)

    # Determine overconfidence
    overconfident_bins = [b for b in reliability if b.get("gap") is not None and b["gap"] < -0.05 and b["n"] > 0]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": exp_id,
        "before": {"ece": ece_before, "mce": mce_before},
        "reliability_diagram": reliability,
        "overconfident_bins": len(overconfident_bins),
    }

    # Apply calibration
    methods_to_try = CALIBRATION_METHODS if method == "auto" else [method]
    results = []

    for m in methods_to_try:
        try:
            if m == "platt" and logits is not None:
                cal = platt_scaling(logits, labels)
            elif m == "isotonic":
                cal = isotonic_calibration(probabilities, labels)
            elif m == "temperature" and logits is not None:
                cal = temperature_scaling(logits, labels)
            else:
                continue

            ece_after = compute_ece(cal["calibrated_probabilities"], labels)
            results.append({
                "method": m,
                "ece_after": ece_after,
                "improvement": round(ece_before - ece_after, 6),
                "params": cal.get("params", {}),
            })
        except Exception:
            continue

    # Find best method
    best = None
    if results:
        best = min(results, key=lambda r: r["ece_after"])

    report["calibration_results"] = results
    report["best_method"] = best

    # Verdict
    if ece_before < 0.02:
        report["verdict"] = "already_calibrated"
        report["reason"] = f"ECE {ece_before:.4f} is already low — calibration not needed"
    elif best and best["improvement"] > 0.01:
        report["verdict"] = "improved"
        report["reason"] = f"{best['method']} reduces ECE from {ece_before:.4f} to {best['ece_after']:.4f}"
    elif best:
        report["verdict"] = "marginal_improvement"
        report["reason"] = f"Best method ({best['method']}) improves ECE by only {best['improvement']:.4f}"
    else:
        report["verdict"] = "no_improvement"
        report["reason"] = "No calibration method improved ECE"

    return report


# --- Report Formatting ---


def save_calibration_report(report: dict, output_dir: str = "experiments/calibration") -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    exp_id = report.get("experiment_id", "unknown")
    filepath = out_path / f"{exp_id}-calibration.yaml"
    clean = json.loads(json.dumps(report, default=str))
    with open(filepath, "w") as f:
        yaml.dump(clean, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_calibration_report(report: dict) -> str:
    if "error" in report:
        return f"ERROR: {report['error']}"

    exp_id = report.get("experiment_id", "?")
    before = report.get("before", {})

    lines = [f"# Calibration: {exp_id}", "",
             f"*Generated {report.get('generated_at', 'N/A')[:19]}*", "",
             f"**ECE before:** {before.get('ece', '?')}",
             f"**MCE before:** {before.get('mce', '?')}", ""]

    # Reliability diagram
    diagram = report.get("reliability_diagram", [])
    if diagram:
        lines.extend(["## Reliability Diagram", "",
                      "| Bin | Predicted | Actual | Gap |",
                      "|-----|-----------|--------|-----|"])
        for b in diagram:
            if b["predicted"] is not None:
                gap_marker = " overconfident" if b["gap"] is not None and b["gap"] < -0.05 else ""
                lines.append(f"| {b['bin']} | {b['predicted']:.4f} | {b['actual']:.4f} | {b['gap']:+.4f}{gap_marker} |")
        lines.append("")

    # Calibration results
    results = report.get("calibration_results", [])
    if results:
        lines.extend(["## Calibration Methods", "",
                      "| Method | ECE After | Improvement |",
                      "|--------|-----------|-------------|"])
        best = report.get("best_method", {})
        for r in results:
            marker = " BEST" if r["method"] == best.get("method") else ""
            lines.append(f"| {r['method']} | {r['ece_after']:.4f} | {r['improvement']:+.4f} |{marker}")
        lines.append("")

    # Verdict
    verdict = report.get("verdict", "?")
    labels = {"already_calibrated": "ALREADY CALIBRATED", "improved": "IMPROVED",
              "marginal_improvement": "MARGINAL IMPROVEMENT", "no_improvement": "NO IMPROVEMENT"}
    lines.extend(["## Verdict", "", f"**{labels.get(verdict, verdict.upper())}**", "",
                  report.get("reason", "")])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probability calibration")
    parser.add_argument("exp_id", nargs="?", help="Experiment ID")
    parser.add_argument("--method", choices=CALIBRATION_METHODS + ["auto"], default="auto")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Without data, show usage
    report = calibrate_model(exp_id=args.exp_id, method=args.method, config_path=args.config)

    if "error" not in report:
        filepath = save_calibration_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_calibration_report(report))


if __name__ == "__main__":
    main()
