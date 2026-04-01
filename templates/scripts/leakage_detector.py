#!/usr/bin/env python3
"""Targeted leakage detection for the autoresearch pipeline.

Probes for data leakage by training on single features, checking
feature-target correlations, detecting train/test overlap, and flagging
suspiciously predictive features.

Usage:
    python scripts/leakage_detector.py
    python scripts/leakage_detector.py --deep
    python scripts/leakage_detector.py --features "feature_1,feature_2"
    python scripts/leakage_detector.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from scripts.turing_io import load_config

DEFAULT_CORRELATION_THRESHOLD = 0.95
DEFAULT_SINGLE_FEATURE_RATIO = 0.80  # Flag if single feature achieves >80% of full model


# --- Leakage Checks ---


def check_feature_target_correlation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> list[dict]:
    """Check for features with very high correlation to the target.

    Returns list of flagged features.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    flags = []
    for i in range(n_features):
        feature = X[:, i].astype(float)
        target = y.astype(float)

        # Skip non-numeric
        if np.any(np.isnan(feature)) or np.any(np.isnan(target)):
            continue

        if np.std(feature) == 0 or np.std(target) == 0:
            continue

        corr = abs(float(np.corrcoef(feature, target)[0, 1]))

        if corr > threshold:
            flags.append({
                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "correlation": round(corr, 4),
                "severity": "critical" if corr > 0.99 else "high",
                "reason": f"Correlation {corr:.4f} with target — likely derived from target",
            })

    return flags


def check_single_feature_predictiveness(
    X: np.ndarray,
    y: np.ndarray,
    full_model_score: float,
    feature_names: list[str] | None = None,
    ratio_threshold: float = DEFAULT_SINGLE_FEATURE_RATIO,
    task_type: str = "classification",
) -> list[dict]:
    """Train a simple model on each feature individually.

    Flags features where single-feature accuracy > ratio_threshold * full_model_score.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    flags = []
    threshold_value = ratio_threshold * full_model_score

    for i in range(n_features):
        feature = X[:, i].reshape(-1, 1)

        # Simple threshold-based classifier / linear regression
        score = _simple_single_feature_score(feature, y, task_type)

        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"

        if score > threshold_value:
            flags.append({
                "feature": name,
                "single_feature_score": round(score, 4),
                "full_model_score": round(full_model_score, 4),
                "ratio": round(score / full_model_score, 4) if full_model_score > 0 else None,
                "severity": "critical" if score > full_model_score else "high",
                "reason": (
                    f"Single feature achieves {score:.4f} "
                    f"({'more than' if score > full_model_score else f'{score/full_model_score:.0%} of'} "
                    f"full model {full_model_score:.4f}) — investigate leakage"
                ),
            })

    return flags


def _simple_single_feature_score(
    feature: np.ndarray,
    y: np.ndarray,
    task_type: str,
) -> float:
    """Quick score for a single feature using threshold-based prediction."""
    feature = feature.ravel()

    if task_type == "classification":
        # Find best threshold (accuracy maximizing)
        thresholds = np.percentile(feature, [25, 50, 75])
        best_acc = 0.0
        classes = np.unique(y)
        if len(classes) <= 1:
            return 0.0

        for t in thresholds:
            pred = (feature > t).astype(int)
            # Map to actual classes
            if len(classes) == 2:
                pred_mapped = np.where(pred, classes[1], classes[0])
            else:
                pred_mapped = pred
            acc = float(np.mean(pred_mapped == y))
            best_acc = max(best_acc, acc, 1 - acc)  # Try both directions

        return best_acc
    else:
        # Correlation-based R² approximation
        if np.std(feature) == 0:
            return 0.0
        corr = np.corrcoef(feature, y.astype(float))[0, 1]
        return float(corr ** 2)  # R²


def check_train_test_overlap(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> dict:
    """Check for identical or near-identical samples across splits.

    Uses hash-based deduplication.
    """
    def hash_row(row):
        return hashlib.md5(row.tobytes()).hexdigest()

    train_hashes = set()
    for row in X_train:
        train_hashes.add(hash_row(np.asarray(row)))

    overlapping = 0
    overlap_indices = []
    for idx, row in enumerate(X_test):
        h = hash_row(np.asarray(row))
        if h in train_hashes:
            overlapping += 1
            if len(overlap_indices) < 10:
                overlap_indices.append(idx)

    n_test = len(X_test)
    overlap_pct = overlapping / n_test if n_test > 0 else 0

    if overlapping == 0:
        status = "pass"
        reason = "No overlapping samples between train and test"
    elif overlap_pct < 0.01:
        status = "warn"
        reason = f"{overlapping} overlapping samples ({overlap_pct:.2%}) — minor but investigate"
    else:
        status = "fail"
        reason = f"{overlapping} overlapping samples ({overlap_pct:.2%}) — significant leakage"

    return {
        "check": "train_test_overlap",
        "status": status,
        "overlapping_samples": overlapping,
        "overlap_percentage": round(overlap_pct, 4),
        "test_size": n_test,
        "reason": reason,
        "severity": "critical",
        "sample_overlap_indices": overlap_indices,
    }


# --- Full Leakage Scan ---


def run_leakage_scan(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    X_train: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    full_model_score: float | None = None,
    feature_names: list[str] | None = None,
    deep: bool = False,
    task_type: str = "classification",
    config_path: str = "config.yaml",
) -> dict:
    """Run a complete leakage scan.

    Args:
        X: Feature matrix (for correlation and single-feature tests).
        y: Target array.
        X_train: Training features (for overlap check).
        X_test: Test features (for overlap check).
        full_model_score: Best model's primary metric (for single-feature comparison).
        feature_names: Names of features.
        deep: Run full single-feature analysis (slower).
        task_type: classification or regression.
        config_path: Path to config.yaml.

    Returns:
        Complete leakage report.
    """
    config = load_config(config_path)
    if not task_type:
        task_type = config.get("task", {}).get("type", "classification")

    checks = []

    # Feature-target correlation
    if X is not None and y is not None:
        corr_flags = check_feature_target_correlation(X, y, feature_names)
        checks.append({
            "check": "feature_target_correlation",
            "status": "fail" if corr_flags else "pass",
            "flags": corr_flags,
            "n_flagged": len(corr_flags),
            "severity": "critical",
            "reason": f"{len(corr_flags)} feature(s) with >{DEFAULT_CORRELATION_THRESHOLD} target correlation" if corr_flags else "No suspicious correlations",
        })

    # Single-feature predictiveness (deep mode)
    if deep and X is not None and y is not None and full_model_score is not None:
        sf_flags = check_single_feature_predictiveness(
            X, y, full_model_score, feature_names, task_type=task_type,
        )
        checks.append({
            "check": "single_feature_predictiveness",
            "status": "fail" if sf_flags else "pass",
            "flags": sf_flags,
            "n_flagged": len(sf_flags),
            "severity": "critical",
            "reason": f"{len(sf_flags)} feature(s) suspiciously predictive alone" if sf_flags else "No single feature is suspiciously predictive",
        })

    # Train/test overlap
    if X_train is not None and X_test is not None:
        overlap = check_train_test_overlap(X_train, X_test)
        checks.append(overlap)

    if not checks:
        return {
            "error": "No data provided for leakage scan. Provide X, y, or X_train/X_test arrays.",
            "note": "Run with --data train.npz --test test.npz to scan for leakage.",
        }

    # Compute verdict
    n_fail = sum(1 for c in checks if c.get("status") == "fail")
    n_warn = sum(1 for c in checks if c.get("status") == "warn")

    if n_fail > 0:
        verdict = "leakage_detected"
    elif n_warn > 0:
        verdict = "suspicious"
    else:
        verdict = "clean"

    return {
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type,
        "deep_mode": deep,
        "checks": checks,
        "verdict": verdict,
        "n_checks": len(checks),
    }


# --- Report Formatting ---


def save_leakage_report(report: dict, output_dir: str = "experiments/leakage") -> Path:
    """Save leakage report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    filepath = out_path / f"leak-{date}.yaml"

    clean = json.loads(json.dumps(report, default=str))
    with open(filepath, "w") as f:
        yaml.dump(clean, f, default_flow_style=False, sort_keys=False)

    return filepath


def format_leakage_report(report: dict) -> str:
    """Format leakage report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}\n{report.get('note', '')}"

    verdict = report.get("verdict", "?")
    verdict_labels = {
        "leakage_detected": "LEAKAGE DETECTED — investigate flagged features",
        "suspicious": "SUSPICIOUS — review warnings",
        "clean": "CLEAN — no leakage detected",
    }

    lines = [
        "# Leakage Scan",
        "",
        f"*Scanned {report.get('scanned_at', 'N/A')[:19]}*",
        f"*Mode: {'deep' if report.get('deep_mode') else 'standard'}*",
        "",
        f"**{verdict_labels.get(verdict, verdict.upper())}**",
        "",
    ]

    for c in report.get("checks", []):
        check_name = c.get("check", "?")
        status = c.get("status", "?")
        marker = {"pass": "OK", "fail": "FLAG", "warn": "WARN"}.get(status, status.upper())

        lines.append(f"### {check_name}")
        lines.append(f"**[{marker}]** {c.get('reason', 'N/A')}")
        lines.append("")

        flags = c.get("flags", [])
        for f in flags[:5]:
            lines.append(f"- **{f.get('feature', '?')}**: {f.get('reason', 'N/A')}")
        if len(flags) > 5:
            lines.append(f"  *...and {len(flags) - 5} more*")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Targeted leakage detection",
    )
    parser.add_argument(
        "--deep", action="store_true",
        help="Run full single-feature analysis (slower but thorough)",
    )
    parser.add_argument(
        "--features",
        help="Specific features to check (comma-separated)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    # In CLI mode without data args, produce a plan
    report = run_leakage_scan(config_path=args.config, deep=args.deep)

    if "error" not in report:
        filepath = save_leakage_report(report)
        print(f"Saved to {filepath}", file=sys.stderr)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_leakage_report(report))

    if report.get("verdict") == "leakage_detected":
        sys.exit(1)


if __name__ == "__main__":
    main()
