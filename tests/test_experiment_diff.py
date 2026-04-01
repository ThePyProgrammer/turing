"""Tests for deep experiment comparison (experiment_diff.py).

Phase 16.1: Verifies config diff, metric diff with significance,
per-class diff with regression detection, training curve divergence,
feature importance shifts, and report formatting.
"""

from __future__ import annotations

import math

import pytest

from scripts.experiment_diff import (
    diff_configs,
    diff_feature_importance,
    diff_metrics,
    diff_per_class,
    find_curve_divergence,
    format_diff_report,
    _build_summary,
    _flatten_dict,
    _norm_cdf,
)


# --- _flatten_dict ---


def test_flatten_simple():
    """Simple dict should remain unchanged."""
    assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_flatten_nested():
    """Nested dict should use dot-separated keys."""
    result = _flatten_dict({"a": {"b": 1, "c": {"d": 2}}})
    assert result == {"a.b": 1, "a.c.d": 2}


def test_flatten_empty():
    """Empty dict should return empty."""
    assert _flatten_dict({}) == {}


# --- diff_configs ---


def test_config_diff_no_changes():
    """Identical configs should show no changes."""
    config = {"learning_rate": 0.1, "max_depth": 6}
    diffs = diff_configs(config, config)
    assert all(not d["changed"] for d in diffs)


def test_config_diff_numeric_change():
    """Numeric change should include pct_change."""
    diffs = diff_configs(
        {"max_depth": 6, "learning_rate": 0.1},
        {"max_depth": 8, "learning_rate": 0.1},
    )
    changed = [d for d in diffs if d["changed"]]
    assert len(changed) == 1
    assert changed[0]["key"] == "max_depth"
    assert changed[0]["value_a"] == 6
    assert changed[0]["value_b"] == 8
    assert abs(changed[0]["pct_change"] - 33.33) < 1


def test_config_diff_string_change():
    """String change should not have pct_change."""
    diffs = diff_configs(
        {"model_type": "xgboost"},
        {"model_type": "lightgbm"},
    )
    changed = [d for d in diffs if d["changed"]]
    assert len(changed) == 1
    assert "pct_change" not in changed[0]


def test_config_diff_nested():
    """Nested config should be flattened and compared."""
    diffs = diff_configs(
        {"hyperparams": {"lr": 0.01, "depth": 6}},
        {"hyperparams": {"lr": 0.01, "depth": 8}},
    )
    changed = [d for d in diffs if d["changed"]]
    assert len(changed) == 1
    assert changed[0]["key"] == "hyperparams.depth"


def test_config_diff_missing_key():
    """Key present in only one config should show as changed."""
    diffs = diff_configs(
        {"a": 1},
        {"a": 1, "b": 2},
    )
    changed = [d for d in diffs if d["changed"]]
    assert len(changed) == 1
    assert changed[0]["key"] == "b"
    assert changed[0]["value_a"] is None


# --- diff_metrics ---


def test_metric_diff_basic():
    """Should compute deltas between metrics."""
    diffs = diff_metrics(
        {"accuracy": 0.85, "f1": 0.83},
        {"accuracy": 0.87, "f1": 0.82},
    )
    acc = next(d for d in diffs if d["metric"] == "accuracy")
    assert abs(acc["delta"] - 0.02) < 1e-6
    assert acc["direction"] == "better"

    f1 = next(d for d in diffs if d["metric"] == "f1")
    assert abs(f1["delta"] - (-0.01)) < 1e-6
    assert f1["direction"] == "worse"


def test_metric_diff_lower_is_better():
    """Lower-is-better metrics should have inverted direction."""
    diffs = diff_metrics(
        {"loss": 0.30},
        {"loss": 0.25},
        lower_is_better_metrics={"loss"},
    )
    loss = next(d for d in diffs if d["metric"] == "loss")
    assert loss["direction"] == "better"  # Lower is better, delta < 0


def test_metric_diff_with_significance():
    """Seed studies should add significance info."""
    seed_studies = {
        "exp-a": {"per_metric": {"accuracy": {"std": 0.005}}},
    }
    diffs = diff_metrics(
        {"accuracy": 0.85},
        {"accuracy": 0.87},
        seed_studies=seed_studies,
    )
    acc = next(d for d in diffs if d["metric"] == "accuracy")
    assert acc.get("significance") is not None
    assert "p_value" in acc["significance"]


def test_metric_diff_no_significance_without_studies():
    """Without seed studies, no significance info."""
    diffs = diff_metrics(
        {"accuracy": 0.85},
        {"accuracy": 0.87},
    )
    acc = next(d for d in diffs if d["metric"] == "accuracy")
    assert acc.get("significance") is None


# --- diff_per_class ---


def test_per_class_diff_with_regression():
    """Should detect per-class regressions."""
    diffs = diff_per_class(
        {"class_0": {"precision": 0.91}, "class_1": {"precision": 0.83}},
        {"class_0": {"precision": 0.93}, "class_1": {"precision": 0.79}},
    )
    assert len(diffs) == 2
    regressed = [d for d in diffs if d.get("regression")]
    assert len(regressed) == 1
    assert regressed[0]["class"] == "class_1"


def test_per_class_diff_no_regression():
    """Small changes should not flag as regression."""
    diffs = diff_per_class(
        {"class_0": {"precision": 0.91}},
        {"class_0": {"precision": 0.905}},
    )
    assert all(not d.get("regression") for d in diffs)


def test_per_class_diff_empty():
    """None inputs should return empty list."""
    assert diff_per_class(None, None) == []
    assert diff_per_class(None, {"a": {"p": 0.5}}) == []


# --- find_curve_divergence ---


def test_curve_divergence_found():
    """Should find divergence point."""
    curve_a = [{"epoch": i, "loss": 1.0 - i * 0.1} for i in range(10)]
    curve_b = [{"epoch": i, "loss": 1.0 - i * 0.1 if i < 5 else 0.8} for i in range(10)]

    result = find_curve_divergence(curve_a, curve_b, metric="loss", threshold=0.05)
    assert result is not None
    assert result["divergence_epoch"] >= 5


def test_curve_divergence_identical():
    """Identical curves should return None."""
    curve = [{"epoch": i, "loss": 1.0 - i * 0.1} for i in range(10)]
    assert find_curve_divergence(curve, curve) is None


def test_curve_divergence_none_inputs():
    """None inputs should return None."""
    assert find_curve_divergence(None, None) is None
    assert find_curve_divergence([], None) is None


def test_curve_divergence_no_overlap():
    """Non-overlapping epochs should return None."""
    curve_a = [{"epoch": 0, "loss": 1.0}]
    curve_b = [{"epoch": 10, "loss": 0.5}]
    assert find_curve_divergence(curve_a, curve_b) is None


# --- diff_feature_importance ---


def test_fi_diff_basic():
    """Should rank by absolute delta."""
    diffs = diff_feature_importance(
        {"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2},
        {"feat_a": 0.3, "feat_b": 0.5, "feat_c": 0.2},
    )
    assert len(diffs) >= 2
    # feat_a and feat_b both changed by 0.2, feat_c unchanged
    assert diffs[0]["abs_delta"] >= diffs[1]["abs_delta"]


def test_fi_diff_top_k():
    """Should respect top_k limit."""
    imp = {f"f{i}": float(i) / 10 for i in range(20)}
    imp2 = {f"f{i}": float(i + 1) / 10 for i in range(20)}
    diffs = diff_feature_importance(imp, imp2, top_k=5)
    assert len(diffs) == 5


def test_fi_diff_empty():
    """None inputs should return empty list."""
    assert diff_feature_importance(None, None) == []


# --- _norm_cdf ---


def test_norm_cdf_zero():
    """CDF at 0 should be 0.5."""
    assert abs(_norm_cdf(0) - 0.5) < 1e-6


def test_norm_cdf_large():
    """CDF at large positive should approach 1."""
    assert _norm_cdf(5) > 0.999


def test_norm_cdf_negative():
    """CDF at large negative should approach 0."""
    assert _norm_cdf(-5) < 0.001


# --- _build_summary ---


def test_build_summary():
    """Summary should count diffs correctly."""
    report = {
        "config_diff": [
            {"key": "a", "changed": True},
            {"key": "b", "changed": False},
        ],
        "metric_diff": [
            {"metric": "accuracy", "delta": 0.02, "direction": "better"},
            {"metric": "f1", "delta": 0.0, "direction": "same"},
        ],
        "per_class_diff": [
            {"class": "c0", "regression": True},
        ],
        "curve_divergence": {"divergence_epoch": 5, "metric": "loss"},
        "feature_importance_diff": [{"feature": "x1"}],
    }
    summary = _build_summary(report, "accuracy")
    assert summary["config_changes"] == 1
    assert summary["metric_changes"] == 1
    assert summary["per_class_regressions"] == 1
    assert summary["has_curve_divergence"] is True
    assert summary["divergence_epoch"] == 5
    assert summary["primary_metric_delta"] == 0.02


# --- format_diff_report ---


def test_format_report_basic():
    """Should produce readable markdown."""
    report = {
        "experiment_a": "exp-001",
        "experiment_b": "exp-002",
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "config_diff": [
            {"key": "lr", "value_a": 0.01, "value_b": 0.02, "changed": True, "pct_change": 100},
        ],
        "metric_diff": [
            {"metric": "accuracy", "value_a": 0.85, "value_b": 0.87, "delta": 0.02, "direction": "better"},
        ],
        "per_class_diff": [],
        "curve_divergence": None,
        "feature_importance_diff": [],
        "summary": {
            "config_changes": 1,
            "metric_changes": 1,
            "per_class_regressions": 0,
            "has_curve_divergence": False,
            "divergence_epoch": None,
            "feature_importance_shifts": 0,
            "primary_metric_delta": 0.02,
            "primary_metric_direction": "better",
        },
    }
    text = format_diff_report(report)
    assert "exp-001" in text
    assert "exp-002" in text
    assert "Config Changes" in text
    assert "Metric Comparison" in text
    assert "+0.0200" in text


def test_format_report_error():
    """Error should show error message."""
    text = format_diff_report({"error": "Experiment not found"})
    assert "ERROR" in text


def test_format_report_with_regressions():
    """Should highlight regressions."""
    report = {
        "experiment_a": "exp-001",
        "experiment_b": "exp-002",
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "config_diff": [],
        "metric_diff": [],
        "per_class_diff": [
            {"class": "c1", "metric": "precision", "value_a": 0.83, "value_b": 0.79, "delta": -0.04, "regression": True},
        ],
        "curve_divergence": {"divergence_epoch": 12, "value_a": 0.5, "value_b": 0.7, "relative_diff": 0.4, "metric": "loss", "total_common_epochs": 50},
        "feature_importance_diff": [],
        "summary": {
            "config_changes": 0,
            "metric_changes": 0,
            "per_class_regressions": 1,
            "has_curve_divergence": True,
            "divergence_epoch": 12,
            "feature_importance_shifts": 0,
            "primary_metric_delta": None,
            "primary_metric_direction": None,
        },
    }
    text = format_diff_report(report)
    assert "REGRESSION" in text
    assert "epoch 12" in text
