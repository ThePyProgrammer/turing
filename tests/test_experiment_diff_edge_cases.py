"""Edge case tests for deep experiment comparison (experiment_diff.py).

Phase 16.1: Covers missing fields, empty experiments, identical experiments,
zero values, single-metric experiments, large configs, and boundary conditions.
"""

from __future__ import annotations

import pytest

from scripts.experiment_diff import (
    diff_configs,
    diff_feature_importance,
    diff_metrics,
    diff_per_class,
    find_curve_divergence,
    format_diff_report,
    _build_summary,
    _check_significance,
    _flatten_dict,
)


# --- diff_configs edge cases ---


def test_config_diff_zero_base_value():
    """Zero base value should produce inf pct_change."""
    diffs = diff_configs({"param": 0}, {"param": 5})
    changed = [d for d in diffs if d["changed"]]
    assert len(changed) == 1
    assert changed[0]["pct_change"] == float("inf")


def test_config_diff_zero_to_zero():
    """Zero to zero should not be marked as changed."""
    diffs = diff_configs({"param": 0}, {"param": 0})
    assert all(not d["changed"] for d in diffs)


def test_config_diff_both_empty():
    """Two empty configs should produce empty diff."""
    assert diff_configs({}, {}) == []


def test_config_diff_deeply_nested():
    """Deeply nested configs should flatten correctly."""
    diffs = diff_configs(
        {"a": {"b": {"c": {"d": 1}}}},
        {"a": {"b": {"c": {"d": 2}}}},
    )
    changed = [d for d in diffs if d["changed"]]
    assert len(changed) == 1
    assert changed[0]["key"] == "a.b.c.d"


def test_config_diff_mixed_types():
    """Different types (int vs string) should show as changed."""
    diffs = diff_configs({"param": 42}, {"param": "42"})
    changed = [d for d in diffs if d["changed"]]
    assert len(changed) == 1


def test_config_diff_negative_pct_change():
    """Decrease should produce negative pct_change."""
    diffs = diff_configs({"n_estimators": 500}, {"n_estimators": 300})
    changed = [d for d in diffs if d["changed"]]
    assert changed[0]["pct_change"] == -40.0


# --- diff_metrics edge cases ---


def test_metric_diff_empty_metrics():
    """Empty metric dicts should produce empty diff."""
    assert diff_metrics({}, {}) == []


def test_metric_diff_missing_in_one():
    """Metric in only one experiment should appear with None."""
    diffs = diff_metrics({"accuracy": 0.85}, {"accuracy": 0.85, "f1": 0.80})
    f1 = next(d for d in diffs if d["metric"] == "f1")
    assert f1["value_a"] is None
    assert f1["value_b"] == 0.80


def test_metric_diff_zero_delta():
    """Identical metrics should show 'same' direction."""
    diffs = diff_metrics({"accuracy": 0.85}, {"accuracy": 0.85})
    acc = next(d for d in diffs if d["metric"] == "accuracy")
    assert acc["delta"] == 0
    assert acc["direction"] == "same"


def test_metric_diff_metadata_keys_na():
    """Metadata keys like train_seconds should have N/A direction."""
    diffs = diff_metrics(
        {"train_seconds": 100.0},
        {"train_seconds": 200.0},
    )
    ts = next(d for d in diffs if d["metric"] == "train_seconds")
    assert ts["direction"] == "N/A"


# --- _check_significance edge cases ---


def test_significance_zero_variance():
    """Zero variance should use exact comparison."""
    result = _check_significance(
        "accuracy", 0.85, 0.86,
        {"exp-a": {"per_metric": {"accuracy": {"std": 0.0}}}},
    )
    assert result is not None
    assert result["method"] == "zero_variance"
    assert result["significant"] is True


def test_significance_identical_values():
    """Same values with zero variance should not be significant."""
    result = _check_significance(
        "accuracy", 0.85, 0.85,
        {"exp-a": {"per_metric": {"accuracy": {"std": 0.0}}}},
    )
    assert result is not None
    assert result["significant"] is False


def test_significance_no_matching_metric():
    """Missing metric in seed study should return None."""
    result = _check_significance(
        "f1", 0.80, 0.82,
        {"exp-a": {"per_metric": {"accuracy": {"std": 0.01}}}},
    )
    assert result is None


def test_significance_empty_studies():
    """Empty seed studies should return None."""
    result = _check_significance("accuracy", 0.85, 0.86, {})
    assert result is None


# --- diff_per_class edge cases ---


def test_per_class_diff_new_class():
    """Class in only one experiment should appear."""
    diffs = diff_per_class(
        {"class_0": {"precision": 0.90}},
        {"class_0": {"precision": 0.90}, "class_1": {"precision": 0.80}},
    )
    new_class = [d for d in diffs if d["class"] == "class_1"]
    assert len(new_class) == 1
    assert new_class[0]["value_a"] is None


def test_per_class_diff_below_regression_threshold():
    """Small decrease (-0.005) should not flag as regression (threshold is <-0.01)."""
    diffs = diff_per_class(
        {"class_0": {"precision": 0.90}},
        {"class_0": {"precision": 0.895}},
    )
    assert all(not d.get("regression") for d in diffs)


def test_per_class_diff_barely_regression():
    """Just over threshold should flag."""
    diffs = diff_per_class(
        {"class_0": {"precision": 0.90}},
        {"class_0": {"precision": 0.8899}},
    )
    regressed = [d for d in diffs if d.get("regression")]
    assert len(regressed) == 1


# --- find_curve_divergence edge cases ---


def test_curve_divergence_single_epoch():
    """Single epoch curves should not diverge (or diverge immediately)."""
    curve_a = [{"epoch": 0, "loss": 1.0}]
    curve_b = [{"epoch": 0, "loss": 2.0}]
    result = find_curve_divergence(curve_a, curve_b, threshold=0.05)
    # 100% difference, should diverge at epoch 0
    assert result is not None
    assert result["divergence_epoch"] == 0


def test_curve_divergence_empty_lists():
    """Empty lists should return None."""
    assert find_curve_divergence([], []) is None


def test_curve_divergence_missing_metric():
    """Entries without the target metric should be skipped."""
    curve_a = [{"epoch": 0, "accuracy": 0.5}]  # No 'loss'
    curve_b = [{"epoch": 0, "loss": 0.5}]
    assert find_curve_divergence(curve_a, curve_b, metric="loss") is None


def test_curve_divergence_high_threshold():
    """High threshold should not trigger divergence."""
    curve_a = [{"epoch": i, "loss": 1.0 - i * 0.1} for i in range(5)]
    curve_b = [{"epoch": i, "loss": 1.0 - i * 0.09} for i in range(5)]
    assert find_curve_divergence(curve_a, curve_b, threshold=0.5) is None


# --- diff_feature_importance edge cases ---


def test_fi_diff_one_empty():
    """One empty importance dict should return empty."""
    assert diff_feature_importance({"a": 0.5}, None) == []
    assert diff_feature_importance(None, {"a": 0.5}) == []


def test_fi_diff_new_feature():
    """Feature in only one experiment should have 0 for the other."""
    diffs = diff_feature_importance(
        {"feat_a": 0.5},
        {"feat_a": 0.5, "feat_b": 0.3},
    )
    new = next(d for d in diffs if d["feature"] == "feat_b")
    assert new["importance_a"] == 0.0
    assert new["importance_b"] == 0.3


def test_fi_diff_top_k_zero():
    """top_k=0 should return empty."""
    diffs = diff_feature_importance({"a": 0.5}, {"a": 0.6}, top_k=0)
    assert len(diffs) == 0


# --- format_diff_report edge cases ---


def test_format_report_no_changes():
    """Report with no changes should still render."""
    report = {
        "experiment_a": "exp-001",
        "experiment_b": "exp-002",
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "config_diff": [],
        "metric_diff": [],
        "per_class_diff": [],
        "curve_divergence": None,
        "feature_importance_diff": [],
        "summary": {
            "config_changes": 0,
            "metric_changes": 0,
            "per_class_regressions": 0,
            "has_curve_divergence": False,
            "divergence_epoch": None,
            "feature_importance_shifts": 0,
            "primary_metric_delta": None,
            "primary_metric_direction": None,
        },
    }
    text = format_diff_report(report)
    assert "No config differences" in text


def test_format_report_with_code_diff():
    """Code diff should be included when present."""
    report = {
        "experiment_a": "exp-001",
        "experiment_b": "exp-002",
        "generated_at": "2026-01-01T00:00:00",
        "primary_metric": "accuracy",
        "config_diff": [],
        "metric_diff": [],
        "per_class_diff": [],
        "curve_divergence": None,
        "feature_importance_diff": [],
        "code_diff": "- old line\n+ new line",
        "summary": {
            "config_changes": 0, "metric_changes": 0,
            "per_class_regressions": 0, "has_curve_divergence": False,
            "divergence_epoch": None, "feature_importance_shifts": 0,
            "primary_metric_delta": None, "primary_metric_direction": None,
        },
    }
    text = format_diff_report(report)
    assert "Code Changes" in text
    assert "- old line" in text


# --- _build_summary edge cases ---


def test_build_summary_empty_report():
    """Empty report should produce zero counts."""
    report = {
        "config_diff": [],
        "metric_diff": [],
        "per_class_diff": [],
        "curve_divergence": None,
        "feature_importance_diff": [],
    }
    summary = _build_summary(report, "accuracy")
    assert summary["config_changes"] == 0
    assert summary["metric_changes"] == 0
    assert summary["per_class_regressions"] == 0
    assert summary["has_curve_divergence"] is False
    assert summary["primary_metric_delta"] is None
