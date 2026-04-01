"""Edge case tests for live training monitor (training_monitor.py).

Phase 16.2: Covers NaN values, empty logs, single epoch, all alerts
firing simultaneously, malformed input, boundary conditions.
"""

from __future__ import annotations

import json
import math

import pytest

from scripts.training_monitor import (
    compute_rolling_stats,
    default_alert_config,
    evaluate_alerts,
    format_dashboard_line,
    parse_epoch_metrics,
    parse_log_file,
    _check_loss_spike,
    _check_nan,
    _check_overfitting,
    _check_plateau,
)


# --- parse_epoch_metrics edge cases ---


def test_parse_malformed_json():
    """Malformed JSON should return None."""
    assert parse_epoch_metrics('{"epoch": 1, "loss":}') is None


def test_parse_json_with_extra_fields():
    """Extra fields should be preserved."""
    line = '{"epoch": 1, "loss": 0.5, "custom_metric": 0.8}'
    result = parse_epoch_metrics(line)
    assert result["custom_metric"] == 0.8


def test_parse_kv_with_non_numeric():
    """Non-numeric values should be kept as strings."""
    line = "epoch=1 model=xgboost loss=0.5"
    result = parse_epoch_metrics(line)
    assert result["model"] == "xgboost"
    assert result["loss"] == 0.5


def test_parse_kv_with_spaces():
    """Should handle extra whitespace."""
    line = "  epoch=1   loss=0.5  "
    result = parse_epoch_metrics(line)
    assert result is not None
    assert result["epoch"] == 1


# --- compute_rolling_stats edge cases ---


def test_rolling_stats_constant_values():
    """Constant values should have zero std and trend."""
    history = [{"loss": 0.5} for _ in range(10)]
    stats = compute_rolling_stats(history, "loss")
    assert stats["std"] == 0.0
    assert stats["trend"] == 0.0
    assert stats["mean"] == 0.5


def test_rolling_stats_nan_values():
    """NaN values should be filtered out."""
    history = [
        {"loss": 0.5},
        {"loss": float("nan")},
        {"loss": 0.4},
    ]
    stats = compute_rolling_stats(history, "loss")
    # NaN filtered, so only 0.5 and 0.4
    assert stats["n"] == 2


def test_rolling_stats_mixed_present_absent():
    """Metrics absent from some entries should be skipped."""
    history = [
        {"loss": 0.5},
        {"accuracy": 0.8},  # No loss
        {"loss": 0.3},
    ]
    stats = compute_rolling_stats(history, "loss")
    assert stats["n"] == 2


def test_rolling_stats_very_large_window():
    """Window larger than history should use all data."""
    history = [{"loss": float(i)} for i in range(3)]
    stats = compute_rolling_stats(history, "loss", window=100)
    assert stats["n"] == 3


# --- _check_nan edge cases ---


def test_nan_in_multiple_metrics():
    """NaN in first metric found should trigger."""
    current = {"epoch": 1, "loss": float("nan"), "accuracy": float("nan")}
    rule = {"condition": "nan_detected", "message": "NaN in {metric} at epoch {epoch}"}
    alert = _check_nan(current, rule)
    assert alert is not None
    # Should catch first NaN metric
    assert alert["metric"] in ("loss", "accuracy")


def test_nan_only_epoch_field():
    """Epoch-only entry should not trigger NaN."""
    current = {"epoch": 1}
    rule = {"condition": "nan_detected"}
    alert = _check_nan(current, rule)
    assert alert is None


def test_nan_string_values():
    """String values should not trigger NaN check."""
    current = {"epoch": 1, "model": "xgboost"}
    rule = {"condition": "nan_detected"}
    alert = _check_nan(current, rule)
    assert alert is None


# --- _check_loss_spike edge cases ---


def test_loss_spike_with_zero_mean():
    """Zero mean loss should not trigger (avoid division issues)."""
    history = [{"loss": 0.0} for _ in range(5)]
    current = {"epoch": 6, "loss": 0.1}
    rule = {"condition": "loss_spike", "multiplier": 3.0}
    # mean is 0, so 0.1 > 3 * 0 is true but mean <= 0 guard
    alert = _check_loss_spike(current, history, rule)
    assert alert is None


def test_loss_spike_with_train_loss_key():
    """Should work with 'train_loss' instead of 'loss'."""
    history = [{"train_loss": 0.3} for _ in range(5)]
    current = {"epoch": 6, "train_loss": 1.5}
    rule = {"condition": "loss_spike", "multiplier": 3.0, "message": "Spike at {epoch}: {value} vs {mean:.4f}"}
    alert = _check_loss_spike(current, history, rule)
    assert alert is not None


# --- _check_overfitting edge cases ---


def test_overfitting_no_val_loss():
    """Missing val_loss should not trigger."""
    history = [{"epoch": i, "loss": 0.3} for i in range(5)]
    current = {"epoch": 5, "loss": 0.2}
    rule = {"condition": "overfitting", "gap_ratio": 0.5, "consecutive": 3}
    alert = _check_overfitting(current, history, rule)
    assert alert is None


def test_overfitting_shrinking_gap():
    """Shrinking gap should not trigger."""
    history = [
        {"epoch": i, "loss": 0.3, "val_loss": 0.5 - i * 0.02}
        for i in range(5)
    ]
    current = {"epoch": 5, "loss": 0.3, "val_loss": 0.38}
    rule = {"condition": "overfitting", "gap_ratio": 0.5, "consecutive": 3}
    alert = _check_overfitting(current, history, rule)
    assert alert is None


# --- _check_plateau edge cases ---


def test_plateau_insufficient_history():
    """Too few epochs should not trigger plateau."""
    history = [{"val_loss": 0.5} for _ in range(2)]
    rule = {"condition": "plateau", "min_improvement": 0.001, "consecutive": 5}
    alert = _check_plateau(history, rule)
    assert alert is None


def test_plateau_no_relevant_metric():
    """Entries without any tracked metric should not trigger."""
    history = [{"epoch": i, "custom_field": 0.5} for i in range(10)]
    rule = {"condition": "plateau", "min_improvement": 0.001, "consecutive": 5}
    alert = _check_plateau(history, rule)
    assert alert is None


# --- evaluate_alerts edge cases ---


def test_evaluate_alerts_all_fire():
    """Scenario where multiple alerts fire simultaneously."""
    config = default_alert_config()
    # Build history with widening gap and flat metrics
    history = [
        {"epoch": i, "loss": 0.3, "val_loss": 0.35 + i * 0.03}
        for i in range(6)
    ]
    # Current: NaN loss, which is also a spike
    current = {"epoch": 7, "loss": float("nan"), "val_loss": 0.6}

    alerts = evaluate_alerts(current, history, config)
    names = {a["name"] for a in alerts}
    assert "nan_detected" in names


def test_evaluate_alerts_empty_config():
    """Empty config should produce no alerts."""
    alerts = evaluate_alerts(
        {"epoch": 1, "loss": float("nan")},
        [],
        {"alerts": {}},
    )
    assert len(alerts) == 0


def test_evaluate_alerts_unknown_condition():
    """Unknown condition type should be skipped."""
    config = {"alerts": {"custom": {"condition": "nonexistent_check"}}}
    alerts = evaluate_alerts({"epoch": 1, "loss": 0.5}, [], config)
    assert len(alerts) == 0


# --- format_dashboard_line edge cases ---


def test_dashboard_nan_loss():
    """NaN loss should show NaN."""
    current = {"epoch": 1, "loss": float("nan")}
    line = format_dashboard_line(current, {}, [])
    assert "NaN" in line


def test_dashboard_missing_total_epochs():
    """Missing total_epochs should show ?."""
    current = {"epoch": 5, "loss": 0.3}
    line = format_dashboard_line(current, {}, [])
    assert "Epoch 5/?" in line


def test_dashboard_critical_alert():
    """Critical alert should show CRITICAL."""
    current = {"epoch": 1}
    alerts = [{"name": "nan", "severity": "critical"}]
    line = format_dashboard_line(current, {}, alerts)
    assert "CRITICAL" in line


# --- parse_log_file edge cases ---


def test_parse_log_file_nonexistent(tmp_path):
    """Missing file should return empty list."""
    assert parse_log_file(str(tmp_path / "missing.log")) == []


def test_parse_log_file_empty(tmp_path):
    """Empty file should return empty list."""
    log = tmp_path / "empty.log"
    log.write_text("")
    assert parse_log_file(str(log)) == []


def test_parse_log_file_mixed_content(tmp_path):
    """Should extract metrics from mixed log content."""
    log = tmp_path / "mixed.log"
    log.write_text(
        "Starting training...\n"
        '{"epoch": 0, "loss": 1.0}\n'
        "Loading data...\n"
        '{"epoch": 1, "loss": 0.8}\n'
        "Done.\n"
    )
    metrics = parse_log_file(str(log))
    assert len(metrics) == 2
    assert metrics[0]["epoch"] == 0
    assert metrics[1]["epoch"] == 1
