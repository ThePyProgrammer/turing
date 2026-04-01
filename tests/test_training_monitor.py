"""Tests for live training monitor (training_monitor.py).

Phase 16.2: Verifies metric parsing, rolling statistics, alert rule
evaluation, dashboard formatting, and analysis reporting.
"""

from __future__ import annotations

import math

import pytest

from scripts.training_monitor import (
    analyze_training_log,
    compute_rolling_stats,
    default_alert_config,
    evaluate_alerts,
    format_analysis_report,
    format_dashboard_line,
    parse_epoch_metrics,
    parse_log_file,
    _check_loss_spike,
    _check_nan,
    _check_overfitting,
    _check_plateau,
)


# --- parse_epoch_metrics ---


def test_parse_json_format():
    """Should parse JSON log lines."""
    line = '{"epoch": 5, "loss": 0.342, "val_loss": 0.401}'
    result = parse_epoch_metrics(line)
    assert result is not None
    assert result["epoch"] == 5
    assert result["loss"] == 0.342


def test_parse_kv_format_equals():
    """Should parse key=value format."""
    line = "epoch=5 loss=0.342 val_loss=0.401"
    result = parse_epoch_metrics(line)
    assert result is not None
    assert result["epoch"] == 5
    assert abs(result["loss"] - 0.342) < 1e-6


def test_parse_kv_format_colon():
    """Should parse key:value format."""
    line = "epoch:5 loss:0.342"
    result = parse_epoch_metrics(line)
    assert result is not None
    assert result["epoch"] == 5


def test_parse_empty_line():
    """Empty line should return None."""
    assert parse_epoch_metrics("") is None
    assert parse_epoch_metrics("   ") is None


def test_parse_non_metric_line():
    """Random text should return None."""
    assert parse_epoch_metrics("Training started...") is None


def test_parse_json_without_epoch():
    """JSON without epoch key should return None."""
    assert parse_epoch_metrics('{"loss": 0.5}') is None


# --- compute_rolling_stats ---


def test_rolling_stats_basic():
    """Should compute mean, std, trend."""
    history = [{"loss": 1.0 - i * 0.1} for i in range(5)]
    stats = compute_rolling_stats(history, "loss")
    assert abs(stats["mean"] - 0.8) < 0.01
    assert stats["trend"] < 0  # Decreasing
    assert stats["min"] == 0.6
    assert stats["max"] == 1.0
    assert stats["n"] == 5


def test_rolling_stats_empty():
    """Empty history should return empty dict."""
    assert compute_rolling_stats([], "loss") == {}


def test_rolling_stats_single_value():
    """Single value should have zero std and trend."""
    stats = compute_rolling_stats([{"loss": 0.5}], "loss")
    assert stats["mean"] == 0.5
    assert stats["std"] == 0.0
    assert stats["trend"] == 0.0


def test_rolling_stats_missing_metric():
    """History without the metric should return empty."""
    stats = compute_rolling_stats([{"accuracy": 0.85}], "loss")
    assert stats == {}


def test_rolling_stats_window():
    """Should respect rolling window size."""
    history = [{"loss": float(i)} for i in range(20)]
    stats = compute_rolling_stats(history, "loss", window=5)
    assert stats["n"] == 5
    # Last 5 values: 15, 16, 17, 18, 19
    assert stats["mean"] == 17.0


# --- Alert evaluation ---


def test_loss_spike_detected():
    """Loss > 3x rolling mean should trigger."""
    history = [{"loss": 0.3} for _ in range(5)]
    current = {"epoch": 6, "loss": 1.5}  # 5x mean
    rule = {"condition": "loss_spike", "multiplier": 3.0, "severity": "warning", "message": "Spike at {epoch}: {value} vs {mean:.4f}"}
    alert = _check_loss_spike(current, history, rule)
    assert alert is not None
    assert "Spike" in alert["message"]


def test_loss_spike_not_triggered():
    """Normal loss should not trigger."""
    history = [{"loss": 0.3} for _ in range(5)]
    current = {"epoch": 6, "loss": 0.35}
    rule = {"condition": "loss_spike", "multiplier": 3.0}
    alert = _check_loss_spike(current, history, rule)
    assert alert is None


def test_loss_spike_insufficient_history():
    """Too little history should not trigger."""
    history = [{"loss": 0.3}]
    current = {"epoch": 2, "loss": 1.5}
    rule = {"condition": "loss_spike", "multiplier": 3.0}
    alert = _check_loss_spike(current, history, rule)
    assert alert is None


def test_nan_detected():
    """NaN in any metric should trigger."""
    current = {"epoch": 10, "loss": float("nan")}
    rule = {"condition": "nan_detected", "severity": "critical", "message": "NaN in {metric} at epoch {epoch}"}
    alert = _check_nan(current, rule)
    assert alert is not None
    assert alert["metric"] == "loss"


def test_nan_not_detected():
    """Normal values should not trigger NaN alert."""
    current = {"epoch": 10, "loss": 0.5}
    rule = {"condition": "nan_detected"}
    alert = _check_nan(current, rule)
    assert alert is None


def test_overfitting_detected():
    """Widening train/val gap should trigger."""
    history = [
        {"epoch": i, "loss": 0.2, "val_loss": 0.3 + i * 0.05}
        for i in range(5)
    ]
    current = {"epoch": 5, "loss": 0.1, "val_loss": 0.65}
    rule = {"condition": "overfitting", "gap_ratio": 0.5, "consecutive": 3, "message": "Overfitting since epoch {onset}"}
    alert = _check_overfitting(current, history, rule)
    assert alert is not None


def test_overfitting_not_detected():
    """Stable gap should not trigger."""
    history = [
        {"epoch": i, "loss": 0.3, "val_loss": 0.35}
        for i in range(5)
    ]
    current = {"epoch": 5, "loss": 0.3, "val_loss": 0.35}
    rule = {"condition": "overfitting", "gap_ratio": 0.5, "consecutive": 3}
    alert = _check_overfitting(current, history, rule)
    assert alert is None


def test_plateau_detected():
    """Flat metric should trigger plateau."""
    history = [{"val_loss": 0.30001 + i * 0.00001} for i in range(6)]
    rule = {"condition": "plateau", "min_improvement": 0.001, "consecutive": 5, "message": "Plateaued"}
    alert = _check_plateau(history, rule)
    assert alert is not None


def test_plateau_not_detected():
    """Improving metric should not trigger."""
    history = [{"val_loss": 0.5 - i * 0.05} for i in range(6)]
    rule = {"condition": "plateau", "min_improvement": 0.001, "consecutive": 5}
    alert = _check_plateau(history, rule)
    assert alert is None


# --- evaluate_alerts (integration) ---


def test_evaluate_alerts_multiple():
    """Should evaluate all configured alert rules."""
    config = default_alert_config()
    history = [{"loss": 0.3, "val_loss": 0.35} for _ in range(5)]
    current = {"epoch": 6, "loss": float("nan"), "val_loss": 0.35}

    alerts = evaluate_alerts(current, history, config)
    # NaN should trigger
    nan_alerts = [a for a in alerts if a["name"] == "nan_detected"]
    assert len(nan_alerts) >= 1


def test_evaluate_alerts_no_triggers():
    """Normal training should produce no alerts."""
    config = default_alert_config()
    history = [{"loss": 0.5 - i * 0.01, "val_loss": 0.55 - i * 0.01} for i in range(5)]
    current = {"epoch": 6, "loss": 0.44, "val_loss": 0.49}

    alerts = evaluate_alerts(current, history, config)
    assert len(alerts) == 0


# --- format_dashboard_line ---


def test_dashboard_basic():
    """Should format a compact dashboard line."""
    current = {"epoch": 23, "total_epochs": 100, "loss": 0.342, "accuracy": 0.865}
    rolling = {"trend": -0.01}
    line = format_dashboard_line(current, rolling, [])
    assert "Epoch 23/100" in line
    assert "loss: 0.342" in line
    assert "↓" in line


def test_dashboard_with_alert():
    """Should include alert indicator."""
    current = {"epoch": 10, "loss": 1.5}
    alerts = [{"name": "loss_spike", "severity": "warning"}]
    line = format_dashboard_line(current, {}, alerts)
    assert "WARNING" in line


def test_dashboard_with_gap():
    """Should show train/val gap."""
    current = {"epoch": 5, "loss": 0.3, "val_loss": 0.35}
    line = format_dashboard_line(current, {}, [])
    assert "gap: 0.050" in line


# --- format_analysis_report ---


def test_format_analysis_report_basic():
    """Should produce readable markdown."""
    report = {
        "log_path": "run.log",
        "analyzed_at": "2026-01-01T00:00:00",
        "total_epochs": 50,
        "alerts": [
            {"severity": "warning", "message": "Loss spike at epoch 12"},
        ],
        "alert_summary": {"total": 1, "critical": 0, "warning": 1, "info": 0},
        "training_stats": {
            "final_loss": 0.25,
            "min_loss": 0.23,
            "final_val_loss": 0.30,
            "final_gap": 0.05,
        },
    }
    text = format_analysis_report(report)
    assert "Training Log Analysis" in text
    assert "50" in text
    assert "Loss spike" in text


def test_format_analysis_report_error():
    """Error should show error message."""
    text = format_analysis_report({"error": "No metrics found"})
    assert "ERROR" in text


def test_format_analysis_report_no_alerts():
    """Clean training should say no issues."""
    report = {
        "log_path": "run.log",
        "analyzed_at": "2026-01-01T00:00:00",
        "total_epochs": 100,
        "alerts": [],
        "alert_summary": {"total": 0, "critical": 0, "warning": 0, "info": 0},
        "training_stats": {},
    }
    text = format_analysis_report(report)
    assert "No issues detected" in text


# --- analyze_training_log ---


def test_analyze_nonexistent_log(tmp_path):
    """Missing log should return error."""
    report = analyze_training_log(str(tmp_path / "missing.log"))
    assert "error" in report


def test_analyze_valid_log(tmp_path):
    """Should analyze a valid training log."""
    log_file = tmp_path / "run.log"
    lines = []
    for i in range(10):
        lines.append(json.dumps({"epoch": i, "loss": 1.0 - i * 0.08, "val_loss": 1.1 - i * 0.07}))
    log_file.write_text("\n".join(lines))

    report = analyze_training_log(str(log_file))
    assert "error" not in report
    assert report["total_epochs"] == 10
    assert report["training_stats"]["final_loss"] < 1.0


import json
