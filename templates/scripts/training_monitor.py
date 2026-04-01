#!/usr/bin/env python3
"""Live training monitor for the autoresearch pipeline.

Streams metrics during a training run with early-warning alerts:
loss spikes, gradient explosion, learning rate too aggressive,
train/val gap widening. Catches problems early instead of at the end.

Can tail a run.log file or read completed logs for post-hoc analysis.

Usage:
    python scripts/training_monitor.py                    # Monitor run.log
    python scripts/training_monitor.py --log custom.log   # Custom log file
    python scripts/training_monitor.py --alerts            # Show only alerts
    python scripts/training_monitor.py --interval 5       # Check every 5 seconds
    python scripts/training_monitor.py --analyze run.log   # Post-hoc analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config

DEFAULT_LOG_PATH = "run.log"
DEFAULT_ALERT_CONFIG = "config/watch_alerts.yaml"
DEFAULT_INTERVAL = 10  # seconds
ROLLING_WINDOW = 10  # epochs for rolling statistics


# --- Metric Parsing ---


def parse_epoch_metrics(line: str) -> dict | None:
    """Parse a single log line into epoch metrics.

    Supports formats:
        JSON: {"epoch": 1, "loss": 0.5, "val_loss": 0.6, ...}
        KV:   epoch=1 loss=0.5 val_loss=0.6
        CSV:  epoch,loss,val_loss\\n1,0.5,0.6
    """
    line = line.strip()
    if not line:
        return None

    # JSON format
    if line.startswith("{"):
        try:
            data = json.loads(line)
            if "epoch" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Key=value format
    if "epoch=" in line or "epoch:" in line:
        metrics = {}
        # Handle both = and : separators
        parts = line.replace(":", "=").split()
        for part in parts:
            if "=" in part:
                key, _, val = part.partition("=")
                key = key.strip()
                val = val.strip()
                try:
                    metrics[key] = int(val) if key == "epoch" else float(val)
                except ValueError:
                    metrics[key] = val
        if "epoch" in metrics:
            return metrics

    return None


def parse_log_file(log_path: str) -> list[dict]:
    """Parse all epoch metrics from a log file."""
    path = Path(log_path)
    if not path.exists():
        return []

    metrics = []
    with open(path) as f:
        for line in f:
            parsed = parse_epoch_metrics(line)
            if parsed is not None:
                metrics.append(parsed)
    return metrics


# --- Rolling Statistics ---


def compute_rolling_stats(
    history: list[dict],
    metric: str,
    window: int = ROLLING_WINDOW,
) -> dict:
    """Compute rolling statistics for a metric.

    Returns:
        Dict with mean, std, trend (slope), min, max over the window.
    """
    values = [h.get(metric) for h in history[-window:] if h.get(metric) is not None]
    if not values:
        return {}

    n = len(values)
    mean = sum(values) / n

    if n >= 2:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)

        # Simple linear trend (slope)
        x_mean = (n - 1) / 2
        numerator = sum((i - x_mean) * (v - mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        trend = numerator / denominator if denominator > 0 else 0.0
    else:
        std = 0.0
        trend = 0.0

    return {
        "mean": mean,
        "std": std,
        "trend": trend,
        "min": min(values),
        "max": max(values),
        "n": n,
    }


# --- Alert Rules ---


def load_alert_config(config_path: str = DEFAULT_ALERT_CONFIG) -> dict:
    """Load alert configuration from YAML."""
    path = Path(config_path)
    if not path.exists():
        return default_alert_config()

    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else default_alert_config()


def default_alert_config() -> dict:
    """Return default alert rules."""
    return {
        "alerts": {
            "loss_spike": {
                "condition": "loss_spike",
                "multiplier": 3.0,
                "severity": "warning",
                "message": "Loss spike at epoch {epoch}: {value} vs rolling mean {mean:.4f}",
            },
            "nan_detected": {
                "condition": "nan_detected",
                "severity": "critical",
                "action": "pause",
                "message": "NaN detected in {metric} at epoch {epoch}",
            },
            "overfitting_onset": {
                "condition": "overfitting",
                "gap_ratio": 0.5,
                "consecutive": 3,
                "severity": "warning",
                "message": "Overfitting detected — train/val gap widening since epoch {onset}",
            },
            "plateau": {
                "condition": "plateau",
                "min_improvement": 0.001,
                "consecutive": 5,
                "severity": "info",
                "message": "Metric plateaued — consider early stopping or LR reduction",
            },
        },
    }


def evaluate_alerts(
    current: dict,
    history: list[dict],
    alert_config: dict,
) -> list[dict]:
    """Evaluate all alert rules against current state.

    Args:
        current: Current epoch metrics.
        history: All previous epoch metrics.
        alert_config: Alert configuration dict.

    Returns:
        List of triggered alert dicts.
    """
    alerts_config = alert_config.get("alerts", {})
    triggered = []

    for name, rule in alerts_config.items():
        condition = rule.get("condition", name)

        if condition == "loss_spike":
            alert = _check_loss_spike(current, history, rule)
        elif condition == "nan_detected":
            alert = _check_nan(current, rule)
        elif condition == "overfitting":
            alert = _check_overfitting(current, history, rule)
        elif condition == "plateau":
            alert = _check_plateau(history, rule)
        else:
            continue

        if alert:
            alert["name"] = name
            alert["severity"] = rule.get("severity", "info")
            alert["action"] = rule.get("action")
            triggered.append(alert)

    return triggered


def _check_loss_spike(current: dict, history: list[dict], rule: dict) -> dict | None:
    """Check for sudden loss spikes."""
    loss = current.get("loss") or current.get("train_loss")
    if loss is None:
        return None

    rolling = compute_rolling_stats(history, "loss")
    if not rolling or rolling["n"] < 3:
        # Also try train_loss
        rolling = compute_rolling_stats(history, "train_loss")
        if not rolling or rolling["n"] < 3:
            return None

    multiplier = rule.get("multiplier", 3.0)
    mean = rolling["mean"]

    if mean > 0 and loss > multiplier * mean:
        msg = rule.get("message", "Loss spike detected").format(
            epoch=current.get("epoch", "?"),
            value=loss,
            mean=mean,
        )
        return {"message": msg, "epoch": current.get("epoch"), "value": loss, "mean": mean}

    return None


def _check_nan(current: dict, rule: dict) -> dict | None:
    """Check for NaN values in any metric."""
    for key, val in current.items():
        if key == "epoch":
            continue
        if isinstance(val, float) and math.isnan(val):
            msg = rule.get("message", "NaN detected").format(
                metric=key,
                epoch=current.get("epoch", "?"),
            )
            return {"message": msg, "epoch": current.get("epoch"), "metric": key}
    return None


def _check_overfitting(current: dict, history: list[dict], rule: dict) -> dict | None:
    """Check for train/val gap widening."""
    gap_ratio = rule.get("gap_ratio", 0.5)
    consecutive_required = rule.get("consecutive", 3)

    # Compute train/val gap over recent history
    gaps = []
    for entry in history + [current]:
        train_loss = entry.get("loss") or entry.get("train_loss")
        val_loss = entry.get("val_loss")
        if train_loss is not None and val_loss is not None:
            gaps.append({
                "epoch": entry.get("epoch"),
                "gap": val_loss - train_loss,
                "ratio": train_loss / val_loss if val_loss != 0 else 0,
            })

    if len(gaps) < consecutive_required + 1:
        return None

    # Check if gap is widening for N consecutive epochs
    recent = gaps[-consecutive_required:]
    widening = all(
        recent[i]["gap"] > recent[i - 1]["gap"]
        for i in range(1, len(recent))
    )

    if widening and recent[-1]["ratio"] < gap_ratio:
        onset = recent[0]["epoch"]
        msg = rule.get("message", "Overfitting detected").format(
            onset=onset,
        )
        return {"message": msg, "onset": onset, "current_gap": recent[-1]["gap"]}

    return None


def _check_plateau(history: list[dict], rule: dict) -> dict | None:
    """Check for metric plateau."""
    min_improvement = rule.get("min_improvement", 0.001)
    consecutive_required = rule.get("consecutive", 5)

    if len(history) < consecutive_required:
        return None

    # Check val_loss or accuracy for plateau
    for metric in ("val_loss", "val_accuracy", "accuracy", "loss"):
        values = [h.get(metric) for h in history[-consecutive_required:] if h.get(metric) is not None]
        if len(values) < consecutive_required:
            continue

        improvements = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        if all(imp < min_improvement for imp in improvements):
            msg = rule.get("message", "Metric plateaued")
            return {"message": msg, "metric": metric, "n_flat_epochs": len(values)}

    return None


# --- Dashboard Formatting ---


def format_dashboard_line(current: dict, rolling_loss: dict, alerts: list[dict]) -> str:
    """Format a compact single-line dashboard.

    Example: Epoch 23/100 | loss: 0.342 ↓ | acc: 0.865 ↑ | gap: 0.018 | ⚠ plateau (5 epochs)
    """
    epoch = current.get("epoch", "?")
    total_epochs = current.get("total_epochs") or current.get("n_epochs", "?")

    parts = [f"Epoch {epoch}/{total_epochs}"]

    # Loss with trend arrow
    loss = current.get("loss") or current.get("train_loss")
    if loss is not None:
        arrow = ""
        if rolling_loss and rolling_loss.get("trend") is not None:
            arrow = " ↓" if rolling_loss["trend"] < 0 else " ↑" if rolling_loss["trend"] > 0 else ""
        if math.isnan(loss):
            parts.append("loss: NaN")
        else:
            parts.append(f"loss: {loss:.4f}{arrow}")

    # Accuracy/val metric
    for metric in ("accuracy", "val_accuracy", "val_loss"):
        val = current.get(metric)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            parts.append(f"{metric}: {val:.4f}")

    # Train/val gap
    train_loss = current.get("loss") or current.get("train_loss")
    val_loss = current.get("val_loss")
    if train_loss is not None and val_loss is not None:
        if not (math.isnan(train_loss) or math.isnan(val_loss)):
            gap = val_loss - train_loss
            parts.append(f"gap: {gap:.4f}")

    # Alert indicators
    for alert in alerts:
        severity = alert.get("severity", "info")
        name = alert.get("name", "alert")
        if severity == "critical":
            parts.append(f"CRITICAL: {name}")
        elif severity == "warning":
            parts.append(f"WARNING: {name}")
        else:
            parts.append(f"info: {name}")

    return " | ".join(parts)


# --- Analysis Report ---


def analyze_training_log(
    log_path: str,
    alert_config_path: str = DEFAULT_ALERT_CONFIG,
    config_path: str = "config.yaml",
) -> dict:
    """Analyze a completed training log for issues.

    Returns a structured report with all alerts that would have
    been triggered during training.
    """
    metrics = parse_log_file(log_path)
    if not metrics:
        return {"error": f"No metrics found in {log_path}", "log_path": log_path}

    alert_config = load_alert_config(alert_config_path)

    all_alerts = []
    for i, current in enumerate(metrics):
        history = metrics[:i]
        alerts = evaluate_alerts(current, history, alert_config)
        all_alerts.extend(alerts)

    # Compute overall statistics
    loss_values = [m.get("loss") or m.get("train_loss") for m in metrics]
    loss_values = [v for v in loss_values if v is not None and not math.isnan(v)]

    val_loss_values = [m.get("val_loss") for m in metrics]
    val_loss_values = [v for v in val_loss_values if v is not None and not math.isnan(v)]

    report = {
        "log_path": log_path,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "total_epochs": len(metrics),
        "alerts": all_alerts,
        "alert_summary": {
            "total": len(all_alerts),
            "critical": len([a for a in all_alerts if a.get("severity") == "critical"]),
            "warning": len([a for a in all_alerts if a.get("severity") == "warning"]),
            "info": len([a for a in all_alerts if a.get("severity") == "info"]),
        },
        "training_stats": {},
    }

    if loss_values:
        report["training_stats"]["final_loss"] = loss_values[-1]
        report["training_stats"]["min_loss"] = min(loss_values)
        report["training_stats"]["loss_reduction"] = loss_values[0] - loss_values[-1] if len(loss_values) > 1 else 0

    if val_loss_values:
        report["training_stats"]["final_val_loss"] = val_loss_values[-1]
        report["training_stats"]["min_val_loss"] = min(val_loss_values)

    if loss_values and val_loss_values:
        report["training_stats"]["final_gap"] = val_loss_values[-1] - loss_values[-1]

    return report


def format_analysis_report(report: dict) -> str:
    """Format analysis report as markdown."""
    if "error" in report:
        return f"ERROR: {report['error']}"

    lines = [
        "# Training Log Analysis",
        "",
        f"*Analyzed {report.get('analyzed_at', 'N/A')[:19]}*",
        f"*Log: {report.get('log_path', 'N/A')}*",
        "",
        f"## Summary",
        "",
        f"- **Total epochs:** {report.get('total_epochs', 0)}",
    ]

    stats = report.get("training_stats", {})
    if stats.get("final_loss") is not None:
        lines.append(f"- **Final loss:** {stats['final_loss']:.4f} (min: {stats.get('min_loss', 0):.4f})")
    if stats.get("final_val_loss") is not None:
        lines.append(f"- **Final val_loss:** {stats['final_val_loss']:.4f}")
    if stats.get("final_gap") is not None:
        lines.append(f"- **Train/val gap:** {stats['final_gap']:.4f}")

    summary = report.get("alert_summary", {})
    lines.extend([
        "",
        "## Alerts",
        "",
        f"- **Critical:** {summary.get('critical', 0)}",
        f"- **Warning:** {summary.get('warning', 0)}",
        f"- **Info:** {summary.get('info', 0)}",
    ])

    alerts = report.get("alerts", [])
    if alerts:
        lines.extend(["", "### Details", ""])
        for alert in alerts:
            sev = alert.get("severity", "info").upper()
            lines.append(f"- **[{sev}]** {alert.get('message', 'N/A')}")
    else:
        lines.extend(["", "No issues detected during training."])

    return "\n".join(lines)


def save_analysis_report(report: dict, output_dir: str = "experiments/monitors") -> Path:
    """Save analysis report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filepath = out_path / f"analysis-{timestamp}.yaml"

    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)

    return filepath


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Live training monitor with early-warning alerts",
    )
    parser.add_argument(
        "--log", default=DEFAULT_LOG_PATH,
        help=f"Path to training log file (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--alerts-config", default=DEFAULT_ALERT_CONFIG,
        help=f"Path to alert config YAML (default: {DEFAULT_ALERT_CONFIG})",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"Check interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--alerts", action="store_true",
        help="Show only alerts, suppress normal output",
    )
    parser.add_argument(
        "--analyze", metavar="LOG_FILE",
        help="Post-hoc analysis of a completed training log",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    if args.analyze:
        # Post-hoc analysis mode
        report = analyze_training_log(
            args.analyze,
            alert_config_path=args.alerts_config,
            config_path=args.config,
        )

        if "error" not in report:
            filepath = save_analysis_report(report)
            print(f"Saved to {filepath}", file=sys.stderr)

        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(format_analysis_report(report))

        if report.get("alert_summary", {}).get("critical", 0) > 0:
            sys.exit(1)
        return

    # Live monitoring mode
    log_path = Path(args.log)
    alert_config = load_alert_config(args.alerts_config)

    print(f"Monitoring {log_path} (interval: {args.interval}s)", file=sys.stderr)
    print("Press Ctrl+C to stop.", file=sys.stderr)
    print(file=sys.stderr)

    history: list[dict] = []
    last_line_count = 0

    try:
        while True:
            if not log_path.exists():
                time.sleep(args.interval)
                continue

            with open(log_path) as f:
                lines = f.readlines()

            new_lines = lines[last_line_count:]
            last_line_count = len(lines)

            for line in new_lines:
                parsed = parse_epoch_metrics(line)
                if parsed is None:
                    continue

                alerts = evaluate_alerts(parsed, history, alert_config)
                rolling = compute_rolling_stats(history, "loss")

                if not args.alerts or alerts:
                    dashboard = format_dashboard_line(parsed, rolling, alerts)
                    print(dashboard)

                # Handle critical alerts with pause action
                for alert in alerts:
                    if alert.get("action") == "pause":
                        print(f"\nCRITICAL: {alert['message']}", file=sys.stderr)
                        print("Training should be paused.", file=sys.stderr)

                history.append(parsed)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nMonitoring stopped. {len(history)} epochs observed.", file=sys.stderr)


if __name__ == "__main__":
    main()
