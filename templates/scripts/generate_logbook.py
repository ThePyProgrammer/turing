#!/usr/bin/env python3
"""Research logbook generator for the autoresearch pipeline.

Generates a self-contained HTML logbook showing the full research
narrative: hypotheses proposed, experiments run, decisions made,
and progress over time. Designed to be shared with collaborators
or archived as a research artifact.

Usage:
    python scripts/generate_logbook.py
    python scripts/generate_logbook.py --output logbook.html
    python scripts/generate_logbook.py --since 2026-03-01
    python scripts/generate_logbook.py --format markdown --output logbook.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments, load_hypotheses


def load_decisions(decisions_dir: str) -> list[dict]:
    """Load decision packets."""
    path = Path(decisions_dir)
    if not path.exists():
        return []
    decisions = []
    for f in sorted(path.glob("*.json")):
        try:
            with open(f) as fh:
                decisions.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return decisions


def build_timeline(experiments: list[dict], hypotheses: list[dict]) -> list[dict]:
    """Build a unified timeline of events from experiments and hypotheses."""
    events = []

    for exp in experiments:
        ts = exp.get("timestamp", "")
        events.append({
            "timestamp": ts,
            "type": "experiment",
            "id": exp.get("experiment_id", "?"),
            "description": exp.get("description", ""),
            "status": exp.get("status", "unknown"),
            "metrics": exp.get("metrics", {}),
            "config": exp.get("config", {}),
        })

    for hyp in hypotheses:
        ts = hyp.get("created_at", "")
        events.append({
            "timestamp": ts,
            "type": "hypothesis",
            "id": hyp.get("id", "?"),
            "description": hyp.get("description", ""),
            "status": hyp.get("status", "queued"),
            "source": hyp.get("source", "unknown"),
            "priority": hyp.get("priority", "medium"),
        })

    events.sort(key=lambda e: e.get("timestamp", ""))
    return events


def compute_trajectory(experiments: list[dict], metric_name: str,
                       lower_is_better: bool = False) -> list[dict]:
    """Compute the improvement trajectory over time."""
    trajectory = []
    best_val = None

    for exp in experiments:
        val = exp.get("metrics", {}).get(metric_name)
        if val is None or not isinstance(val, (int, float)):
            continue

        is_new_best = False
        if best_val is None:
            best_val = val
            is_new_best = True
        elif lower_is_better and val < best_val:
            best_val = val
            is_new_best = True
        elif not lower_is_better and val > best_val:
            best_val = val
            is_new_best = True

        trajectory.append({
            "experiment_id": exp.get("experiment_id", "?"),
            "timestamp": exp.get("timestamp", ""),
            "value": val,
            "best_so_far": best_val,
            "is_new_best": is_new_best,
            "status": exp.get("status", "unknown"),
        })

    return trajectory


def generate_markdown(config: dict, experiments: list[dict],
                      hypotheses: list[dict], trajectory: list[dict],
                      since: str | None = None) -> str:
    """Generate a markdown logbook."""
    metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")
    task_desc = config.get("task", {}).get("description", "ML experiment campaign")
    lower_is_better = config.get("evaluation", {}).get("lower_is_better", False)

    lines = []
    lines.append(f"# Research Logbook: {task_desc}")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    if since:
        lines.append(f"Period: {since} to present")
    lines.append("")

    # Campaign summary
    kept = [e for e in experiments if e.get("status") == "kept"]
    discarded = [e for e in experiments if e.get("status") == "discarded"]
    lines.append("## Campaign Summary")
    lines.append("")
    lines.append(f"- **Total experiments:** {len(experiments)}")
    lines.append(f"- **Kept:** {len(kept)} ({len(kept)/len(experiments)*100:.0f}%)" if experiments else "- **Kept:** 0")
    lines.append(f"- **Discarded:** {len(discarded)}")
    lines.append(f"- **Hypotheses proposed:** {len(hypotheses)}")
    human_hyps = [h for h in hypotheses if h.get("source") == "human"]
    lines.append(f"- **Human-injected:** {len(human_hyps)}")
    lines.append("")

    # Best result
    if trajectory:
        best = max(trajectory, key=lambda t: -t["best_so_far"] if lower_is_better else t["best_so_far"])
        lines.append("## Best Result")
        lines.append("")
        lines.append(f"- **{metric_name}:** {best['best_so_far']}")
        lines.append(f"- **Experiment:** {best['experiment_id']}")
        lines.append("")

    # Improvement trajectory
    if trajectory:
        lines.append("## Improvement Trajectory")
        lines.append("")
        lines.append(f"| # | Experiment | {metric_name} | Best | New Best? |")
        lines.append("|---|-----------|--------|------|-----------|")
        for i, t in enumerate(trajectory, 1):
            marker = "**yes**" if t["is_new_best"] else ""
            lines.append(f"| {i} | {t['experiment_id']} | {t['value']:.4f} | {t['best_so_far']:.4f} | {marker} |")
        lines.append("")

    # Experiment log
    lines.append("## Experiment Log")
    lines.append("")
    for exp in experiments:
        status_icon = "kept" if exp.get("status") == "kept" else "discarded"
        exp_id = exp.get("experiment_id", "?")
        desc = exp.get("description", "No description")
        metrics = exp.get("metrics", {})
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                              if isinstance(v, (int, float)))

        lines.append(f"### {exp_id} [{status_icon}]")
        lines.append("")
        lines.append(f"**Description:** {desc}")
        lines.append(f"**Metrics:** {metric_str}")
        ts = exp.get("timestamp", "")
        if ts:
            lines.append(f"**Time:** {ts[:19]}")
        lines.append("")

    # Hypothesis log
    if hypotheses:
        lines.append("## Hypothesis Queue")
        lines.append("")
        lines.append("| ID | Description | Source | Status | Priority |")
        lines.append("|---|-----------|--------|--------|----------|")
        for h in hypotheses:
            lines.append(
                f"| {h.get('id', '?')} "
                f"| {h.get('description', '')[:60]} "
                f"| {h.get('source', '?')} "
                f"| {h.get('status', '?')} "
                f"| {h.get('priority', '?')} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_html(config: dict, experiments: list[dict],
                  hypotheses: list[dict], trajectory: list[dict],
                  since: str | None = None) -> str:
    """Generate a self-contained HTML logbook."""
    metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")
    task_desc = config.get("task", {}).get("description", "ML experiment campaign")
    lower_is_better = config.get("evaluation", {}).get("lower_is_better", False)

    kept = [e for e in experiments if e.get("status") == "kept"]
    discarded = [e for e in experiments if e.get("status") == "discarded"]
    keep_rate = f"{len(kept)/len(experiments)*100:.0f}" if experiments else "0"

    best_metric = ""
    best_exp = ""
    if trajectory:
        best = max(trajectory, key=lambda t: -t["best_so_far"] if lower_is_better else t["best_so_far"])
        best_metric = f"{best['best_so_far']:.4f}"
        best_exp = best["experiment_id"]

    # Build trajectory data for the chart
    traj_labels = json.dumps([t["experiment_id"] for t in trajectory])
    traj_values = json.dumps([round(t["value"], 4) for t in trajectory])
    traj_best = json.dumps([round(t["best_so_far"], 4) for t in trajectory])

    # Build experiment rows
    exp_rows = ""
    for exp in experiments:
        status = exp.get("status", "unknown")
        cls = "kept" if status == "kept" else "discarded"
        metrics = exp.get("metrics", {})
        metric_val = metrics.get(metric_name, "—")
        if isinstance(metric_val, float):
            metric_val = f"{metric_val:.4f}"
        exp_rows += f"""
        <tr class="{cls}">
            <td><code>{exp.get('experiment_id', '?')}</code></td>
            <td>{exp.get('description', '')}</td>
            <td>{metric_val}</td>
            <td><span class="badge {cls}">{status}</span></td>
            <td>{exp.get('timestamp', '')[:10]}</td>
        </tr>"""

    # Build hypothesis rows
    hyp_rows = ""
    for h in hypotheses:
        status = h.get("status", "queued")
        hyp_rows += f"""
        <tr>
            <td><code>{h.get('id', '?')}</code></td>
            <td>{h.get('description', '')}</td>
            <td><span class="badge source-{h.get('source', 'unknown')}">{h.get('source', '?')}</span></td>
            <td><span class="badge status-{status}">{status}</span></td>
            <td>{h.get('priority', '?')}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Logbook — {task_desc}</title>
<style>
  :root {{ --blue: #2563eb; --green: #16a34a; --red: #dc2626; --gray: #6b7280;
           --bg: #f8fafc; --card: #fff; --border: #e2e8f0; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: #1e293b; line-height: 1.6; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem; }}
  header {{ background: linear-gradient(135deg, #1e40af, #3b82f6); color: #fff;
           padding: 2rem; border-radius: 12px; margin-bottom: 2rem; }}
  header h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  header p {{ opacity: 0.85; font-size: 0.95rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
           gap: 1rem; margin-bottom: 2rem; }}
  .stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 1.2rem; text-align: center; }}
  .stat .value {{ font-size: 1.8rem; font-weight: 700; color: var(--blue); }}
  .stat .label {{ font-size: 0.85rem; color: var(--gray); margin-top: 0.25rem; }}
  .section {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
             padding: 1.5rem; margin-bottom: 1.5rem; }}
  .section h2 {{ font-size: 1.2rem; margin-bottom: 1rem; color: #1e293b;
                border-bottom: 2px solid var(--blue); padding-bottom: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ text-align: left; padding: 0.6rem; background: #f1f5f9; border-bottom: 2px solid var(--border); }}
  td {{ padding: 0.6rem; border-bottom: 1px solid var(--border); }}
  tr.kept td:first-child {{ border-left: 3px solid var(--green); }}
  tr.discarded td:first-child {{ border-left: 3px solid var(--red); }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
           font-size: 0.75rem; font-weight: 600; }}
  .badge.kept {{ background: #dcfce7; color: #166534; }}
  .badge.discarded {{ background: #fee2e2; color: #991b1b; }}
  .badge.source-human {{ background: #dbeafe; color: #1e40af; }}
  .badge.source-agent {{ background: #f3e8ff; color: #6b21a8; }}
  .badge.source-literature {{ background: #fef3c7; color: #92400e; }}
  .badge.status-queued {{ background: #e0e7ff; color: #3730a3; }}
  .badge.status-tested {{ background: #d1fae5; color: #065f46; }}
  .badge.status-dead-end {{ background: #fee2e2; color: #991b1b; }}
  .badge.status-promising {{ background: #dcfce7; color: #166534; }}
  .badge.status-in-progress {{ background: #fef3c7; color: #92400e; }}
  .chart {{ width: 100%; height: 250px; position: relative; }}
  canvas {{ width: 100% !important; height: 100% !important; }}
  code {{ background: #f1f5f9; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.85em; }}
  .footer {{ text-align: center; color: var(--gray); font-size: 0.8rem; margin-top: 2rem; }}
  @media print {{
    body {{ background: #fff; }}
    .container {{ max-width: none; padding: 1rem; }}
    header {{ break-inside: avoid; }}
    .section {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>{task_desc}</h1>
    <p>Research Logbook &mdash; Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
    {f' &mdash; Since {since}' if since else ''}</p>
  </header>

  <div class="stats">
    <div class="stat"><div class="value">{len(experiments)}</div><div class="label">Experiments</div></div>
    <div class="stat"><div class="value">{keep_rate}%</div><div class="label">Keep Rate</div></div>
    <div class="stat"><div class="value">{best_metric}</div><div class="label">Best {metric_name}</div></div>
    <div class="stat"><div class="value">{best_exp}</div><div class="label">Best Experiment</div></div>
    <div class="stat"><div class="value">{len(hypotheses)}</div><div class="label">Hypotheses</div></div>
  </div>

  <div class="section">
    <h2>Improvement Trajectory</h2>
    <div class="chart"><canvas id="trajChart"></canvas></div>
  </div>

  <div class="section">
    <h2>Experiment Log</h2>
    <table>
      <thead><tr><th>ID</th><th>Description</th><th>{metric_name}</th><th>Status</th><th>Date</th></tr></thead>
      <tbody>{exp_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>Hypothesis Queue</h2>
    <table>
      <thead><tr><th>ID</th><th>Description</th><th>Source</th><th>Status</th><th>Priority</th></tr></thead>
      <tbody>{hyp_rows if hyp_rows else '<tr><td colspan="5" style="text-align:center;color:var(--gray)">No hypotheses yet</td></tr>'}</tbody>
    </table>
  </div>

  <div class="footer">
    <p>Generated by Turing &mdash; Autonomous ML Research Harness</p>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script>
const labels = {traj_labels};
const values = {traj_values};
const best = {traj_best};
if (labels.length > 0) {{
  new Chart(document.getElementById('trajChart'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: '{metric_name}', data: values, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)',
           fill: true, tension: 0.3, pointRadius: 4 }},
        {{ label: 'Best so far', data: best, borderColor: '#16a34a', borderDash: [5,5],
           fill: false, tension: 0, pointRadius: 0 }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{ y: {{ title: {{ display: true, text: '{metric_name}' }} }} }}
    }}
  }});
}}
</script>
</body>
</html>"""

    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research logbook")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log", default="experiments/log.jsonl")
    parser.add_argument("--hypotheses", default="hypotheses.yaml")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--format", choices=["html", "markdown"], default="html")
    parser.add_argument("--since", default=None, help="Filter events after this date (YYYY-MM-DD)")
    args = parser.parse_args()

    config = load_config(args.config)
    experiments = load_experiments(args.log)
    hypotheses = load_hypotheses(args.hypotheses)

    if args.since:
        experiments = [e for e in experiments
                      if e.get("timestamp", "") >= args.since]
        hypotheses = [h for h in hypotheses
                     if h.get("created_at", "") >= args.since]

    metric_name = config.get("evaluation", {}).get("primary_metric", "accuracy")
    lower_is_better = config.get("evaluation", {}).get("lower_is_better", False)
    trajectory = compute_trajectory(experiments, metric_name, lower_is_better)

    if args.format == "markdown":
        output = generate_markdown(config, experiments, hypotheses, trajectory, args.since)
        default_name = "logbook.md"
    else:
        output = generate_html(config, experiments, hypotheses, trajectory, args.since)
        default_name = "logbook.html"

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Logbook written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
