"""Tests for research logbook generator (generate_logbook.py).

Verifies timeline building, trajectory computation, markdown and
HTML output generation, date filtering, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.generate_logbook import (
    build_timeline,
    compute_trajectory,
    generate_html,
    generate_markdown,
    load_experiments,
)


def _exp(exp_id, desc, status="kept", accuracy=0.85, ts="2026-03-15T10:00:00"):
    return {
        "experiment_id": exp_id,
        "description": desc,
        "status": status,
        "metrics": {"accuracy": accuracy},
        "config": {"model_type": "xgboost"},
        "timestamp": ts,
    }


def _hyp(hid, desc, source="human", status="queued", priority="high",
         ts="2026-03-15T09:00:00"):
    return {
        "id": hid,
        "description": desc,
        "source": source,
        "status": status,
        "priority": priority,
        "created_at": ts,
    }


EXPERIMENTS = [
    _exp("exp-001", "baseline xgboost", "kept", 0.82, "2026-03-15T10:00:00"),
    _exp("exp-002", "increase depth", "kept", 0.85, "2026-03-15T11:00:00"),
    _exp("exp-003", "try lightgbm", "discarded", 0.80, "2026-03-15T12:00:00"),
    _exp("exp-004", "add features", "kept", 0.87, "2026-03-16T10:00:00"),
]

HYPOTHESES = [
    _hyp("hyp-001", "try xgboost baseline", "human", "tested"),
    _hyp("hyp-002", "switch to lightgbm", "agent", "dead-end"),
    _hyp("hyp-003", "add polynomial features", "literature", "promising"),
]

CONFIG = {
    "evaluation": {"primary_metric": "accuracy", "lower_is_better": False},
    "task": {"description": "Predict customer churn"},
}


# --- build_timeline ---

def test_timeline_combines_events():
    events = build_timeline(EXPERIMENTS, HYPOTHESES)
    assert len(events) == len(EXPERIMENTS) + len(HYPOTHESES)


def test_timeline_sorted_by_timestamp():
    events = build_timeline(EXPERIMENTS, HYPOTHESES)
    timestamps = [e["timestamp"] for e in events if e["timestamp"]]
    assert timestamps == sorted(timestamps)


def test_timeline_empty():
    assert build_timeline([], []) == []


# --- compute_trajectory ---

def test_trajectory_tracks_best():
    traj = compute_trajectory(EXPERIMENTS, "accuracy")
    best_values = [t["best_so_far"] for t in traj]
    # Best should monotonically increase (higher is better)
    assert best_values == [0.82, 0.85, 0.85, 0.87]


def test_trajectory_marks_new_best():
    traj = compute_trajectory(EXPERIMENTS, "accuracy")
    new_bests = [t["is_new_best"] for t in traj]
    assert new_bests == [True, True, False, True]


def test_trajectory_lower_is_better():
    exps = [
        _exp("e1", "a", "kept", 0.5),
        _exp("e2", "b", "kept", 0.3),
        _exp("e3", "c", "kept", 0.4),
    ]
    traj = compute_trajectory(exps, "accuracy", lower_is_better=True)
    assert traj[0]["best_so_far"] == 0.5
    assert traj[1]["best_so_far"] == 0.3
    assert traj[1]["is_new_best"] is True
    assert traj[2]["is_new_best"] is False


def test_trajectory_empty():
    assert compute_trajectory([], "accuracy") == []


def test_trajectory_missing_metric():
    exps = [{"experiment_id": "e1", "metrics": {}, "status": "kept"}]
    assert compute_trajectory(exps, "accuracy") == []


# --- generate_markdown ---

def test_markdown_contains_sections():
    traj = compute_trajectory(EXPERIMENTS, "accuracy")
    md = generate_markdown(CONFIG, EXPERIMENTS, HYPOTHESES, traj)
    assert "# Research Logbook" in md
    assert "## Campaign Summary" in md
    assert "## Best Result" in md
    assert "## Improvement Trajectory" in md
    assert "## Experiment Log" in md
    assert "## Hypothesis Queue" in md


def test_markdown_contains_data():
    traj = compute_trajectory(EXPERIMENTS, "accuracy")
    md = generate_markdown(CONFIG, EXPERIMENTS, HYPOTHESES, traj)
    assert "exp-001" in md
    assert "0.87" in md
    assert "hyp-001" in md
    assert "customer churn" in md.lower() or "Predict" in md


def test_markdown_keep_rate():
    traj = compute_trajectory(EXPERIMENTS, "accuracy")
    md = generate_markdown(CONFIG, EXPERIMENTS, HYPOTHESES, traj)
    # 3 kept out of 4 = 75%
    assert "75%" in md


def test_markdown_empty_experiments():
    md = generate_markdown(CONFIG, [], [], [])
    assert "Research Logbook" in md
    assert "Total experiments" in md


# --- generate_html ---

def test_html_self_contained():
    traj = compute_trajectory(EXPERIMENTS, "accuracy")
    html = generate_html(CONFIG, EXPERIMENTS, HYPOTHESES, traj)
    assert "<!DOCTYPE html>" in html
    assert "chart.js" in html.lower() or "Chart" in html
    assert "</html>" in html


def test_html_contains_data():
    traj = compute_trajectory(EXPERIMENTS, "accuracy")
    html = generate_html(CONFIG, EXPERIMENTS, HYPOTHESES, traj)
    assert "exp-001" in html
    assert "0.87" in html
    assert "accuracy" in html


def test_html_contains_trajectory_data():
    traj = compute_trajectory(EXPERIMENTS, "accuracy")
    html = generate_html(CONFIG, EXPERIMENTS, HYPOTHESES, traj)
    # Chart data should be embedded as JSON
    assert "0.82" in html
    assert "0.85" in html


def test_html_has_styles():
    traj = compute_trajectory(EXPERIMENTS, "accuracy")
    html = generate_html(CONFIG, EXPERIMENTS, HYPOTHESES, traj)
    assert "<style>" in html
    assert "kept" in html
    assert "discarded" in html


def test_html_empty_experiments():
    html = generate_html(CONFIG, [], [], [])
    assert "<!DOCTYPE html>" in html
    assert "No hypotheses yet" in html


# --- load_experiments ---

def test_load_experiments_from_file(tmp_path):
    log = tmp_path / "log.jsonl"
    with open(log, "w") as f:
        for exp in EXPERIMENTS:
            f.write(json.dumps(exp) + "\n")
    loaded = load_experiments(str(log))
    assert len(loaded) == 4


def test_load_experiments_missing_file():
    assert load_experiments("/nonexistent/log.jsonl") == []
