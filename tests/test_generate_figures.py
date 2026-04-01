"""Tests for figure generation (generate_figures.py). Phase 25.2."""
from __future__ import annotations
import pytest
from scripts.generate_figures import (
    generate_training_figure, generate_comparison_figure,
    generate_ablation_figure, format_figures_report,
)

EXPS = [
    {"experiment_id": f"exp-{i:03d}", "status": "kept", "metrics": {"accuracy": 0.70 + i * 0.02},
     "config": {"model_type": "xgboost" if i < 3 else "lightgbm"},
     "timestamp": f"2026-03-{i+1:02d}T00:00:00"}
    for i in range(6)
]
CONFIG = {"evaluation": {"primary_metric": "accuracy"}}
STYLE = {"palette": ["#1f77b4"], "font_size": 12, "line_width": 2}

def test_training_figure():
    fig = generate_training_figure(EXPS, CONFIG, STYLE)
    assert fig["type"] == "training"
    assert "data" in fig

def test_training_empty():
    fig = generate_training_figure([], CONFIG, STYLE)
    assert fig["type"] == "training"

def test_comparison_figure():
    fig = generate_comparison_figure(EXPS, CONFIG, STYLE)
    assert fig["type"] == "comparison"

def test_comparison_empty():
    fig = generate_comparison_figure([], CONFIG, STYLE)
    assert fig["type"] == "comparison"

def test_ablation_figure():
    data = [{"component": "dropout", "full_metric": 0.87, "ablated_metric": 0.85, "delta": -0.02}]
    fig = generate_ablation_figure(data, CONFIG, STYLE)
    assert fig["type"] == "ablation"

def test_format_report():
    figs = [{"type": "training", "title": "Curve", "data": []}]
    text = format_figures_report(figs)
    assert isinstance(text, str)

def test_format_empty():
    text = format_figures_report([])
    assert isinstance(text, str)
