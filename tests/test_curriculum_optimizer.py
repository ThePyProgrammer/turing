"""Tests for training curriculum optimization (curriculum_optimizer.py).

Phase 22.2: Verifies difficulty scoring, curriculum strategies, impossible
sample detection, strategy comparison, and reporting.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.curriculum_optimizer import (
    apply_curriculum,
    compare_strategies,
    curriculum_analysis,
    detect_impossible_samples,
    format_curriculum_report,
    score_difficulty_by_disagreement,
    score_difficulty_by_loss,
    score_difficulty_by_margin,
)


# --- score_difficulty_by_loss ---

def test_loss_scoring():
    losses = np.array([0.1, 0.5, 0.9, 0.3])
    scores = score_difficulty_by_loss(losses)
    assert scores[2] == 1.0  # Highest loss = hardest
    assert scores[0] == 0.0  # Lowest loss = easiest

def test_loss_constant():
    scores = score_difficulty_by_loss(np.array([0.5, 0.5, 0.5]))
    assert np.all(scores == 0)

def test_loss_empty():
    assert len(score_difficulty_by_loss(np.array([]))) == 0

# --- score_difficulty_by_margin ---

def test_margin_scoring():
    margins = np.array([0.1, 0.5, 0.9])
    scores = score_difficulty_by_margin(margins)
    assert scores[0] > scores[2]  # Small margin = harder

def test_margin_constant():
    scores = score_difficulty_by_margin(np.array([0.5, 0.5]))
    assert np.all(scores == 0.5)

def test_margin_empty():
    assert len(score_difficulty_by_margin(np.array([]))) == 0

# --- score_difficulty_by_disagreement ---

def test_disagreement_scoring():
    labels = np.array([0, 1, 0, 1])
    preds = [
        np.array([0, 1, 0, 0]),  # Seed 1: wrong on sample 3
        np.array([0, 1, 1, 1]),  # Seed 2: wrong on sample 2
        np.array([0, 1, 0, 1]),  # Seed 3: all correct
    ]
    scores = score_difficulty_by_disagreement(preds, labels)
    assert scores[0] == 0.0  # Always correct = easy
    assert scores[2] > 0  # Some disagreement

def test_disagreement_empty():
    assert len(score_difficulty_by_disagreement([], np.array([]))) == 0

# --- apply_curriculum ---

def test_curriculum_easy_to_hard():
    indices = np.array([0, 1, 2, 3])
    difficulties = np.array([0.8, 0.2, 0.5, 0.1])
    result = apply_curriculum(indices, difficulties, "easy_to_hard")
    assert result[0] == 3  # Easiest first
    assert result[-1] == 0  # Hardest last

def test_curriculum_hard_to_easy():
    indices = np.array([0, 1, 2, 3])
    difficulties = np.array([0.8, 0.2, 0.5, 0.1])
    result = apply_curriculum(indices, difficulties, "hard_to_easy")
    assert result[0] == 0  # Hardest first

def test_curriculum_random():
    indices = np.arange(100)
    difficulties = np.random.rand(100)
    result = apply_curriculum(indices.copy(), difficulties, "random")
    assert len(result) == 100  # Same length

def test_curriculum_self_paced():
    indices = np.arange(20)
    difficulties = np.linspace(0, 1, 20)
    result = apply_curriculum(indices, difficulties, "self_paced")
    assert len(result) == 20
    # First band should be the easiest samples
    assert np.mean(difficulties[result[:4]]) < np.mean(difficulties[result[-4:]])

def test_curriculum_empty():
    result = apply_curriculum(np.array([]), np.array([]), "easy_to_hard")
    assert len(result) == 0

# --- detect_impossible_samples ---

def test_impossible_detected():
    difficulties = np.array([0.1, 0.3, 0.95, 0.5, 0.92])
    impossible = detect_impossible_samples(difficulties, threshold=0.9)
    assert 2 in impossible
    assert 4 in impossible
    assert 0 not in impossible

def test_impossible_none():
    difficulties = np.array([0.1, 0.3, 0.5])
    assert detect_impossible_samples(difficulties) == []

# --- compare_strategies ---

def test_compare_basic():
    results = {
        "random": {"metric_value": 0.872, "convergence_epoch": 47},
        "easy_to_hard": {"metric_value": 0.878, "convergence_epoch": 38},
        "hard_to_easy": {"metric_value": 0.869, "convergence_epoch": 52},
    }
    comparison = compare_strategies(results)
    assert comparison["best_strategy"] == "easy_to_hard"
    assert comparison["verdict"] == "curriculum_helps"

def test_compare_no_improvement():
    results = {
        "random": {"metric_value": 0.872, "convergence_epoch": 47},
        "easy_to_hard": {"metric_value": 0.873, "convergence_epoch": 45},
    }
    comparison = compare_strategies(results)
    assert comparison["verdict"] in ("no_improvement", "faster_convergence")

def test_compare_empty():
    comparison = compare_strategies({})
    assert comparison["best_strategy"] is None

# --- curriculum_analysis ---

def test_analysis_with_difficulties():
    difficulties = np.array([0.1, 0.3, 0.95, 0.5, 0.92])
    report = curriculum_analysis(difficulties=difficulties)
    assert report["difficulty_stats"]["n_impossible"] == 2

def test_analysis_with_results():
    results = {
        "random": {"metric_value": 0.87, "convergence_epoch": 50},
        "easy_to_hard": {"metric_value": 0.88, "convergence_epoch": 40},
    }
    report = curriculum_analysis(strategy_results=results)
    assert "comparison" in report

def test_analysis_no_data():
    report = curriculum_analysis()
    assert "note" in report

# --- format_curriculum_report ---

def test_format_comparison():
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "experiment_id": "exp-042",
        "primary_metric": "accuracy",
        "comparison": {
            "results": [
                {"strategy": "random", "metric_value": 0.872, "delta_vs_random": 0.0, "convergence_epoch": 47, "speedup": None},
                {"strategy": "easy_to_hard", "metric_value": 0.878, "delta_vs_random": 0.006, "convergence_epoch": 38, "speedup": 0.19},
            ],
            "best_strategy": "easy_to_hard",
            "verdict": "curriculum_helps",
        },
    }
    text = format_curriculum_report(report)
    assert "Curriculum" in text
    assert "easy_to_hard" in text
    assert "BEST" in text

def test_format_error():
    assert "ERROR" in format_curriculum_report({"error": "fail"})

def test_format_no_comparison():
    report = {
        "generated_at": "2026-01-01T00:00:00",
        "experiment_id": "exp-001",
        "primary_metric": "accuracy",
        "note": "Run strategies first",
    }
    text = format_curriculum_report(report)
    assert "Run strategies" in text
