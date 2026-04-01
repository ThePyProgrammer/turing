"""Tests for what-if analysis engine (whatif_engine.py).

Phase 27.1: Verifies question classification, estimator routing, confidence calibration, reporting.
"""

from __future__ import annotations

import pytest

from scripts.whatif_engine import (
    classify_question,
    extract_multiplier,
    extract_experiment_ids,
    extract_target_value,
    estimate_scaling,
    estimate_ablation,
    estimate_sensitivity,
    estimate_ensemble,
    estimate_pruning,
    estimate_stitch,
    estimate_budget,
    whatif_analysis,
    format_whatif_report,
    _get_best_metric,
    _r_squared_to_confidence,
    _linear_interpolate,
    _generate_recommendation,
    ROUTE_PATTERNS,
)


# --- classify_question ---

def test_classify_scaling():
    result = classify_question("what if I had 2x more data")
    assert result["route"] == "scaling"

def test_classify_ablation():
    result = classify_question("what if I removed class 3")
    assert result["route"] == "ablation"

def test_classify_stitch():
    result = classify_question("combine features from exp-031 with model from exp-042")
    assert result["route"] == "stitch"

def test_classify_sensitivity():
    result = classify_question("what if learning_rate was 0.01")
    assert result["route"] == "sensitivity"

def test_classify_ensemble():
    result = classify_question("what if I ensembled the top models")
    assert result["route"] == "ensemble"

def test_classify_pruning():
    result = classify_question("what if I pruned to 50% sparsity")
    assert result["route"] == "pruning"

def test_classify_budget():
    result = classify_question("what if I spent my remaining budget on tuning vs ensembling")
    assert result["route"] == "budget"

def test_classify_unknown():
    result = classify_question("what is the meaning of life")
    assert result["route"] == "unknown"


# --- extract_multiplier ---

def test_multiplier_2x():
    assert extract_multiplier("2x more data") == 2.0

def test_multiplier_10x():
    assert extract_multiplier("10x the data") == 10.0

def test_multiplier_double():
    assert extract_multiplier("double the data") == 2.0

def test_multiplier_triple():
    assert extract_multiplier("triple the samples") == 3.0

def test_multiplier_half():
    assert extract_multiplier("half the data") == 0.5

def test_multiplier_none():
    assert extract_multiplier("remove class 3") is None


# --- extract_experiment_ids ---

def test_extract_ids():
    ids = extract_experiment_ids("combine exp-031 with exp-042")
    assert ids == ["031", "042"]

def test_extract_no_ids():
    assert extract_experiment_ids("more data please") == []


# --- extract_target_value ---

def test_target_percent():
    assert extract_target_value("prune to 50% sparsity") == 50.0

def test_target_decimal_percent():
    assert extract_target_value("at 33.5% sparsity") == 33.5

def test_target_none():
    assert extract_target_value("more data") is None


# --- _get_best_metric ---

def test_best_metric():
    exps = [
        {"metrics": {"accuracy": 0.82}},
        {"metrics": {"accuracy": 0.87}},
        {"metrics": {"accuracy": 0.85}},
    ]
    assert _get_best_metric(exps, "accuracy") == 0.87

def test_best_metric_empty():
    assert _get_best_metric([], "accuracy") is None

def test_best_metric_missing():
    exps = [{"metrics": {"loss": 0.3}}]
    assert _get_best_metric(exps, "accuracy") is None


# --- _r_squared_to_confidence ---

def test_confidence_high():
    assert _r_squared_to_confidence(0.99) == "HIGH"

def test_confidence_med():
    assert _r_squared_to_confidence(0.90) == "MED"

def test_confidence_low():
    assert _r_squared_to_confidence(0.50) == "LOW"


# --- _linear_interpolate ---

def test_interpolate_middle():
    result = _linear_interpolate([0, 10, 20], [0, 1, 2], 5)
    assert result == pytest.approx(0.5)

def test_interpolate_boundary():
    result = _linear_interpolate([0, 10], [0, 1], 0)
    assert result == 0

def test_interpolate_beyond():
    result = _linear_interpolate([0, 10], [0, 1], 20)
    assert result == 1  # clamps to last

def test_interpolate_empty():
    assert _linear_interpolate([], [], 5) is None


# --- estimate_scaling ---

def test_scaling_no_multiplier():
    result = estimate_scaling("remove class 3", [], "accuracy")
    assert "error" in result

def test_scaling_no_experiments():
    result = estimate_scaling("2x more data", [], "accuracy")
    assert "error" in result

def test_scaling_fallback():
    exps = [{"metrics": {"accuracy": 0.85}}]
    result = estimate_scaling("2x more data", exps, "accuracy")
    assert result["confidence"] == "LOW"
    assert result["estimate"] is not None
    assert result["current"] == 0.85


# --- estimate_ablation ---

def test_ablation_no_data():
    result = estimate_ablation("remove class 3", [], "accuracy", ablation_dir="/nonexistent")
    assert "error" in result


# --- estimate_sensitivity ---

def test_sensitivity_no_data():
    result = estimate_sensitivity("learning_rate 0.01", [], "accuracy", sensitivity_dir="/nonexistent")
    assert "error" in result


# --- estimate_ensemble ---

def test_ensemble_conservative():
    exps = [{"metrics": {"accuracy": 0.85}}]
    result = estimate_ensemble("ensemble top models", exps, "accuracy", ensemble_dir="/nonexistent")
    assert result["confidence"] == "LOW"
    assert result["estimate"] > 0.85

def test_ensemble_no_data():
    result = estimate_ensemble("ensemble", [], "accuracy", ensemble_dir="/nonexistent")
    assert "error" in result


# --- estimate_pruning ---

def test_pruning_no_data():
    result = estimate_pruning("prune to 50%", [], "accuracy", pruning_dir="/nonexistent")
    assert "error" in result


# --- estimate_stitch ---

def test_stitch_missing_ids():
    result = estimate_stitch("combine models", [], "accuracy")
    assert "error" in result

def test_stitch_missing_experiments():
    result = estimate_stitch("combine exp-031 with exp-042", [], "accuracy")
    assert "error" in result

def test_stitch_found():
    exps = [
        {"experiment_id": "exp-031", "metrics": {"accuracy": 0.85}},
        {"experiment_id": "exp-042", "metrics": {"accuracy": 0.88}},
    ]
    result = estimate_stitch("combine exp-031 with exp-042", exps, "accuracy")
    assert result["estimate"] is not None
    assert result["estimate"] > 0.88


# --- estimate_budget ---

def test_budget_basic():
    exps = [
        {"status": "kept", "metrics": {"accuracy": 0.85}},
        {"status": "discarded", "metrics": {"accuracy": 0.80}},
    ]
    result = estimate_budget("spend budget on tuning vs data", exps, "accuracy")
    assert result["total_experiments"] == 2
    assert result["kept_ratio"] == 0.5

def test_budget_empty():
    result = estimate_budget("remaining budget", [], "accuracy")
    assert result["total_experiments"] == 0


# --- whatif_analysis ---

def test_analysis_unknown(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text("")

    result = whatif_analysis("meaning of life", str(config), str(log))
    assert result["route"] == "unknown"
    assert "error" in result

def test_analysis_scaling(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"metrics": {"accuracy": 0.85}}\n')

    result = whatif_analysis("what if I had 2x more data", str(config), str(log))
    assert result["route"] == "scaling"
    assert "result" in result


# --- _generate_recommendation ---

def test_rec_error():
    rec = _generate_recommendation({"error": "no data"}, "scaling", {})
    assert "no data" in rec

def test_rec_marginal():
    rec = _generate_recommendation({"estimate": 0.85, "current": 0.85, "delta": 0.0005}, "scaling", {"verify_cmd": "/turing:scale"})
    assert "Marginal" in rec

def test_rec_promising():
    rec = _generate_recommendation({"estimate": 0.87, "current": 0.85, "delta": 0.02}, "scaling", {"verify_cmd": "/turing:scale"})
    assert "Promising" in rec

def test_rec_negative():
    rec = _generate_recommendation({"estimate": 0.83, "current": 0.85, "delta": -0.02}, "scaling", {"verify_cmd": "/turing:scale"})
    assert "Negative" in rec


# --- format_whatif_report ---

def test_format_report():
    report = {
        "question": "2x more data",
        "route": "scaling",
        "source": "scaling law",
        "result": {"current": 0.85, "estimate": 0.89, "delta": 0.04, "confidence": "HIGH"},
        "recommendation": "Promising",
        "verify_cmd": "/turing:scale",
        "generated_at": "2026-04-01T00:00:00Z",
    }
    text = format_whatif_report(report)
    assert "What-If Analysis" in text
    assert "0.85" in text
    assert "0.89" in text

def test_format_error_report():
    report = {"error": "boom"}
    text = format_whatif_report(report)
    assert "ERROR" in text
