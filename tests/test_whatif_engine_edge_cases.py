"""Edge case tests for what-if analysis engine (whatif_engine.py).

Phase 27.1: No data scenarios, ambiguous questions, boundary conditions.
"""

from __future__ import annotations

import json

import pytest
import yaml

from scripts.whatif_engine import (
    classify_question,
    extract_multiplier,
    extract_experiment_ids,
    extract_target_value,
    estimate_scaling,
    estimate_stitch,
    whatif_analysis,
    format_whatif_report,
    save_whatif_report,
    _get_best_metric,
    _linear_interpolate,
)


# --- Ambiguous questions ---

def test_classify_empty():
    result = classify_question("")
    assert result["route"] == "unknown"

def test_classify_gibberish():
    result = classify_question("asdf1234 !!!!")
    assert result["route"] == "unknown"

def test_classify_partial_match():
    """Question mentions data but not in scaling context."""
    result = classify_question("what is data science")
    assert result["route"] == "unknown"

def test_classify_case_insensitive():
    result = classify_question("WHAT IF I HAD 2X MORE DATA")
    assert result["route"] == "scaling"

def test_classify_multiple_routes():
    """First matching route wins."""
    result = classify_question("what if I had 2x more data and pruned to 50%")
    assert result["route"] == "scaling"  # scaling matched first


# --- Multiplier edge cases ---

def test_multiplier_float():
    assert extract_multiplier("1.5x more data") == 1.5

def test_multiplier_large():
    assert extract_multiplier("100x the data") == 100.0

def test_multiplier_no_context():
    """Multiplier without 'data' context."""
    assert extract_multiplier("2x faster") is None


# --- Experiment ID edge cases ---

def test_extract_single_id():
    ids = extract_experiment_ids("analyze exp-001")
    assert ids == ["001"]

def test_extract_many_ids():
    ids = extract_experiment_ids("exp-001 exp-002 exp-003 exp-100")
    assert len(ids) == 4


# --- Linear interpolation edge cases ---

def test_interpolate_single_point():
    # Single point clamps to boundary
    assert _linear_interpolate([5], [0.5], 5) == 0.5

def test_interpolate_unsorted():
    """Points given out of order should still interpolate correctly."""
    result = _linear_interpolate([20, 0, 10], [2, 0, 1], 5)
    assert result == pytest.approx(0.5)

def test_interpolate_negative():
    result = _linear_interpolate([-10, 0, 10], [0, 1, 2], -5)
    assert result == pytest.approx(0.5)

def test_interpolate_same_x():
    """Two points with same x value."""
    result = _linear_interpolate([5, 5, 10], [1, 1, 2], 7)
    # Should handle gracefully (division by zero protection)
    assert result is not None


# --- Scaling with existing scaling data ---

def test_scaling_with_fit_data(tmp_path):
    scaling_dir = tmp_path / "scaling"
    scaling_dir.mkdir()
    fit_data = {
        "fit": {"a": 0.5, "b": 0.1, "c": 0.4, "r_squared": 0.98},
    }
    (scaling_dir / "results.yaml").write_text(yaml.dump(fit_data))

    exps = [{"metrics": {"accuracy": 0.85}}]
    result = estimate_scaling("2x more data", exps, "accuracy", str(scaling_dir))
    assert result["confidence"] == "HIGH"
    assert result["estimate"] is not None


# --- Stitch edge cases ---

def test_stitch_one_id_only():
    result = estimate_stitch("use exp-042", [], "accuracy")
    assert "error" in result

def test_stitch_partial_match():
    """One exp found, one not."""
    exps = [{"experiment_id": "exp-031", "metrics": {"accuracy": 0.85}}]
    result = estimate_stitch("combine exp-031 with exp-999", exps, "accuracy")
    assert "error" in result


# --- whatif_analysis with missing config ---

def test_analysis_no_config(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"metrics": {"accuracy": 0.85}}\n')
    result = whatif_analysis("2x more data", str(tmp_path / "missing.yaml"), str(log))
    assert result["route"] == "scaling"
    # Should still work with default config


# --- save_whatif_report ---

def test_save_report(tmp_path):
    report = {"question": "test", "route": "scaling", "generated_at": "2026-01-01"}
    path = save_whatif_report(report, str(tmp_path / "whatif"))
    assert path.exists()
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data["question"] == "test"


# --- format edge cases ---

def test_format_unknown_route():
    report = {
        "question": "gibberish",
        "route": "unknown",
        "source": None,
        "result": {"error": "cannot classify"},
        "available_routes": ["scaling", "ablation"],
        "suggestion": "Try...",
        "generated_at": "2026-01-01",
    }
    text = format_whatif_report(report)
    assert "cannot classify" in text

def test_format_no_estimate():
    report = {
        "question": "test",
        "route": "sensitivity",
        "source": "sens",
        "result": {"current": 0.85, "estimate": None, "confidence": "LOW"},
        "generated_at": "2026-01-01",
    }
    text = format_whatif_report(report)
    assert "0.85" in text

def test_format_negative_delta():
    report = {
        "question": "test",
        "route": "ablation",
        "source": "ablation",
        "result": {"current": 0.85, "estimate": 0.82, "delta": -0.03, "confidence": "HIGH"},
        "generated_at": "2026-01-01",
    }
    text = format_whatif_report(report)
    assert "-0.03" in text


# --- _get_best_metric edge cases ---

def test_best_metric_none_values():
    exps = [{"metrics": {"accuracy": None}}, {"metrics": {"accuracy": 0.85}}]
    assert _get_best_metric(exps, "accuracy") == 0.85

def test_best_metric_no_metrics_key():
    exps = [{"something": "else"}]
    assert _get_best_metric(exps, "accuracy") is None
