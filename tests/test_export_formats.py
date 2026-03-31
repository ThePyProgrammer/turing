"""Tests for export_formats.py and export_card.py."""

from __future__ import annotations

import yaml
import pytest

from scripts.export_formats import (
    DEFAULT_FORMAT,
    FORMAT_DEPENDENCIES,
    FORMAT_EXTENSIONS,
    FORMAT_REGISTRY,
    detect_model_type,
    export_joblib,
    export_model,
    get_default_format,
    get_supported_formats,
)
from scripts.export_card import (
    format_export_card,
    generate_export_card,
    save_export_card,
)


# --- export_formats: registry ---


def test_registry_has_common_types():
    """Should have entries for common model types."""
    for mt in ["xgboost", "lightgbm", "random_forest", "pytorch", "tensorflow"]:
        assert mt in FORMAT_REGISTRY


def test_default_format_exists():
    """Every registered type should have a default format."""
    for mt in FORMAT_REGISTRY:
        assert mt in DEFAULT_FORMAT


def test_supported_formats_include_default():
    """Default format should be in supported list."""
    for mt in FORMAT_REGISTRY:
        default = DEFAULT_FORMAT[mt]
        supported = FORMAT_REGISTRY[mt]
        assert default in supported


def test_extensions_for_all_formats():
    """Every format in registry should have an extension."""
    all_formats = set()
    for formats in FORMAT_REGISTRY.values():
        all_formats.update(formats)
    for fmt in all_formats:
        assert fmt in FORMAT_EXTENSIONS, f"No extension for format {fmt}"


def test_dependencies_for_all_formats():
    """Every format should have dependency info."""
    all_formats = set()
    for formats in FORMAT_REGISTRY.values():
        all_formats.update(formats)
    for fmt in all_formats:
        assert fmt in FORMAT_DEPENDENCIES, f"No dependencies for format {fmt}"


# --- export_formats: detection ---


def test_detect_xgboost():
    """Should detect XGBoost from config."""
    assert detect_model_type({"model": {"type": "xgboost"}}) == "xgboost"


def test_detect_lightgbm():
    """Should detect LightGBM from config."""
    assert detect_model_type({"model": {"type": "LightGBM"}}) == "lightgbm"


def test_detect_pytorch():
    """Should detect PyTorch from config."""
    assert detect_model_type({"model": {"type": "pytorch"}}) == "pytorch"


def test_detect_unknown():
    """Should return unknown for missing type."""
    assert detect_model_type({}) == "unknown"


def test_detect_normalizes():
    """Should normalize hyphens and spaces."""
    assert detect_model_type({"model": {"type": "Random-Forest"}}) == "random_forest"
    assert detect_model_type({"model": {"type": "random forest"}}) == "random_forest"


# --- export_formats: get functions ---


def test_get_supported_xgboost():
    """XGBoost should support joblib, xgboost_json, onnx."""
    formats = get_supported_formats("xgboost")
    assert "joblib" in formats
    assert "xgboost_json" in formats


def test_get_supported_unknown():
    """Unknown type should default to joblib."""
    formats = get_supported_formats("unknown_model")
    assert "joblib" in formats


def test_get_default_xgboost():
    """XGBoost default should be joblib."""
    assert get_default_format("xgboost") == "joblib"


def test_get_default_pytorch():
    """PyTorch default should be torchscript."""
    assert get_default_format("pytorch") == "torchscript"


# --- export_formats: joblib export ---


def test_export_joblib(tmp_path):
    """Should copy model file to output directory."""
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"fake model data" * 100)
    output_dir = tmp_path / "output"

    result = export_joblib(str(model_file), str(output_dir), "test-model")
    assert "error" not in result
    assert result["format"] == "joblib"
    assert result["size_bytes"] > 0
    assert (output_dir / "test-model.joblib").exists()


def test_export_joblib_missing_file():
    """Should return error for missing model file."""
    result = export_joblib("/nonexistent/model.joblib", "/tmp/out", "test")
    assert "error" in result


# --- export_formats: export_model dispatcher ---


def test_export_model_unsupported_format():
    """Should return error for unsupported format."""
    result = export_model("/tmp/model.joblib", "/tmp/out", "test", "xgboost", "tflite")
    assert "error" in result
    assert "not supported" in result["error"]


def test_export_model_unknown_format():
    """Should return error for unknown format."""
    result = export_model("/tmp/model.joblib", "/tmp/out", "test", "xgboost", "banana_format")
    assert "error" in result


# --- export_card: generation ---


def test_generate_card_basic():
    """Should generate a card with required fields."""
    experiment = {
        "experiment_id": "exp-042",
        "metrics": {"accuracy": 0.872, "f1": 0.869},
        "config": {"model_type": "xgboost"},
        "timestamp": "2026-04-01T00:00:00Z",
    }
    export_result = {
        "path": "exports/exp-042/model.joblib",
        "format": "joblib",
        "size_mb": 2.3,
        "dependencies": ["joblib"],
    }
    config = {
        "evaluation": {"primary_metric": "accuracy"},
        "task_description": "Binary classification",
    }
    card = generate_export_card(experiment, export_result, config=config)

    assert card["experiment_id"] == "exp-042"
    assert card["export_format"] == "joblib"
    assert card["metrics"]["accuracy"] == 0.872
    assert card["task"] == "Binary classification"
    assert card["dependencies"] == ["joblib"]


def test_generate_card_with_equivalence():
    """Should include equivalence info."""
    experiment = {"experiment_id": "exp-001", "metrics": {}, "config": {}}
    export_result = {"format": "joblib", "dependencies": []}
    equivalence = {"verdict": "equivalent", "max_delta": 0.0, "n_samples": 100}

    card = generate_export_card(experiment, export_result, equivalence=equivalence)
    assert card["equivalence"]["verdict"] == "equivalent"


def test_generate_card_with_latency():
    """Should include latency info."""
    experiment = {"experiment_id": "exp-001", "metrics": {}, "config": {}}
    export_result = {"format": "joblib", "dependencies": []}
    latency = {
        "verdict": "faster",
        "exported_p50_ms": 1.0,
        "exported_p95_ms": 2.0,
        "original_p50_ms": 5.0,
        "speedup_ratio": 5.0,
    }

    card = generate_export_card(experiment, export_result, latency=latency)
    assert card["inference_latency"]["speedup"] == 5.0


# --- export_card: save ---


def test_save_card(tmp_path):
    """Should save YAML card file."""
    card = {"name": "test-model", "experiment_id": "exp-042"}
    filepath = save_export_card(card, str(tmp_path))
    assert filepath.exists()
    assert filepath.name == "model_card.yaml"
    with open(filepath) as f:
        loaded = yaml.safe_load(f)
    assert loaded["experiment_id"] == "exp-042"


# --- export_card: format ---


def test_format_card():
    """Should produce readable markdown."""
    card = {
        "name": "exp-042-xgboost",
        "experiment_id": "exp-042",
        "task": "Binary classification",
        "model_type": "xgboost",
        "export_format": "joblib",
        "size_mb": 2.3,
        "dependencies": ["joblib"],
        "metrics": {"accuracy": 0.872},
        "export_date": "2026-04-01",
    }
    report = format_export_card(card)
    assert "exp-042" in report
    assert "joblib" in report
    assert "0.8720" in report


def test_format_card_with_seed_study():
    """Should include seed study section."""
    card = {
        "name": "test",
        "experiment_id": "exp-001",
        "task": "test",
        "model_type": "xgboost",
        "export_format": "joblib",
        "size_mb": 1.0,
        "dependencies": [],
        "metrics": {},
        "export_date": "2026-04-01",
        "seed_study": {"mean": 0.868, "std": 0.007, "cv_percent": 0.8, "seed_sensitive": False, "seeds_tested": 5},
    }
    report = format_export_card(card)
    assert "Seed Study" in report
    assert "STABLE" in report


def test_format_card_with_equivalence():
    """Should include equivalence section."""
    card = {
        "name": "test",
        "experiment_id": "exp-001",
        "task": "test",
        "model_type": "xgboost",
        "export_format": "joblib",
        "size_mb": 1.0,
        "dependencies": [],
        "metrics": {},
        "export_date": "2026-04-01",
        "equivalence": {"verdict": "equivalent", "max_delta": 0.0, "n_samples_tested": 100},
    }
    report = format_export_card(card)
    assert "Equivalence" in report
    assert "PASS" in report
