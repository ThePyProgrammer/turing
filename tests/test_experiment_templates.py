"""Tests for experiment templates (experiment_templates.py). Phase 24.6."""
from __future__ import annotations
import pytest
from scripts.experiment_templates import (
    extract_template, save_template, list_templates, show_template,
    delete_template, format_templates_report, _strip_path_values,
)

EXP = {
    "experiment_id": "exp-042",
    "config": {"model_type": "xgboost", "hyperparams": {"max_depth": 6, "n_estimators": 500}},
    "metrics": {"accuracy": 0.883},
}

def test_extract_template():
    tmpl = extract_template(EXP)
    assert "config" in tmpl or "model_type" in tmpl

def test_strip_paths():
    d = {"path": "/home/user/data.csv", "value": 42, "nested": {"file": "/tmp/x.py"}}
    stripped = _strip_path_values(d)
    assert stripped["value"] == 42
    assert "<" in str(stripped.get("path", "")) or stripped.get("path") != "/home/user/data.csv"

def test_save_and_list(tmp_path):
    tmpl_dir = str(tmp_path / "templates")
    save_template("test-tmpl", {"config": {"model_type": "xgboost"}}, "Test", template_dir=tmpl_dir)
    templates = list_templates(tmpl_dir)
    assert len(templates) == 1
    assert templates[0]["name"] == "test-tmpl"

def test_show_template(tmp_path):
    tmpl_dir = str(tmp_path / "templates")
    save_template("test-tmpl", {"config": {"model_type": "xgboost"}}, "Test", template_dir=tmpl_dir)
    tmpl = show_template("test-tmpl", tmpl_dir)
    assert tmpl is not None

def test_show_missing(tmp_path):
    assert show_template("missing", str(tmp_path)) is None

def test_delete_template(tmp_path):
    tmpl_dir = str(tmp_path / "templates")
    save_template("test-tmpl", {"config": {}}, "Test", template_dir=tmpl_dir)
    result = delete_template("test-tmpl", tmpl_dir)
    assert "deleted" in str(result).lower() or result.get("status") == "deleted"
    assert list_templates(tmpl_dir) == []

def test_list_empty(tmp_path):
    assert list_templates(str(tmp_path / "nonexistent")) == []

def test_format_templates():
    templates = [{"name": "xgboost-v1", "description": "XGBoost baseline", "source_project": "fraud"}]
    text = format_templates_report(templates)
    assert "xgboost-v1" in text
