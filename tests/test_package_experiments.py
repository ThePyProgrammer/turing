"""Tests for package_experiments.py. Phase 26.2."""
from __future__ import annotations
import pytest
from scripts.package_experiments import collect_experiment_artifacts, build_manifest

EXP = {"experiment_id": "exp-089", "config": {"model_type": "lightgbm", "hyperparams": {"max_depth": 6}},
       "metrics": {"accuracy": 0.891}, "timestamp": "2026-03-25T12:00:00"}

def test_collect_artifacts():
    arts = collect_experiment_artifacts(EXP, [])
    assert arts["experiment_id"] == "exp-089"
    assert "metrics" in arts

def test_collect_with_includes():
    arts = collect_experiment_artifacts(EXP, ["model", "code"])
    assert isinstance(arts, dict)

def test_build_manifest():
    arts = [collect_experiment_artifacts(EXP, [])]
    manifest = build_manifest("test-pkg", {}, arts, [])
    assert isinstance(manifest, dict)
    assert len(manifest.get("experiments", [])) >= 1
