"""Tests for hyperparameter sweep queue management (sweep.py).

Tier 2: Verifies cartesian product generation and config override application.
"""

from __future__ import annotations

import copy

from scripts.sweep import apply_overrides, _make_experiment_name


def test_apply_overrides_dotted_path():
    """Dotted path should set nested dict value."""
    config = {"model": {"hyperparams": {"n_estimators": 100}}}
    result = apply_overrides(config, {"model.hyperparams.n_estimators": 200})
    assert result["model"]["hyperparams"]["n_estimators"] == 200


def test_apply_overrides_creates_intermediate():
    """Missing intermediate keys should be created."""
    config = {}
    result = apply_overrides(config, {"model.hyperparams.n_estimators": 100})
    assert result["model"]["hyperparams"]["n_estimators"] == 100


def test_apply_overrides_does_not_mutate():
    """Original config should not be modified."""
    config = {"model": {"hyperparams": {"n_estimators": 100}}}
    original = copy.deepcopy(config)
    apply_overrides(config, {"model.hyperparams.n_estimators": 200})
    assert config == original


def test_apply_overrides_multiple():
    """Multiple overrides should all be applied."""
    config = {"model": {"hyperparams": {"n_estimators": 100, "max_depth": 4}}}
    overrides = {
        "model.hyperparams.n_estimators": 200,
        "model.hyperparams.max_depth": 8,
    }
    result = apply_overrides(config, overrides)
    assert result["model"]["hyperparams"]["n_estimators"] == 200
    assert result["model"]["hyperparams"]["max_depth"] == 8


def test_make_experiment_name():
    """Experiment names should use abbreviated parameter names."""
    overrides = {
        "model.hyperparams.n_estimators": 100,
        "model.hyperparams.max_depth": 4,
    }
    name = _make_experiment_name(overrides)
    assert "n100" in name
    assert "d4" in name
