"""Tests for model card enhancements (Phase 28.3).

Verifies fairness section, registry status integration, and enhanced assembly.
"""

from __future__ import annotations

import pytest
import yaml

from scripts.generate_model_card import (
    compute_fairness_metrics,
    load_registry_status,
    generate_card,
)


# --- compute_fairness_metrics ---

def test_fairness_basic():
    result = compute_fairness_metrics(
        predictions=[1, 1, 0, 0, 1, 0],
        labels=[1, 0, 0, 1, 1, 0],
        protected_attribute=["A", "A", "A", "B", "B", "B"],
    )
    assert result is not None
    assert "demographic_parity_difference" in result
    assert result["n_groups"] == 2
    assert "A" in result["group_positive_rates"]
    assert "B" in result["group_positive_rates"]

def test_fairness_with_group_names():
    result = compute_fairness_metrics(
        predictions=[1, 0, 1, 0],
        protected_attribute=[0, 0, 1, 1],
        group_names=["Male", "Female"],
    )
    assert "Male" in result["group_positive_rates"]
    assert "Female" in result["group_positive_rates"]

def test_fairness_equal_opportunity():
    result = compute_fairness_metrics(
        predictions=[1, 1, 0, 0, 1, 0],
        labels=[1, 1, 0, 0, 1, 0],
        protected_attribute=["A", "A", "A", "B", "B", "B"],
    )
    assert "equal_opportunity_difference" in result
    assert "group_true_positive_rates" in result

def test_fairness_perfect_parity():
    result = compute_fairness_metrics(
        predictions=[1, 0, 1, 0],
        protected_attribute=["A", "A", "B", "B"],
    )
    assert result["demographic_parity_difference"] == 0.0

def test_fairness_no_predictions():
    assert compute_fairness_metrics(None, None, None) is None

def test_fairness_empty():
    assert compute_fairness_metrics([], [], []) is None

def test_fairness_mismatched_lengths():
    assert compute_fairness_metrics([1, 0], None, [1]) is None

def test_fairness_single_group():
    result = compute_fairness_metrics(
        predictions=[1, 0, 1],
        protected_attribute=["A", "A", "A"],
    )
    assert result["demographic_parity_difference"] == 0


# --- load_registry_status ---

def test_load_registry_exists(tmp_path):
    reg_path = tmp_path / "registry.yaml"
    reg_path.write_text(yaml.dump({
        "models": [{"exp_id": "exp-001", "stage": "production", "version": "v1"}],
        "history": [],
    }))
    result = load_registry_status(str(reg_path))
    assert result is not None
    assert len(result["models"]) == 1

def test_load_registry_missing(tmp_path):
    assert load_registry_status(str(tmp_path / "nonexistent.yaml")) is None

def test_load_registry_empty_models(tmp_path):
    reg_path = tmp_path / "registry.yaml"
    reg_path.write_text(yaml.dump({"models": [], "history": []}))
    assert load_registry_status(str(reg_path)) is None


# --- generate_card with fairness ---

def test_card_with_fairness(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\ndata:\n  source: test\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-001", "status": "kept", "config": {"model_type": "xgboost"}, "metrics": {"accuracy": 0.85}}\n')

    card = generate_card(
        str(config), str(log),
        contract_path=str(tmp_path / "contract.md"),
        include_fairness=True,
        fairness_data={
            "predictions": [1, 0, 1, 0, 1, 0],
            "labels": [1, 0, 0, 1, 1, 0],
            "protected_attribute": ["A", "A", "A", "B", "B", "B"],
        },
    )
    assert "Fairness Analysis" in card
    assert "Demographic Parity" in card

def test_card_fairness_no_data(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\ndata:\n  source: test\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-001", "status": "kept", "config": {}, "metrics": {"accuracy": 0.85}}\n')

    card = generate_card(
        str(config), str(log),
        contract_path=str(tmp_path / "contract.md"),
        include_fairness=True,
    )
    assert "Fairness" in card
    assert "no protected attribute" in card

def test_card_without_fairness(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\ndata:\n  source: test\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-001", "status": "kept", "config": {}, "metrics": {"accuracy": 0.85}}\n')

    card = generate_card(
        str(config), str(log),
        contract_path=str(tmp_path / "contract.md"),
    )
    assert "Fairness" not in card


# --- generate_card with registry ---

def test_card_with_registry(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\ndata:\n  source: test\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-001", "status": "kept", "config": {}, "metrics": {"accuracy": 0.85}}\n')
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.dump({
        "models": [{"exp_id": "exp-001", "stage": "staging", "version": "v2",
                     "registered_at": "2026-04-01T00:00:00Z", "gates_passed": ["regression"]}],
        "history": [],
    }))

    card = generate_card(
        str(config), str(log),
        contract_path=str(tmp_path / "contract.md"),
        registry_path=str(reg),
    )
    assert "Registry Status" in card
    assert "staging" in card
    assert "v2" in card
