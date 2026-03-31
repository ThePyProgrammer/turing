"""Tests for tree-search-guided hypothesis exploration (treequest_suggest.py).

Verifies HypothesisNode dataclass, seed generation, child perturbation,
critique-based scoring, greedy fallback search, result formatting,
and JSON serialization.
"""

from __future__ import annotations

import json

import pytest

from scripts.treequest_suggest import (
    HypothesisNode,
    format_results,
    generate_children,
    generate_seed_hypotheses,
    results_to_json,
    run_greedy_search,
    score_hypothesis,
)


# --- Fixtures ---


def _exp(exp_id, desc, status="kept", accuracy=0.85, exp_type="hyperparameter"):
    """Create a minimal experiment dict matching log.jsonl schema."""
    return {
        "experiment_id": exp_id,
        "description": desc,
        "status": status,
        "config": {"model_type": "xgboost", "experiment_type": exp_type},
        "metrics": {"accuracy": accuracy},
    }


def _config(model_type="xgboost", metric="accuracy"):
    """Create a minimal config dict matching config.yaml schema."""
    return {
        "model": {"type": model_type},
        "evaluation": {"primary_metric": metric},
    }


# --- HypothesisNode ---


class TestHypothesisNode:
    def test_basic_creation(self):
        node = HypothesisNode(description="try LightGBM")
        assert node.description == "try LightGBM"
        assert node.depth == 0
        assert node.model_type is None
        assert node.critique_scores == {}

    def test_full_creation(self):
        node = HypothesisNode(
            description="switch to catboost",
            model_type="catboost",
            hyperparameters={"depth": 6, "learning_rate": 0.03},
            feature_changes={"add": ["polynomial"]},
            parent_description="try gradient boosting",
            depth=2,
        )
        assert node.model_type == "catboost"
        assert node.hyperparameters["depth"] == 6
        assert node.depth == 2

    def test_to_dict_roundtrip(self):
        node = HypothesisNode(
            description="test hypothesis",
            model_type="xgboost",
            hyperparameters={"n_estimators": 100},
            depth=1,
        )
        d = node.to_dict()
        assert d["description"] == "test hypothesis"
        assert d["model_type"] == "xgboost"
        assert d["depth"] == 1

        # Roundtrip
        restored = HypothesisNode.from_dict(d)
        assert restored.description == node.description
        assert restored.model_type == node.model_type
        assert restored.depth == node.depth

    def test_from_dict_missing_optional_fields(self):
        d = {"description": "minimal node"}
        node = HypothesisNode.from_dict(d)
        assert node.description == "minimal node"
        assert node.model_type is None
        assert node.depth == 0

    def test_to_dict_includes_critique_scores(self):
        node = HypothesisNode(description="scored node")
        node.critique_scores = {"overall": 7.5, "novelty": 8}
        d = node.to_dict()
        assert d["critique_scores"]["overall"] == 7.5


# --- Seed Generation ---


class TestSeedGeneration:
    def test_seeds_from_xgboost_config(self):
        config = _config("xgboost", "accuracy")
        seeds = generate_seed_hypotheses(config, [])
        # Should have model alternatives + regularization + features + lr schedule
        assert len(seeds) >= 4
        descriptions = [s.description.lower() for s in seeds]
        # Should suggest alternatives to xgboost
        assert any("lightgbm" in d for d in descriptions)

    def test_seeds_from_lightgbm_config(self):
        config = _config("lightgbm", "rmse")
        seeds = generate_seed_hypotheses(config, [])
        descriptions = [s.description.lower() for s in seeds]
        # Should suggest XGBoost as an alternative
        assert any("xgboost" in d for d in descriptions)

    def test_seeds_include_regularization(self):
        config = _config("xgboost", "accuracy")
        seeds = generate_seed_hypotheses(config, [])
        descriptions = [s.description.lower() for s in seeds]
        assert any("regularization" in d for d in descriptions)

    def test_seeds_include_feature_engineering(self):
        config = _config("xgboost", "accuracy")
        seeds = generate_seed_hypotheses(config, [])
        descriptions = [s.description.lower() for s in seeds]
        assert any("polynomial" in d or "feature" in d for d in descriptions)

    def test_seeds_include_lr_schedule(self):
        config = _config("xgboost", "accuracy")
        seeds = generate_seed_hypotheses(config, [])
        descriptions = [s.description.lower() for s in seeds]
        assert any("learning rate" in d for d in descriptions)

    def test_seeds_with_experiment_history(self):
        """When there are kept experiments, should generate a refinement seed."""
        config = _config("xgboost", "accuracy")
        experiments = [
            _exp("exp-001", "baseline xgboost", "kept", 0.80),
            _exp("exp-002", "tuned learning rate to 0.05", "kept", 0.83),
        ]
        seeds = generate_seed_hypotheses(config, experiments)
        descriptions = [s.description.lower() for s in seeds]
        # Should have a refinement based on last kept experiment
        assert any("refine" in d for d in descriptions)

    def test_seeds_without_history_no_refinement(self):
        """Without experiments, no refinement seed."""
        config = _config("xgboost", "accuracy")
        seeds = generate_seed_hypotheses(config, [])
        descriptions = [s.description.lower() for s in seeds]
        assert not any("refine" in d for d in descriptions)

    def test_seeds_unknown_model_type(self):
        """Unknown model type should still produce seeds."""
        config = _config("my_custom_model", "f1")
        seeds = generate_seed_hypotheses(config, [])
        assert len(seeds) >= 3  # At least regularization, features, lr

    def test_all_seeds_have_descriptions(self):
        config = _config("xgboost", "accuracy")
        seeds = generate_seed_hypotheses(config, [])
        for seed in seeds:
            assert seed.description
            assert len(seed.description) > 10


# --- Child Generation ---


class TestChildGeneration:
    def test_generates_requested_count(self):
        parent = HypothesisNode(description="parent hypothesis")
        children = generate_children(parent, n_children=3)
        assert len(children) == 3

    def test_children_reference_parent(self):
        parent = HypothesisNode(description="parent hypothesis")
        children = generate_children(parent, n_children=2)
        for child in children:
            assert child.parent_description == "parent hypothesis"
            assert child.depth == 1

    def test_children_extend_parent_description(self):
        parent = HypothesisNode(description="try XGBoost")
        children = generate_children(parent, n_children=1)
        assert children[0].description.startswith("try XGBoost; additionally")

    def test_children_inherit_model_type(self):
        parent = HypothesisNode(description="test", model_type="lightgbm")
        children = generate_children(parent, n_children=2)
        for child in children:
            assert child.model_type == "lightgbm"

    def test_deterministic_with_same_seed(self):
        parent = HypothesisNode(description="fixed parent")
        children_a = generate_children(parent, n_children=3, rng_seed=42)
        children_b = generate_children(parent, n_children=3, rng_seed=42)
        for a, b in zip(children_a, children_b):
            assert a.description == b.description

    def test_different_seeds_different_children(self):
        parent = HypothesisNode(description="fixed parent")
        children_a = generate_children(parent, n_children=3, rng_seed=1)
        children_b = generate_children(parent, n_children=3, rng_seed=99)
        # At least one child should differ
        descs_a = {c.description for c in children_a}
        descs_b = {c.description for c in children_b}
        assert descs_a != descs_b

    def test_depth_increments(self):
        root = HypothesisNode(description="root", depth=0)
        children = generate_children(root, n_children=1)
        grandchildren = generate_children(children[0], n_children=1)
        assert children[0].depth == 1
        assert grandchildren[0].depth == 2

    def test_children_are_diverse(self):
        """Children of the same parent should have different perturbations."""
        parent = HypothesisNode(description="test diversity")
        children = generate_children(parent, n_children=3)
        descriptions = [c.description for c in children]
        assert len(set(descriptions)) == 3


# --- Scoring ---


class TestScoring:
    def test_score_returns_float(self, tmp_path):
        # Create minimal log and config for scoring
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("evaluation:\n  primary_metric: accuracy\n")

        node = HypothesisNode(description="try a novel approach")
        score = score_hypothesis(node, str(log_path), str(config_path))
        assert isinstance(score, float)
        assert 0 <= score <= 10

    def test_score_populates_critique_scores(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("evaluation:\n  primary_metric: accuracy\n")

        node = HypothesisNode(description="novel hypothesis")
        score_hypothesis(node, str(log_path), str(config_path))
        assert "overall" in node.critique_scores
        assert "novelty" in node.critique_scores
        assert "feasibility" in node.critique_scores
        assert "impact" in node.critique_scores
        assert "verdict" in node.critique_scores

    def test_score_with_experiment_history(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        experiments = [
            _exp("exp-001", "increase max_depth", "discarded", 0.78),
            _exp("exp-002", "decrease learning rate", "kept", 0.85),
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in experiments))
        config_path = tmp_path / "config.yaml"
        config_path.write_text("evaluation:\n  primary_metric: accuracy\n")

        # Near-duplicate of a discarded experiment should score lower
        node_dup = HypothesisNode(description="increase max_depth to higher value")
        score_dup = score_hypothesis(node_dup, str(log_path), str(config_path))

        node_novel = HypothesisNode(description="switch to completely different ensemble method")
        score_novel = score_hypothesis(node_novel, str(log_path), str(config_path))

        # Novel hypothesis should score higher than near-duplicate of failure
        assert score_novel > score_dup


# --- Greedy Search ---


class TestGreedySearch:
    def test_returns_results(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  type: xgboost\nevaluation:\n  primary_metric: accuracy\n")

        config = _config("xgboost", "accuracy")
        seeds = generate_seed_hypotheses(config, [])

        results = run_greedy_search(
            seeds,
            str(log_path),
            str(config_path),
            iterations=3,
            top_k=3,
            children_per_node=2,
        )
        assert len(results) > 0
        assert len(results) <= 3

    def test_results_are_ranked(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  type: xgboost\nevaluation:\n  primary_metric: accuracy\n")

        config = _config("xgboost", "accuracy")
        seeds = generate_seed_hypotheses(config, [])

        results = run_greedy_search(
            seeds,
            str(log_path),
            str(config_path),
            iterations=5,
            top_k=5,
            children_per_node=2,
        )
        # Results should be in descending score order
        scores = [r.critique_scores.get("overall", 0) for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_results_are_deduplicated(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  type: xgboost\nevaluation:\n  primary_metric: accuracy\n")

        config = _config("xgboost", "accuracy")
        seeds = generate_seed_hypotheses(config, [])

        results = run_greedy_search(
            seeds,
            str(log_path),
            str(config_path),
            iterations=5,
            top_k=10,
            children_per_node=2,
        )
        descriptions = [r.description.lower().strip() for r in results]
        assert len(descriptions) == len(set(descriptions))

    def test_zero_iterations_returns_seeds(self, tmp_path):
        log_path = tmp_path / "log.jsonl"
        log_path.write_text("")
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  type: xgboost\nevaluation:\n  primary_metric: accuracy\n")

        config = _config("xgboost", "accuracy")
        seeds = generate_seed_hypotheses(config, [])

        results = run_greedy_search(
            seeds,
            str(log_path),
            str(config_path),
            iterations=0,
            top_k=3,
        )
        # With zero iterations, should still return scored seeds
        assert len(results) > 0


# --- Output Formatting ---


class TestFormatting:
    def test_format_results_basic(self):
        nodes = [
            HypothesisNode(description="try LightGBM", depth=0),
            HypothesisNode(description="add polynomial features", depth=1),
        ]
        nodes[0].critique_scores = {"overall": 7.5, "verdict": "proceed", "novelty": 8, "feasibility": 9, "impact": 6}
        nodes[1].critique_scores = {"overall": 6.0, "verdict": "proceed", "novelty": 7, "feasibility": 7, "impact": 5}

        output = format_results(nodes, "accuracy", "greedy", 10)
        assert "TreeQuest Hypothesis Exploration" in output
        assert "try LightGBM" in output
        assert "add polynomial features" in output
        assert "7.5" in output
        assert "Nodes explored: 10" in output

    def test_format_results_shows_depth(self):
        node = HypothesisNode(description="refined idea", depth=2)
        node.critique_scores = {"overall": 5.0, "verdict": "modify", "novelty": 5, "feasibility": 5, "impact": 5}
        output = format_results([node], "accuracy", "abmcts-a", 20)
        assert "Depth: 2" in output

    def test_format_results_empty(self):
        output = format_results([], "accuracy", "greedy", 0)
        assert "Top 0 hypotheses" in output

    def test_results_to_json(self):
        nodes = [
            HypothesisNode(description="test", model_type="xgboost", depth=1),
        ]
        nodes[0].critique_scores = {"overall": 8.0}
        result = results_to_json(nodes)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["description"] == "test"
        assert result[0]["critique_scores"]["overall"] == 8.0

    def test_results_to_json_is_serializable(self):
        nodes = [
            HypothesisNode(
                description="full node",
                model_type="lightgbm",
                hyperparameters={"lr": 0.01},
                feature_changes={"add": ["poly"]},
                depth=3,
            ),
        ]
        nodes[0].critique_scores = {"overall": 7.0, "verdict": "proceed"}
        result = results_to_json(nodes)
        # Should be JSON-serializable without error
        serialized = json.dumps(result)
        assert '"full node"' in serialized
