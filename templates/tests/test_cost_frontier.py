"""Tests for the cost-performance frontier analysis module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.cost_frontier import (
    CostRecord,
    compute_cost_efficiency,
    compute_pareto_frontier,
    format_cost_report,
    load_cost_data,
)


def _make_record(exp_id: str, metric: float, seconds: float, model: str = "xgboost") -> CostRecord:
    """Helper to create a CostRecord."""
    return CostRecord(
        experiment_id=exp_id,
        metric_value=metric,
        train_seconds=seconds,
        model_type=model,
    )


def _write_log(tmp_path: Path, entries: list[dict]) -> Path:
    """Write experiment entries to a JSONL file and return the path."""
    log_path = tmp_path / "log.jsonl"
    with open(log_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return log_path


class TestParetoFrontierBasic:
    """test_pareto_frontier_basic — 3 experiments, identify the efficient frontier."""

    def test_three_experiments(self):
        data = [
            _make_record("exp-001", 0.85, 3.0, "xgboost"),
            _make_record("exp-002", 0.87, 3.0, "xgboost"),
            _make_record("exp-003", 0.89, 2400.0, "neural_net"),
        ]
        frontier = compute_pareto_frontier(data, lower_is_better=False)
        frontier_ids = {r.experiment_id for r in frontier}

        # exp-002 dominates exp-001 (same time, better metric)
        assert "exp-001" not in frontier_ids
        # exp-002 and exp-003 are both on frontier (tradeoff)
        assert "exp-002" in frontier_ids
        assert "exp-003" in frontier_ids
        assert len(frontier) == 2


class TestParetoFrontierSingle:
    """test_pareto_frontier_single — 1 experiment is always on frontier."""

    def test_single_experiment(self):
        data = [_make_record("exp-001", 0.90, 10.0)]
        frontier = compute_pareto_frontier(data, lower_is_better=False)
        assert len(frontier) == 1
        assert frontier[0].experiment_id == "exp-001"

    def test_single_experiment_lower_is_better(self):
        data = [_make_record("exp-001", 0.15, 10.0)]
        frontier = compute_pareto_frontier(data, lower_is_better=True)
        assert len(frontier) == 1
        assert frontier[0].experiment_id == "exp-001"


class TestParetoFrontierDominated:
    """test_pareto_frontier_dominated — dominated experiment excluded."""

    def test_dominated_excluded(self):
        data = [
            _make_record("exp-001", 0.80, 100.0),  # slow AND worse
            _make_record("exp-002", 0.90, 10.0),    # fast AND better
        ]
        frontier = compute_pareto_frontier(data, lower_is_better=False)
        frontier_ids = {r.experiment_id for r in frontier}

        assert "exp-001" not in frontier_ids
        assert "exp-002" in frontier_ids
        assert len(frontier) == 1

    def test_dominated_lower_is_better(self):
        data = [
            _make_record("exp-001", 0.50, 100.0),  # slow AND worse (higher is worse)
            _make_record("exp-002", 0.10, 10.0),    # fast AND better (lower is better)
        ]
        frontier = compute_pareto_frontier(data, lower_is_better=True)
        frontier_ids = {r.experiment_id for r in frontier}

        assert "exp-001" not in frontier_ids
        assert "exp-002" in frontier_ids


class TestCostEfficiency:
    """test_cost_efficiency — metric improvement per second computed correctly."""

    def test_efficiency_computation(self):
        data = [
            _make_record("exp-001", 0.80, 10.0),   # baseline (worst)
            _make_record("exp-002", 0.85, 5.0),     # 0.05 improvement in 5s
            _make_record("exp-003", 0.90, 100.0),   # 0.10 improvement in 100s
        ]
        results = compute_cost_efficiency(data, lower_is_better=False)

        eff_map = {r["experiment_id"]: r for r in results}

        # exp-002: improvement=0.05, time=5s -> 0.01/s
        assert abs(eff_map["exp-002"]["metric_per_second"] - 0.01) < 1e-9
        # exp-003: improvement=0.10, time=100s -> 0.001/s
        assert abs(eff_map["exp-003"]["metric_per_second"] - 0.001) < 1e-9
        # exp-001 is baseline: improvement=0, so efficiency=0
        assert eff_map["exp-001"]["metric_per_second"] == 0.0

        # Should be sorted by efficiency descending
        assert results[0]["experiment_id"] == "exp-002"

    def test_efficiency_lower_is_better(self):
        data = [
            _make_record("exp-001", 0.50, 10.0),   # worst (highest)
            _make_record("exp-002", 0.30, 5.0),     # 0.20 improvement in 5s
        ]
        results = compute_cost_efficiency(data, lower_is_better=True)
        eff_map = {r["experiment_id"]: r for r in results}
        assert abs(eff_map["exp-002"]["metric_per_second"] - 0.04) < 1e-9


class TestFormatCostReport:
    """test_format_cost_report — report contains key experiments and summary."""

    def test_report_content(self):
        data = [
            _make_record("exp-001", 0.87, 3.0, "xgboost"),
            _make_record("exp-002", 0.89, 2400.0, "neural_net"),
        ]
        frontier = compute_pareto_frontier(data, lower_is_better=False)
        report = format_cost_report(data, frontier, "accuracy", lower_is_better=False)

        assert "exp-001" in report
        assert "exp-002" in report
        assert "xgboost" in report
        assert "neural_net" in report
        assert "Best accuracy" in report
        assert "Best cost-efficiency" in report
        assert "improvement costs" in report
        assert "Pareto" in report

    def test_report_with_single_experiment(self):
        data = [_make_record("exp-001", 0.90, 10.0)]
        frontier = compute_pareto_frontier(data, lower_is_better=False)
        report = format_cost_report(data, frontier, "accuracy")
        assert "exp-001" in report
        assert "Best accuracy" in report


class TestEmptyData:
    """test_empty_data — handles empty log gracefully."""

    def test_empty_pareto(self):
        assert compute_pareto_frontier([], lower_is_better=False) == []

    def test_empty_efficiency(self):
        assert compute_cost_efficiency([], lower_is_better=False) == []

    def test_empty_report(self):
        report = format_cost_report([], [], "accuracy")
        assert "No cost-performance data" in report

    def test_load_from_empty_file(self, tmp_path):
        log_path = tmp_path / "empty.jsonl"
        log_path.touch()
        records = load_cost_data(str(log_path), "accuracy")
        assert records == []

    def test_load_from_missing_file(self, tmp_path):
        records = load_cost_data(str(tmp_path / "nonexistent.jsonl"), "accuracy")
        assert records == []


class TestLoadCostData:
    """Test load_cost_data extracts records correctly from JSONL."""

    def test_load_with_train_seconds(self, tmp_path):
        entries = [
            {
                "experiment_id": "exp-001",
                "status": "kept",
                "config": {"model_type": "xgboost"},
                "metrics": {"accuracy": 0.87, "train_seconds": 3.2},
            },
            {
                "experiment_id": "exp-002",
                "status": "kept",
                "config": {"model_type": "neural_net"},
                "metrics": {"accuracy": 0.89, "train_seconds": 2400.0},
            },
            {
                "experiment_id": "exp-003",
                "status": "discarded",
                "config": {"model_type": "svm"},
                "metrics": {"accuracy": 0.50, "train_seconds": 1.0},
            },
            {
                "experiment_id": "exp-004",
                "status": "kept",
                "config": {"model_type": "rf"},
                "metrics": {"accuracy": 0.85},
            },
        ]
        log_path = _write_log(tmp_path, entries)
        records = load_cost_data(str(log_path), "accuracy")

        # exp-003 is discarded, exp-004 has no train_seconds
        assert len(records) == 2
        assert records[0].experiment_id == "exp-001"
        assert records[1].experiment_id == "exp-002"
