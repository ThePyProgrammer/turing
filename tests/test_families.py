"""Tests for experiment family viewer (show_families.py).

Phase 6.3: Strategic grouping of experiments.
"""

from __future__ import annotations

from scripts.show_families import family_stats, group_by_family


def _exp(exp_id: str, status: str = "kept", accuracy: float = 0.85, family: str | None = None) -> dict:
    return {"experiment_id": exp_id, "status": status, "metrics": {"accuracy": accuracy}, "family": family}


def test_group_by_family():
    exps = [
        _exp("exp-001", family="optimizer"),
        _exp("exp-002", family="optimizer"),
        _exp("exp-003", family="architecture"),
        _exp("exp-004"),  # untagged
    ]
    groups = group_by_family(exps)
    assert len(groups["optimizer"]) == 2
    assert len(groups["architecture"]) == 1
    assert len(groups["untagged"]) == 1


def test_family_stats_basic():
    exps = [
        _exp("exp-001", "kept", 0.80, "opt"),
        _exp("exp-002", "discarded", 0.75, "opt"),
        _exp("exp-003", "kept", 0.85, "opt"),
    ]
    stats = family_stats(exps, "accuracy", False)
    assert stats["total"] == 3
    assert stats["kept"] == 2
    assert stats["discarded"] == 1
    assert stats["best_metric"] == 0.85
    assert stats["best_experiment"] == "exp-003"


def test_family_exhausted():
    exps = [
        _exp("exp-001", "discarded", 0.70),
        _exp("exp-002", "discarded", 0.69),
        _exp("exp-003", "discarded", 0.68),
    ]
    stats = family_stats(exps, "accuracy", False)
    assert stats["exhausted"] is True


def test_family_not_exhausted():
    exps = [
        _exp("exp-001", "kept", 0.85),
        _exp("exp-002", "discarded", 0.80),
        _exp("exp-003", "kept", 0.87),
    ]
    stats = family_stats(exps, "accuracy", False)
    assert stats["exhausted"] is False


def test_family_empty():
    stats = family_stats([], "accuracy", False)
    assert stats["total"] == 0
    assert stats["keep_rate"] == 0


def test_family_keep_rate():
    exps = [_exp(f"exp-{i:03d}", "kept" if i % 2 == 0 else "discarded") for i in range(10)]
    stats = family_stats(exps, "accuracy", False)
    assert stats["keep_rate"] == 0.5
