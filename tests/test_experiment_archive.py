"""Tests for experiment lifecycle cleanup (experiment_archive.py). Phase 24.3."""
from __future__ import annotations
import pytest
from scripts.experiment_archive import (
    identify_archivable, format_archive_report,
)

EXPS = [
    {"experiment_id": f"exp-{i:03d}", "status": "kept" if i > 5 else "discarded",
     "metrics": {"accuracy": 0.70 + i * 0.01},
     "timestamp": f"2026-01-{i+1:02d}T00:00:00"}
    for i in range(10)
]

def test_identify_archivable():
    archivable, protected = identify_archivable(EXPS, "accuracy", False, keep_best=3, older_than_days=0)
    arch_ids = {e["experiment_id"] for e in archivable}
    assert "exp-009" not in arch_ids  # Best — protected
    assert "exp-009" in protected

def test_identify_archivable_empty():
    archivable, protected = identify_archivable([], "accuracy", False)
    assert archivable == []

def test_identify_protects_best():
    archivable, protected = identify_archivable(EXPS, "accuracy", False, keep_best=3, older_than_days=0)
    # Top 3 by accuracy (exp-009, exp-008, exp-007) should be protected
    assert "exp-009" in protected
    assert "exp-008" in protected

def test_format_report_basic():
    report = {
        "generated_at": "2026-04-01T00:00:00",
        "dry_run": False,
        "archived": [{"experiment_id": "exp-001"}],
        "preserved": [{"experiment_id": "exp-009"}],
        "n_archived": 1, "n_preserved": 1,
        "total_experiments": 10,
        "space_saved_bytes": 5000000,
    }
    text = format_archive_report(report)
    assert "rchive" in text  # Archive or archive

def test_format_error():
    assert "ERROR" in format_archive_report({"error": "fail"})
