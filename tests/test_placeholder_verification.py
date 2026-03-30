"""Tests for placeholder verification (verify_placeholders.py).

Tier 2: Verifies that the placeholder detection system works correctly,
catching unreplaced markers before they produce silent runtime failures.
"""

from __future__ import annotations

from pathlib import Path

from scripts.verify_placeholders import scan_file, scan_directory, KNOWN_PLACEHOLDERS


def test_scan_file_detects_placeholders(tmp_path: Path):
    """Should detect {{PLACEHOLDER}} markers in files."""
    f = tmp_path / "test.py"
    f.write_text('metric = "{{TARGET_METRIC}}"\nname = "{{PROJECT_NAME}}"')
    findings = scan_file(f)
    assert len(findings) == 2
    assert findings[0][1] == "TARGET_METRIC"
    assert findings[1][1] == "PROJECT_NAME"


def test_scan_file_clean(tmp_path: Path):
    """Files without placeholders should return empty list."""
    f = tmp_path / "test.py"
    f.write_text('metric = "accuracy"\nname = "my-project"')
    findings = scan_file(f)
    assert findings == []


def test_scan_file_ignores_unknown_placeholders(tmp_path: Path):
    """Only known placeholders should be detected — not arbitrary {{text}}."""
    f = tmp_path / "test.py"
    f.write_text('template = "{{UNKNOWN_THING}}"')
    findings = scan_file(f)
    assert findings == []


def test_scan_file_returns_line_numbers(tmp_path: Path):
    """Findings should include correct line numbers."""
    f = tmp_path / "test.py"
    f.write_text('line1\nline2\nmetric = "{{TARGET_METRIC}}"\nline4')
    findings = scan_file(f)
    assert len(findings) == 1
    assert findings[0][0] == 3  # line number


def test_scan_directory_finds_all(tmp_path: Path):
    """Should find placeholders across multiple files."""
    (tmp_path / "a.py").write_text('x = "{{PROJECT_NAME}}"')
    (tmp_path / "b.yaml").write_text('metric: "{{TARGET_METRIC}}"')
    (tmp_path / "c.txt").write_text("clean file")
    results = scan_directory(tmp_path)
    assert len(results) == 2  # a.py and b.yaml


def test_scan_directory_skips_venv(tmp_path: Path):
    """Should skip .venv directory."""
    venv_dir = tmp_path / ".venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "test.py").write_text('x = "{{PROJECT_NAME}}"')
    (tmp_path / "main.py").write_text('x = "{{PROJECT_NAME}}"')
    results = scan_directory(tmp_path)
    assert len(results) == 1  # only main.py


def test_scan_directory_skips_non_scannable(tmp_path: Path):
    """Should skip binary and non-scannable file extensions."""
    (tmp_path / "model.joblib").write_text("binary data {{PROJECT_NAME}}")
    (tmp_path / "test.py").write_text('x = "{{PROJECT_NAME}}"')
    results = scan_directory(tmp_path)
    assert len(results) == 1  # only test.py


def test_known_placeholders_complete():
    """All 6 documented placeholders should be in KNOWN_PLACEHOLDERS."""
    expected = {
        "PROJECT_NAME",
        "TARGET_METRIC",
        "TASK_DESCRIPTION",
        "ML_DIR",
        "DATA_SOURCE",
        "METRIC_DIRECTION",
    }
    assert KNOWN_PLACEHOLDERS == expected
