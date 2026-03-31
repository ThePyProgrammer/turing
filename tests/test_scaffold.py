"""Tests for unified scaffolding (scaffold.py).

ADR-0016: Verifies the single scaffolding implementation that both
/turing:init and turing-init.sh delegate to.
"""

from __future__ import annotations

from pathlib import Path

from scripts.scaffold import (
    replace_placeholders,
    verify_placeholders,
    PLACEHOLDER_MAP,
)


# --- replace_placeholders ---


def test_replace_all_placeholders():
    """All 6 placeholders should be replaced."""
    text = "{{PROJECT_NAME}} {{TARGET_METRIC}} {{TASK_DESCRIPTION}} {{ML_DIR}} {{DATA_SOURCE}} {{METRIC_DIRECTION}}"
    values = {
        "project_name": "sentiment",
        "target_metric": "accuracy",
        "task_description": "Predict sentiment",
        "ml_dir": "ml/sentiment",
        "data_source": "data/reviews.csv",
        "metric_direction": "higher",
    }
    result = replace_placeholders(text, values)
    assert "{{" not in result
    assert "sentiment" in result
    assert "accuracy" in result
    assert "Predict sentiment" in result


def test_replace_preserves_non_placeholders():
    """Text without placeholders should be unchanged."""
    text = "This has no placeholders at all."
    result = replace_placeholders(text, {"project_name": "test"})
    assert result == text


def test_replace_multiple_occurrences():
    """Same placeholder appearing multiple times should all be replaced."""
    text = "{{PROJECT_NAME}} is {{PROJECT_NAME}}"
    result = replace_placeholders(text, {"project_name": "churn"})
    assert result == "churn is churn"


def test_replace_empty_value():
    """Empty string value should replace placeholder with empty string."""
    text = "name: {{PROJECT_NAME}}"
    result = replace_placeholders(text, {"project_name": ""})
    assert result == "name: "


# --- verify_placeholders ---


def test_verify_clean(tmp_path: Path):
    """No unreplaced placeholders should return empty list."""
    (tmp_path / "test.py").write_text('metric = "accuracy"')
    findings = verify_placeholders(str(tmp_path))
    assert findings == []


def test_verify_detects_unreplaced(tmp_path: Path):
    """Unreplaced placeholders should be found."""
    (tmp_path / "test.py").write_text('metric = "{{TARGET_METRIC}}"')
    findings = verify_placeholders(str(tmp_path))
    assert len(findings) == 1
    assert findings[0][2] == "TARGET_METRIC"


def test_verify_skips_venv(tmp_path: Path):
    """Files in .venv should be skipped."""
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "test.py").write_text('x = "{{PROJECT_NAME}}"')
    findings = verify_placeholders(str(tmp_path))
    assert findings == []


def test_verify_reports_line_numbers(tmp_path: Path):
    """Findings should include correct line numbers."""
    (tmp_path / "test.py").write_text('line1\nline2\nx = "{{PROJECT_NAME}}"\nline4')
    findings = verify_placeholders(str(tmp_path))
    assert findings[0][1] == 3


def test_verify_multiple_files(tmp_path: Path):
    """Should find placeholders across multiple files."""
    (tmp_path / "a.py").write_text('x = "{{PROJECT_NAME}}"')
    (tmp_path / "b.yaml").write_text('metric: "{{TARGET_METRIC}}"')
    findings = verify_placeholders(str(tmp_path))
    assert len(findings) == 2


# --- PLACEHOLDER_MAP ---


def test_placeholder_map_complete():
    """All 6 documented placeholders should be in the map."""
    expected = {"PROJECT_NAME", "TARGET_METRIC", "TASK_DESCRIPTION", "ML_DIR", "DATA_SOURCE", "METRIC_DIRECTION"}
    assert set(PLACEHOLDER_MAP.keys()) == expected
