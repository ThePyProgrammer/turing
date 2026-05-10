"""Tests for unified scaffolding (scaffold.py).

ADR-0016: Verifies the single scaffolding implementation that both
/turing:init and turing-init.sh delegate to.
"""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

from scripts import scaffold

REPO_ROOT = Path(__file__).parent.parent
from scripts.scaffold import (
    replace_placeholders,
    scaffold_project,
    verify_placeholders,
    make_command_hook_group,
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


def test_lower_metric_direction_renders_lower_is_better_true():
    """A lower-is-better metric direction should render true in config.yaml."""
    templates_dir = Path(__file__).parent.parent / "templates"
    values = scaffold.derive_values({
        "project_name": "cost-model",
        "target_metric": "mae",
        "task_description": "Predict cost",
        "ml_dir": "ml/cost-model",
        "data_source": "data/cost.csv",
        "metric_direction": "lower",
    })

    result = replace_placeholders((templates_dir / "config.yaml").read_text(), values)

    assert "lower_is_better: true" in result


def test_scaffold_scopes_memory_by_project_name(tmp_path: Path, monkeypatch):
    """Scaffolded agent memory should not collide across projects."""
    templates_dir = Path(__file__).parent.parent / "templates"
    values = scaffold.derive_values({
        "project_name": "Foo Model!",
        "target_metric": "accuracy",
        "task_description": "Predict labels",
        "ml_dir": "ml/foo",
        "data_source": "data/foo.csv",
        "metric_direction": "higher",
    })
    monkeypatch.chdir(tmp_path)

    scaffold_project(
        templates_dir=templates_dir,
        ml_dir=values["ml_dir"],
        values=values,
        setup_venv=False,
        setup_hooks=False,
    )

    assert (tmp_path / ".claude" / "agent-memory" / "ml-researcher-Foo-Model" / "MEMORY.md").exists()
    assert not (tmp_path / ".claude" / "agent-memory" / "ml-researcher" / "MEMORY.md").exists()


def test_scaffold_accepts_raw_values_for_derived_placeholders(tmp_path: Path, monkeypatch):
    """Direct callers should not need to precompute scaffold-derived values."""
    templates_dir = Path(__file__).parent.parent / "templates"
    values = {
        "project_name": "raw-values",
        "target_metric": "mae",
        "task_description": "Predict labels",
        "ml_dir": "ml/raw",
        "data_source": "data/raw.csv",
        "metric_direction": "lower",
    }
    monkeypatch.chdir(tmp_path)

    scaffold_project(
        templates_dir=templates_dir,
        ml_dir=values["ml_dir"],
        values=values,
        setup_venv=False,
        setup_hooks=False,
    )

    config = (tmp_path / "ml" / "raw" / "config.yaml").read_text()
    assert "lower_is_better: true" in config
    assert (tmp_path / ".claude" / "agent-memory" / "ml-researcher-raw-values" / "MEMORY.md").exists()


def test_scaffolded_program_points_at_scoped_memory(tmp_path: Path, monkeypatch):
    """Generated program.md should tell the agent to read the scoped memory path."""
    templates_dir = Path(__file__).parent.parent / "templates"
    values = scaffold.derive_values({
        "project_name": "Foo Model!",
        "target_metric": "accuracy",
        "task_description": "Predict labels",
        "ml_dir": "ml/foo",
        "data_source": "data/foo.csv",
        "metric_direction": "higher",
    })
    monkeypatch.chdir(tmp_path)

    scaffold_project(
        templates_dir=templates_dir,
        ml_dir=values["ml_dir"],
        values=values,
        setup_venv=False,
        setup_hooks=False,
    )

    program = (tmp_path / "ml" / "foo" / "program.md").read_text()
    assert ".claude/agent-memory/ml-researcher-Foo-Model/MEMORY.md" in program
    assert ".claude/agent-memory/ml-researcher/MEMORY.md" not in program


def test_make_command_hook_group():
    group = make_command_hook_group("bash ml/demo/scripts/stop-hook.sh")

    assert group == {
        "matcher": "",
        "hooks": [{"type": "command", "command": "bash ml/demo/scripts/stop-hook.sh"}],
    }


def test_scaffold_configures_post_tool_use_and_stop_hook_groups(tmp_path: Path, monkeypatch):
    templates_dir = Path(__file__).parent.parent / "templates"
    values = scaffold.derive_values({
        "project_name": "Foo Model!",
        "target_metric": "accuracy",
        "task_description": "Predict labels",
        "ml_dir": "ml/foo",
        "data_source": "data/foo.csv",
        "metric_direction": "higher",
    })
    monkeypatch.chdir(tmp_path)

    scaffold_project(
        templates_dir=templates_dir,
        ml_dir=values["ml_dir"],
        values=values,
        setup_venv=False,
        setup_hooks=True,
    )

    settings = json.loads((tmp_path / ".claude" / "settings.local.json").read_text())
    assert settings["hooks"]["PostToolUse"] == [{
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "bash ml/foo/scripts/post-train-hook.sh"}],
    }]
    assert settings["hooks"]["Stop"] == [{
        "matcher": "",
        "hooks": [{"type": "command", "command": "bash ml/foo/scripts/stop-hook.sh"}],
    }]


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


def test_find_templates_dir_checks_env_override(tmp_path: Path, monkeypatch):
    """TURING_TEMPLATES_DIR should override discovered locations."""
    templates = tmp_path / "custom" / "templates"
    templates.mkdir(parents=True)
    (templates / "prepare.py").write_text("")
    monkeypatch.setenv("TURING_TEMPLATES_DIR", str(templates))
    monkeypatch.setattr(scaffold, "__file__", str(tmp_path / "not-installed" / "scripts" / "scaffold.py"))

    assert scaffold.find_templates_dir() == templates


def test_find_templates_dir_checks_script_relative_path(tmp_path: Path, monkeypatch):
    """A script inside templates/scripts should find its parent templates directory."""
    templates = tmp_path / "templates"
    scripts = templates / "scripts"
    scripts.mkdir(parents=True)
    (templates / "prepare.py").write_text("")
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.delenv("TURING_TEMPLATES_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(outside)
    monkeypatch.setattr(scaffold, "__file__", str(scripts / "scaffold.py"))

    assert scaffold.find_templates_dir() == templates


def test_find_templates_dir_checks_project_installed_path(tmp_path: Path, monkeypatch):
    """Project-scoped installs should be discoverable from the current project."""
    project_templates = tmp_path / ".claude" / "commands" / "turing" / "templates"
    project_templates.mkdir(parents=True)
    (project_templates / "prepare.py").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(scaffold, "__file__", str(tmp_path / "not-installed" / "scripts" / "scaffold.py"))

    assert scaffold.find_templates_dir() == project_templates


def test_find_templates_dir_checks_global_installed_path(tmp_path: Path, monkeypatch):
    """Global command-pack templates should be discoverable."""
    global_templates = tmp_path / "home" / ".claude" / "commands" / "turing" / "templates"
    global_templates.mkdir(parents=True)
    (global_templates / "prepare.py").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(scaffold, "__file__", str(tmp_path / "not-installed" / "scripts" / "scaffold.py"))

    assert scaffold.find_templates_dir() == global_templates


def test_find_templates_dir_checks_plugin_path(tmp_path: Path, monkeypatch):
    """Legacy plugin templates should remain a fallback."""
    plugin_templates = tmp_path / "home" / ".claude" / "plugins" / "claude-turing" / "templates"
    plugin_templates.mkdir(parents=True)
    (plugin_templates / "prepare.py").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(scaffold, "__file__", str(tmp_path / "not-installed" / "scripts" / "scaffold.py"))

    assert scaffold.find_templates_dir() == plugin_templates


def test_find_templates_dir_checks_node_modules_path(tmp_path: Path, monkeypatch):
    """npm-installed templates should be discoverable from cwd."""
    node_templates = tmp_path / "node_modules" / "claude-turing" / "templates"
    node_templates.mkdir(parents=True)
    (node_templates / "prepare.py").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(scaffold, "__file__", str(tmp_path / "not-installed" / "scripts" / "scaffold.py"))

    assert scaffold.find_templates_dir() == node_templates


def test_scaffold_cli_accepts_templates_dir_override(tmp_path: Path):
    """--templates-dir should let CLI scaffolding run without discovery."""
    target = tmp_path / "ml" / "override"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "templates" / "scripts" / "scaffold.py"),
            "--project-name", "override",
            "--target-metric", "accuracy",
            "--metric-direction", "higher",
            "--task-description", "Predict labels",
            "--ml-dir", str(target),
            "--data-source", "data/train.csv",
            "--templates-dir", str(REPO_ROOT / "templates"),
            "--no-venv",
            "--no-hooks",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (target / "config.yaml").exists()


# --- PLACEHOLDER_MAP ---


def test_placeholder_map_complete():
    """All documented placeholders should be in the map."""
    expected = {
        "PROJECT_NAME",
        "TARGET_METRIC",
        "TASK_DESCRIPTION",
        "ML_DIR",
        "DATA_SOURCE",
        "METRIC_DIRECTION",
        "LOWER_IS_BETTER",
        "MEMORY_DIR_NAME",
    }
    assert set(PLACEHOLDER_MAP.keys()) == expected
