"""Tests for harness self-diagnosis (harness_doctor.py).

Phase 29.2: Verifies each check category, --fix mode, scoring, formatting.
"""

from __future__ import annotations

import json

import pytest

from scripts.harness_doctor import (
    check_environment,
    check_dependencies,
    check_config,
    check_experiment_log,
    check_scripts,
    check_disk_space,
    check_git_state,
    check_claude_hooks,
    fix_claude_hooks,
    fix_corrupt_log,
    run_doctor,
    format_doctor_report,
)


# --- check_environment ---

def test_environment_pass():
    result = check_environment()
    assert result["status"] in ("PASS", "WARN")
    assert "Python" in result["detail"]


# --- check_dependencies ---

def test_deps_all_present():
    result = check_dependencies(["json", "os", "sys"])
    assert result["status"] == "PASS"

def test_deps_missing():
    result = check_dependencies(["nonexistent_package_xyz"])
    assert result["status"] == "FAIL"
    assert "fix" in result

def test_deps_default():
    result = check_dependencies()
    assert result["status"] in ("PASS", "FAIL")


# --- check_config ---

def test_config_valid(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    result = check_config(str(config))
    assert result["status"] == "PASS"

def test_config_missing(tmp_path):
    result = check_config(str(tmp_path / "nonexistent.yaml"))
    assert result["status"] == "FAIL"

def test_config_invalid_yaml(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("not: valid: yaml: [")
    result = check_config(str(config))
    assert result["status"] == "FAIL"

def test_config_missing_fields(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("data:\n  source: test\n")
    result = check_config(str(config))
    assert result["status"] == "WARN"


# --- check_experiment_log ---

def test_log_valid(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"metrics": {"accuracy": 0.85}}\n{"metrics": {"accuracy": 0.87}}\n')
    result = check_experiment_log(str(log))
    assert result["status"] == "PASS"
    assert "2/2" in result["detail"]

def test_log_corrupt(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"metrics": {"accuracy": 0.85}}\nCORRUPT LINE\n{"metrics": {"accuracy": 0.87}}\n')
    result = check_experiment_log(str(log))
    assert result["status"] == "FAIL"
    assert len(result["corrupt_lines"]) == 1

def test_log_missing(tmp_path):
    result = check_experiment_log(str(tmp_path / "missing.jsonl"))
    assert result["status"] == "WARN"

def test_log_missing_metrics(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"experiment_id": "exp-001"}\n')
    result = check_experiment_log(str(log))
    assert result["status"] == "WARN"


# --- check_scripts ---

def test_scripts_valid(tmp_path):
    for name in ["train.py", "prepare.py", "evaluate.py"]:
        (tmp_path / name).write_text("def main(): pass\n")
    result = check_scripts(str(tmp_path))
    assert result["status"] == "PASS"

def test_scripts_missing(tmp_path):
    result = check_scripts(str(tmp_path))
    assert result["status"] == "FAIL"

def test_scripts_syntax_error(tmp_path):
    (tmp_path / "train.py").write_text("def main(:\n")  # Syntax error
    (tmp_path / "prepare.py").write_text("x = 1\n")
    (tmp_path / "evaluate.py").write_text("x = 2\n")
    result = check_scripts(str(tmp_path))
    assert result["status"] == "WARN"
    assert any("syntax" in i.lower() for i in result["issues"])


# --- check_disk_space ---

def test_disk_space():
    result = check_disk_space()
    assert result["status"] in ("PASS", "FAIL")
    assert "free_mb" in result


# --- check_git_state ---

def test_git_state():
    result = check_git_state()
    assert result["status"] in ("PASS", "WARN")


# --- check_claude_hooks ---


def test_claude_hooks_valid(tmp_path):
    settings = tmp_path / "settings.local.json"
    settings.write_text(json.dumps({
        "hooks": {
            "PostToolUse": [{
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": "bash ml/demo/scripts/post-train-hook.sh"}],
            }],
            "Stop": [{
                "matcher": "",
                "hooks": [{"type": "command", "command": "bash ml/demo/scripts/stop-hook.sh"}],
            }],
        }
    }))

    result = check_claude_hooks(str(settings))

    assert result["status"] == "PASS"
    assert result["fixable"] is False


def test_claude_hooks_legacy_bare_command_is_fixable(tmp_path):
    settings = tmp_path / "settings.local.json"
    settings.write_text(json.dumps({
        "hooks": {
            "Stop": [{"type": "command", "command": "bash ml/demo/scripts/stop-hook.sh"}],
        }
    }))

    result = check_claude_hooks(str(settings))

    assert result["status"] == "FAIL"
    assert result["fixable"] is True
    assert "legacy bare command hook shape" in result["issues"][0]


def test_fix_claude_hooks_migrates_legacy_shape_and_preserves_valid_hooks(tmp_path):
    settings = tmp_path / "settings.local.json"
    settings.write_text(json.dumps({
        "hooks": {
            "PostToolUse": [{
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": "bash ml/demo/scripts/post-train-hook.sh"}],
            }],
            "Stop": [{"type": "command", "command": "bash ml/demo/scripts/stop-hook.sh"}],
        }
    }))

    result = fix_claude_hooks(str(settings))
    updated = json.loads(settings.read_text())

    assert result["fixed"] is True
    assert result["migrated"] == 1
    assert settings.with_suffix(".json.bak").exists()
    assert updated["hooks"]["PostToolUse"] == [{
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "bash ml/demo/scripts/post-train-hook.sh"}],
    }]
    assert updated["hooks"]["Stop"] == [{
        "matcher": "",
        "hooks": [{"type": "command", "command": "bash ml/demo/scripts/stop-hook.sh"}],
    }]
    assert check_claude_hooks(str(settings))["status"] == "PASS"


def test_claude_hooks_invalid_json_reports_failure_without_mutation(tmp_path):
    settings = tmp_path / "settings.local.json"
    settings.write_text("{")

    check = check_claude_hooks(str(settings))
    result = fix_claude_hooks(str(settings))

    assert check["status"] == "FAIL"
    assert result["fixed"] is False
    assert settings.read_text() == "{"
    assert not settings.with_suffix(".json.bak").exists()


# --- fix_corrupt_log ---

def test_fix_removes_corrupt(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"good": 1}\nBAD LINE\n{"good": 2}\n')
    result = fix_corrupt_log(str(log))
    assert result["fixed"]
    assert result["removed"] == 1
    # Verify backup exists
    assert (tmp_path / "log.jsonl.bak").exists()
    # Verify clean log
    with open(log) as f:
        lines = f.readlines()
    assert len(lines) == 2

def test_fix_no_corrupt(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"good": 1}\n')
    result = fix_corrupt_log(str(log))
    assert not result["fixed"]

def test_fix_missing_log():
    result = fix_corrupt_log("/nonexistent/log.jsonl")
    assert not result["fixed"]


# --- run_doctor ---

def test_doctor_basic(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"metrics": {"accuracy": 0.85}}\n')
    report = run_doctor(config_path=str(config), log_path=str(log))
    assert "checks" in report
    assert "score" in report
    assert report["score"]["total"] == 8

def test_doctor_with_fix(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    log = tmp_path / "log.jsonl"
    log.write_text('{"good": 1}\nBAD\n')
    report = run_doctor(config_path=str(config), log_path=str(log), fix=True)
    assert len(report["fixes_applied"]) > 0

def test_doctor_overall_healthy(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("evaluation:\n  primary_metric: accuracy\n")
    report = run_doctor(config_path=str(config))
    assert report["overall"] in ("HEALTHY", "DEGRADED", "UNHEALTHY")


# --- format_doctor_report ---

def test_format_report():
    report = {
        "checks": [
            {"name": "Python env", "status": "PASS", "detail": "3.12", "issues": []},
            {"name": "Disk", "status": "FAIL", "detail": "500 MB", "issues": ["Low disk"], "fix": "archive"},
        ],
        "score": {"passed": 1, "warned": 0, "failed": 1, "total": 2},
        "overall": "UNHEALTHY",
        "fixes_applied": [],
        "generated_at": "2026-04-01",
    }
    text = format_doctor_report(report)
    assert "✓ PASS" in text
    assert "✗ FAIL" in text
    assert "UNHEALTHY" in text

def test_format_with_fixes():
    report = {
        "checks": [{"name": "Log", "status": "PASS", "detail": "clean", "issues": []}],
        "score": {"passed": 1, "warned": 0, "failed": 0, "total": 1},
        "overall": "HEALTHY",
        "fixes_applied": ["Removed 3 corrupt lines"],
        "generated_at": "2026-04-01",
    }
    text = format_doctor_report(report)
    assert "Removed 3" in text
