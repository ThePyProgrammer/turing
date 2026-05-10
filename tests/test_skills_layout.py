"""Modern skills layout mirror tests.

The editable command source remains commands/ in this migration stage, while
skills/turing/ is a package-included mirror for modern Claude Code conventions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS_DIR = REPO_ROOT / "commands"
SKILLS_DIR = REPO_ROOT / "skills" / "turing"
REGISTRY_PATH = REPO_ROOT / "config" / "commands.yaml"
SYNC_SCRIPT = REPO_ROOT / "src" / "sync-skills-layout.js"


def load_registry() -> dict[str, Any]:
    data = yaml.safe_load(REGISTRY_PATH.read_text())
    assert isinstance(data, dict)
    assert isinstance(data.get("commands"), dict)
    return data


def test_router_skill_mirror_matches_command_source() -> None:
    assert (SKILLS_DIR / "SKILL.md").read_text() == (COMMANDS_DIR / "turing.md").read_text()


def test_registered_command_skill_mirrors_match_command_sources() -> None:
    registry = load_registry()

    for command_name in registry["commands"]:
        source = COMMANDS_DIR / f"{command_name}.md"
        mirror = SKILLS_DIR / command_name / "SKILL.md"
        assert mirror.is_file(), f"missing skills/turing/{command_name}/SKILL.md"
        assert mirror.read_text() == source.read_text(), command_name


def test_rule_skill_mirror_matches_command_source() -> None:
    assert (SKILLS_DIR / "rules" / "loop-protocol.md").read_text() == (
        COMMANDS_DIR / "rules" / "loop-protocol.md"
    ).read_text()


def test_skills_layout_has_no_unregistered_command_directories() -> None:
    registry = load_registry()
    expected_dirs = set(registry["commands"]) | {"rules"}
    actual_dirs = {
        path.name
        for path in SKILLS_DIR.iterdir()
        if path.is_dir()
    }

    assert actual_dirs == expected_dirs


def test_sync_skills_layout_check_rejects_stale_empty_directories(tmp_path: Path) -> None:
    stale_dir = SKILLS_DIR / "stale-empty"
    stale_dir.mkdir()
    try:
        result = subprocess.run(
            ["node", str(SYNC_SCRIPT), "--check"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        stale_dir.rmdir()

    assert result.returncode != 0
    assert "stale mirror skills/turing/stale-empty" in result.stderr


def test_sync_skills_layout_check_passes() -> None:
    result = subprocess.run(
        ["node", str(SYNC_SCRIPT), "--check"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "skills/turing mirror is in sync" in result.stdout
