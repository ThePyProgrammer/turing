"""Legacy commands compatibility layout tests.

The editable command source is skills/turing/. The legacy commands/ tree is
generated from skills/turing/ so existing packaging, docs links, and transition
tooling can continue to see the old command filenames.
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
SYNC_SCRIPT = REPO_ROOT / "src" / "sync-commands-layout.js"


def load_registry() -> dict[str, Any]:
    data = yaml.safe_load(REGISTRY_PATH.read_text())
    assert isinstance(data, dict)
    assert isinstance(data.get("commands"), dict)
    return data


def test_router_command_compat_matches_skill_source() -> None:
    assert (COMMANDS_DIR / "turing.md").read_text() == (SKILLS_DIR / "SKILL.md").read_text()


def test_registered_command_compat_files_match_skill_sources() -> None:
    registry = load_registry()

    for command_name in registry["commands"]:
        source = SKILLS_DIR / command_name / "SKILL.md"
        compat = COMMANDS_DIR / f"{command_name}.md"
        assert source.is_file(), f"missing skills/turing/{command_name}/SKILL.md"
        assert compat.is_file(), f"missing commands/{command_name}.md"
        assert compat.read_text() == source.read_text(), command_name


def test_rule_command_compat_matches_skill_source() -> None:
    assert (COMMANDS_DIR / "rules" / "loop-protocol.md").read_text() == (
        SKILLS_DIR / "rules" / "loop-protocol.md"
    ).read_text()


def test_skills_layout_has_exactly_registered_command_directories() -> None:
    registry = load_registry()
    expected_dirs = set(registry["commands"]) | {"rules"}
    actual_dirs = {
        path.name
        for path in SKILLS_DIR.iterdir()
        if path.is_dir()
    }

    assert actual_dirs == expected_dirs


def test_commands_compat_layout_has_exactly_registered_command_files() -> None:
    registry = load_registry()
    expected_files = {f"{command_name}.md" for command_name in registry["commands"]} | {"turing.md"}
    actual_files = {
        path.name
        for path in COMMANDS_DIR.glob("*.md")
    }

    assert actual_files == expected_files


def test_sync_commands_layout_check_rejects_stale_empty_directories(tmp_path: Path) -> None:
    stale_dir = COMMANDS_DIR / "stale-empty"
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
    assert "stale compatibility path commands/stale-empty" in result.stderr


def test_sync_commands_layout_check_passes() -> None:
    result = subprocess.run(
        ["node", str(SYNC_SCRIPT), "--check"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "commands compatibility tree is in sync" in result.stdout
