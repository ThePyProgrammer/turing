"""Manifest consistency tests.

Validates that the installer, verifier, scaffold, and filesystem
are all in sync. Catches the exact class of drift that caused
the v1.0.0 blockers — commands/configs added to the filesystem
but not to the deployment manifests.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
COMMANDS_DIR = REPO_ROOT / "commands"
CONFIG_DIR = REPO_ROOT / "config"
INSTALL_JS = REPO_ROOT / "src" / "install.js"
VERIFY_JS = REPO_ROOT / "src" / "verify.js"
CLI_JS = REPO_ROOT / "bin" / "cli.js"
SCAFFOLD_PY = REPO_ROOT / "templates" / "scripts" / "scaffold.py"
TEMPLATES_DIR = REPO_ROOT / "templates"


def _registry() -> dict:
    """Load the command registry."""
    return yaml.safe_load((CONFIG_DIR / "commands.yaml").read_text())


def _get_command_files() -> set[str]:
    """Get all command .md files (excluding router and rules/)."""
    files = set()
    for f in COMMANDS_DIR.glob("*.md"):
        name = f.stem
        if name != "turing":  # router is separate
            files.add(name)
    return files


def _get_config_files() -> set[str]:
    """Get all config files."""
    return {f.name for f in CONFIG_DIR.iterdir() if f.is_file()}


def _scaffold_template_files() -> set[str]:
    """Get all non-cache template files scaffold/install/verify must keep in sync."""
    files = set()
    for path in TEMPLATES_DIR.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if path.suffix == ".pyc":
            continue
        files.add(path.relative_to(REPO_ROOT).as_posix())
    return files


# --- Install manifest ---


EXPECTED_TEMPLATE_FILES = _scaffold_template_files()


def test_installer_copies_templates(tmp_path: Path):
    """Installer must deploy templates next to installed commands."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    result = subprocess.run(
        ["node", str(INSTALL_JS), "--global"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    install_root = tmp_path / ".claude" / "commands" / "turing"
    for relative_path in EXPECTED_TEMPLATE_FILES:
        assert (install_root / relative_path).exists(), f"missing {relative_path}"


def test_project_installer_copies_templates(tmp_path: Path):
    """Project-scoped installer must deploy the full template tree."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")

    result = subprocess.run(
        ["node", str(INSTALL_JS), "--project"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    install_root = tmp_path / ".claude" / "commands" / "turing"
    for relative_path in EXPECTED_TEMPLATE_FILES:
        assert (install_root / relative_path).exists(), f"missing {relative_path}"


def test_registry_commands_match_filesystem():
    """Every registered command must match command files on disk."""
    registered = set(_registry()["commands"])
    on_disk = _get_command_files()
    assert registered == on_disk


def test_registry_configs_match_filesystem():
    """Every registered config file must match config files on disk."""
    registered = set(_registry()["config_files"])
    on_disk = _get_config_files()
    assert registered == on_disk


def test_installer_does_not_define_command_or_config_manifests():
    """Installer must consume the registry instead of defining local manifests."""
    content = INSTALL_JS.read_text()
    assert "SUB_COMMANDS" not in content
    assert "CONFIG_FILES" not in content
    assert "getCommandNames" in content
    assert "getConfigFiles" in content


def test_installer_copies_registered_commands_and_configs(tmp_path: Path):
    """Installer must deploy every registered command and config file."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    result = subprocess.run(
        ["node", str(INSTALL_JS), "--global"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    registry = _registry()
    install_root = tmp_path / ".claude" / "commands" / "turing"
    for command in registry["commands"]:
        assert (install_root / command / "SKILL.md").exists(), f"missing {command}/SKILL.md"
    for config_file in registry["config_files"]:
        assert (install_root / "config" / config_file).exists(), f"missing config/{config_file}"


# --- Verify manifest ---


def test_verifier_does_not_define_command_or_config_manifests():
    """Verifier must consume the registry instead of defining local manifests."""
    content = VERIFY_JS.read_text()
    assert "EXPECTED_COMMANDS" not in content
    assert "EXPECTED_CONFIG" not in content
    assert "getExpectedCommandPaths" in content
    assert "getConfigFiles" in content


def test_verify_reports_missing_registered_command(tmp_path: Path):
    """Verifier must report missing commands from the registry-derived manifest."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    install_result = subprocess.run(
        ["node", str(INSTALL_JS), "--global"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert install_result.returncode == 0, install_result.stderr

    missing_command = tmp_path / ".claude" / "commands" / "turing" / "suggest" / "SKILL.md"
    missing_command.unlink()

    result = subprocess.run(
        ["node", str(VERIFY_JS), "--scope", "global"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "commands/suggest/SKILL.md" in result.stdout


def test_verify_reports_missing_registered_config(tmp_path: Path):
    """Verifier must report missing configs from the registry-derived manifest."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    install_result = subprocess.run(
        ["node", str(INSTALL_JS), "--global"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert install_result.returncode == 0, install_result.stderr

    missing_config = tmp_path / ".claude" / "commands" / "turing" / "config" / "commands.yaml"
    missing_config.unlink()

    result = subprocess.run(
        ["node", str(VERIFY_JS), "--scope", "global"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "config/commands.yaml" in result.stdout


def test_verify_checks_full_template_manifest(tmp_path: Path):
    """Verifier must check every scaffold template installed from the package."""
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    install_result = subprocess.run(
        ["node", str(INSTALL_JS), "--global"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert install_result.returncode == 0, install_result.stderr

    missing_template = tmp_path / ".claude" / "commands" / "turing" / "templates" / "MEMORY.md"
    missing_template.unlink()

    result = subprocess.run(
        ["node", str(VERIFY_JS), "--scope", "global"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "templates/MEMORY.md" in result.stdout


def test_verify_fails_when_templates_missing(tmp_path: Path):
    """Verifier must exit nonzero when an install lacks scaffold templates."""
    install_root = tmp_path / ".claude" / "commands" / "turing"
    install_root.mkdir(parents=True)
    (install_root / "SKILL.md").write_text("router")

    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    result = subprocess.run(
        ["node", str(VERIFY_JS), "--scope", "global"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "templates/scripts/scaffold.py" in result.stdout


def test_init_docs_mention_installed_template_location():
    """/turing:init should document project and global installed template paths."""
    content = (COMMANDS_DIR / "init.md").read_text()
    assert ".claude/commands/turing/templates" in content
    assert "~/.claude/commands/turing/templates" in content


# --- Scaffold manifest ---


def test_scaffold_includes_all_scripts():
    """scaffold.py TEMPLATE_DIRS['scripts'] must include all template scripts."""
    content = SCAFFOLD_PY.read_text()
    # Extract the scripts list from TEMPLATE_DIRS
    scripts_match = re.search(r'"scripts":\s*\[(.*?)\]', content, re.DOTALL)
    assert scripts_match, "Could not find scripts list in scaffold.py"
    scaffold_scripts = set(re.findall(r'"([^"]+)"', scripts_match.group(1)))

    # Get actual scripts on disk (excluding __pycache__)
    scripts_dir = REPO_ROOT / "templates" / "scripts"
    on_disk = {f.name for f in scripts_dir.iterdir()
               if f.is_file() and f.name != "__init__.py" and not f.name.startswith(".")}

    missing = on_disk - scaffold_scripts
    assert missing == set(), f"Scripts on disk but not in scaffold.py: {missing}"


def test_cli_init_args_preserve_shell_metacharacters():
    """CLI init should pass user args as argv values, not shell-split strings."""
    script = "import { buildInitArgs } from './bin/cli.js'; console.log(JSON.stringify(buildInitArgs(process.argv[1], process.argv[2])));"
    name = "project; touch injected"
    directory = "ml/path with spaces"

    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, name, directory],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [name, directory]


def test_cli_parses_when_invoked_through_bin_symlink(tmp_path: Path):
    """npm-style bin symlinks should still execute the CLI parser."""
    package = json.loads((REPO_ROOT / "package.json").read_text())
    cli_link = tmp_path / "claude-turing"
    cli_link.symlink_to(CLI_JS)

    result = subprocess.run(
        ["node", str(cli_link), "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == package["version"]
