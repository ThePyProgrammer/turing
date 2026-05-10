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

REPO_ROOT = Path(__file__).parent.parent
COMMANDS_DIR = REPO_ROOT / "commands"
CONFIG_DIR = REPO_ROOT / "config"
INSTALL_JS = REPO_ROOT / "src" / "install.js"
VERIFY_JS = REPO_ROOT / "src" / "verify.js"
CLI_JS = REPO_ROOT / "bin" / "cli.js"
SCAFFOLD_PY = REPO_ROOT / "templates" / "scripts" / "scaffold.py"
TEMPLATES_DIR = REPO_ROOT / "templates"


def _extract_js_array(file_path: Path, var_name: str) -> list[str]:
    """Extract a JS array variable from a source file."""
    content = file_path.read_text()
    # Match: const VAR_NAME = [...]; across multiple lines
    pattern = rf'(?:const|let|var)\s+{var_name}\s*=\s*\[(.*?)\]'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    array_content = match.group(1)
    # Extract quoted strings
    return re.findall(r'"([^"]+)"', array_content)


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


def test_install_commands_match_filesystem():
    """Every command .md in commands/ must be in install.js SUB_COMMANDS."""
    installed = set(_extract_js_array(INSTALL_JS, "SUB_COMMANDS"))
    on_disk = _get_command_files()
    missing = on_disk - installed
    assert missing == set(), f"Commands on disk but not in install.js: {missing}"


def test_install_commands_no_ghosts():
    """install.js SUB_COMMANDS should not reference commands that don't exist."""
    installed = set(_extract_js_array(INSTALL_JS, "SUB_COMMANDS"))
    on_disk = _get_command_files()
    ghosts = installed - on_disk
    assert ghosts == set(), f"Commands in install.js but not on disk: {ghosts}"


def test_install_configs_match_filesystem():
    """Every config file must be in install.js CONFIG_FILES."""
    installed = set(_extract_js_array(INSTALL_JS, "CONFIG_FILES"))
    on_disk = _get_config_files()
    missing = on_disk - installed
    assert missing == set(), f"Configs on disk but not in install.js: {missing}"


# --- Verify manifest ---


def test_verify_commands_cover_install():
    """verify.js EXPECTED_COMMANDS must cover all install.js SUB_COMMANDS."""
    installed = set(_extract_js_array(INSTALL_JS, "SUB_COMMANDS"))
    verified_raw = _extract_js_array(VERIFY_JS, "EXPECTED_COMMANDS")
    # verify.js uses paths like "init/SKILL.md" — extract the directory name
    verified = set()
    for v in verified_raw:
        if "/" in v:
            verified.add(v.split("/")[0])
        else:
            verified.add(v.replace(".md", "").replace("SKILL", ""))
    missing = installed - verified
    assert missing == set(), f"Commands installed but not verified: {missing}"


def test_verify_configs_cover_install():
    """verify.js EXPECTED_CONFIG must cover all install.js CONFIG_FILES."""
    installed = set(_extract_js_array(INSTALL_JS, "CONFIG_FILES"))
    verified = set(_extract_js_array(VERIFY_JS, "EXPECTED_CONFIG"))
    missing = installed - verified
    assert missing == set(), f"Configs installed but not verified: {missing}"


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
