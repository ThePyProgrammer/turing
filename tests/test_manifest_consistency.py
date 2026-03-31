"""Manifest consistency tests.

Validates that the installer, verifier, scaffold, and filesystem
are all in sync. Catches the exact class of drift that caused
the v1.0.0 blockers — commands/configs added to the filesystem
but not to the deployment manifests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
COMMANDS_DIR = REPO_ROOT / "commands"
CONFIG_DIR = REPO_ROOT / "config"
INSTALL_JS = REPO_ROOT / "src" / "install.js"
VERIFY_JS = REPO_ROOT / "src" / "verify.js"
SCAFFOLD_PY = REPO_ROOT / "templates" / "scripts" / "scaffold.py"


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


# --- Install manifest ---


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
