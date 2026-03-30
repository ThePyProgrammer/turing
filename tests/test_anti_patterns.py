"""Anti-pattern tests — architectural invariant enforcement.

Tier 4: These tests verify that the ADR invariants hold in code,
not just in documentation. When an ADR invariant is violated by a
code change, one of these tests should fail.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
AGENTS_DIR = REPO_ROOT / "agents"
CONFIG_DIR = REPO_ROOT / "config"


# --- ADR-0002: Hypothesis-measurement separation ---


def test_evaluate_py_is_readonly():
    """evaluate.py must be labeled READ-ONLY in its docstring."""
    content = (TEMPLATES_DIR / "evaluate.py").read_text()
    assert "READ-ONLY" in content
    assert "MEASUREMENT APPARATUS" in content


def test_prepare_py_is_readonly():
    """prepare.py must be labeled READ-ONLY in its docstring."""
    content = (TEMPLATES_DIR / "prepare.py").read_text()
    assert "READ-ONLY" in content
    assert "MEASUREMENT APPARATUS" in content


def test_train_py_is_agent_editable():
    """train.py must be labeled AGENT-EDITABLE in its header."""
    content = (TEMPLATES_DIR / "train.py").read_text()
    assert "AGENT-EDITABLE" in content
    assert "HYPOTHESIS SPACE" in content


def test_loop_protocol_access_matrix():
    """loop-protocol.md must contain the access control table."""
    content = (REPO_ROOT / "commands" / "rules" / "loop-protocol.md").read_text()
    assert "READ-ONLY" in content
    assert "READ-WRITE" in content
    assert "prepare.py" in content
    assert "evaluate.py" in content
    assert "train.py" in content


# --- ADR-0003: Two-agent least privilege ---


def test_evaluator_has_no_write_tools():
    """ml-evaluator must NOT have Write or Edit in its tools list."""
    content = (AGENTS_DIR / "ml-evaluator.md").read_text()
    # Parse frontmatter tools line
    tools_match = re.search(r"^tools:\s*(.+)$", content, re.MULTILINE)
    assert tools_match, "ml-evaluator.md must have a tools: line"
    tools = tools_match.group(1)
    assert "Write" not in tools, "Evaluator must not have Write tool"
    assert "Edit" not in tools, "Evaluator must not have Edit tool"


def test_researcher_has_write_tools():
    """ml-researcher must have Write and Edit in its tools list."""
    content = (AGENTS_DIR / "ml-researcher.md").read_text()
    tools_match = re.search(r"^tools:\s*(.+)$", content, re.MULTILINE)
    assert tools_match
    tools = tools_match.group(1)
    assert "Write" in tools
    assert "Edit" in tools


# --- ADR-0004: Config format separation ---


def test_domain_knowledge_is_toml():
    """lifecycle and taxonomy must be TOML files."""
    assert (CONFIG_DIR / "lifecycle.toml").exists()
    assert (CONFIG_DIR / "taxonomy.toml").exists()


def test_project_defaults_is_yaml():
    """Plugin defaults must be YAML."""
    assert (CONFIG_DIR / "defaults.yaml").exists()


# --- ADR-0007: Append-only logging ---


def test_log_experiment_uses_kept_status():
    """log_experiment.py default status should be 'kept' (matching lifecycle.toml)."""
    content = (TEMPLATES_DIR / "scripts" / "log_experiment.py").read_text()
    assert 'status: str = "kept"' in content


# --- ADR-0008: Template placeholders ---


def test_all_template_placeholders_documented():
    """Every {{PLACEHOLDER}} in templates/ must appear in defaults.yaml."""
    import yaml

    defaults_path = CONFIG_DIR / "defaults.yaml"
    with open(defaults_path) as f:
        defaults = yaml.safe_load(f)
    documented = set(defaults.get("placeholders", {}).keys())

    # Scan all templates for placeholder usage
    found_placeholders: set[str] = set()
    placeholder_re = re.compile(r"\{\{([A-Z_]+)\}\}")
    for path in TEMPLATES_DIR.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in placeholder_re.finditer(text):
            found_placeholders.add(match.group(1))

    undocumented = found_placeholders - documented
    assert undocumented == set(), f"Undocumented placeholders: {undocumented}"
