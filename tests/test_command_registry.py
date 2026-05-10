from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMANDS_DIR = REPO_ROOT / "commands"
CONFIG_DIR = REPO_ROOT / "config"
REGISTRY_PATH = CONFIG_DIR / "commands.yaml"

REQUIRED_FIELDS = {
    "description",
    "lifecycle",
    "invocation_mode",
    "model_invocation",
    "mutates_project",
    "tools",
}
SUPPORTED_INVOCATION_MODES = {"slash_only"}
SUPPORTED_MODEL_INVOCATIONS = {"disabled", "enabled"}
SUPPORTED_SCRIPT_LOCATIONS = {"repo", "scaffold"}


def load_registry() -> dict[str, Any]:
    data = yaml.safe_load(REGISTRY_PATH.read_text())
    assert isinstance(data, dict)
    assert isinstance(data.get("commands"), dict)
    return data


def command_files() -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(COMMANDS_DIR.glob("*.md"))
        if path.name != "turing.md"
    }


def parse_frontmatter(path: Path) -> dict[str, Any]:
    lines = path.read_text().splitlines()
    assert lines[0] == "---"
    end = lines.index("---", 1)
    frontmatter = yaml.safe_load("\n".join(lines[1:end]))
    assert isinstance(frontmatter, dict)
    return frontmatter


def split_allowed_tools(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in value:
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def normalize_allowed_tools(value: str | None) -> list[str]:
    if not value:
        return ["Read", "Bash"]
    tools: list[str] = []
    for tool in split_allowed_tools(value):
        if not tool:
            continue
        tools.append(re.sub(r"\(.*$", "", tool).strip())
    return tools


def test_registry_covers_exactly_command_files() -> None:
    registry = load_registry()

    assert set(registry["commands"]) == set(command_files())


def test_entries_have_required_metadata_and_valid_values() -> None:
    registry = load_registry()

    for command_name, entry in registry["commands"].items():
        assert isinstance(entry, dict), command_name
        assert REQUIRED_FIELDS <= set(entry), command_name
        assert isinstance(entry["description"], str) and entry["description"], command_name
        assert isinstance(entry["lifecycle"], str) and entry["lifecycle"], command_name
        assert entry["invocation_mode"] in SUPPORTED_INVOCATION_MODES, command_name
        assert entry["model_invocation"] in SUPPORTED_MODEL_INVOCATIONS, command_name
        assert isinstance(entry["mutates_project"], bool), command_name
        assert isinstance(entry["tools"], list) and entry["tools"], command_name
        assert all(isinstance(tool, str) and tool for tool in entry["tools"]), command_name
        if "argument_hint" in entry:
            assert isinstance(entry["argument_hint"], str) and entry["argument_hint"], command_name


def routing_table_lifecycles() -> dict[str, str]:
    lifecycles: dict[str, str] = {}
    for line in (COMMANDS_DIR / "turing.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in {"User says...", "---"}:
            continue
        route = cells[1]
        match = re.fullmatch(r"`/turing:([a-z]+)`", route)
        if match:
            lifecycles[match.group(1)] = cells[2].lower()
    return lifecycles


def test_registry_lifecycle_matches_router_table() -> None:
    registry = load_registry()
    lifecycles = routing_table_lifecycles()

    assert set(lifecycles) == set(command_files())
    for command_name, lifecycle in lifecycles.items():
        assert registry["commands"][command_name]["lifecycle"] == lifecycle, command_name


def test_registry_matches_command_frontmatter() -> None:
    registry = load_registry()

    for command_name, path in command_files().items():
        entry = registry["commands"][command_name]
        frontmatter = parse_frontmatter(path)
        assert frontmatter["name"] == command_name
        assert entry["description"] == frontmatter["description"], command_name
        expected_model_invocation = (
            "disabled" if frontmatter.get("disable-model-invocation") is True else "enabled"
        )
        assert entry["model_invocation"] == expected_model_invocation, command_name
        if "argument-hint" in frontmatter:
            assert entry.get("argument_hint") == frontmatter["argument-hint"], command_name
        else:
            assert "argument_hint" not in entry, command_name
        assert entry["tools"] == normalize_allowed_tools(frontmatter.get("allowed-tools")), command_name


def test_config_file_manifest_matches_config_directory_files() -> None:
    registry = load_registry()

    assert registry.get("config_files") == sorted(path.name for path in CONFIG_DIR.iterdir() if path.is_file())


def test_equivalent_script_entries_have_valid_shape_and_existing_paths() -> None:
    registry = load_registry()

    for command_name, entry in registry["commands"].items():
        if "equivalent_script" not in entry:
            continue
        script = entry["equivalent_script"]
        assert set(script) == {"path", "location"}, command_name
        assert isinstance(script["path"], str) and script["path"], command_name
        assert script["location"] in SUPPORTED_SCRIPT_LOCATIONS, command_name
        if script["location"] == "repo":
            assert (REPO_ROOT / script["path"]).is_file(), command_name
        else:
            assert (REPO_ROOT / "templates" / script["path"]).is_file(), command_name


def test_suggest_registry_contract() -> None:
    registry = load_registry()

    suggest = registry["commands"]["suggest"]
    assert suggest["invocation_mode"] == "slash_only"
    assert suggest["model_invocation"] == "disabled"
    assert suggest["equivalent_script"] == {
        "path": "scripts/suggest_next.py",
        "location": "scaffold",
    }


def test_node_registry_loader_exports_sorted_manifests() -> None:
    registry = load_registry()
    script = """
        import {
            getCommandNames,
            getConfigFiles,
            getExpectedCommandPaths,
        } from './src/command-registry.js';

        const [names, configs, paths] = await Promise.all([
            getCommandNames(),
            getConfigFiles(),
            getExpectedCommandPaths(),
        ]);
        console.log(JSON.stringify({ names, configs, paths }));
    """

    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["names"] == sorted(registry["commands"])
    assert manifest["paths"][0] == "SKILL.md"
    assert "suggest/SKILL.md" in manifest["paths"]
    assert manifest["configs"] == sorted(registry["config_files"])


def test_node_registry_loader_rejects_extra_equivalent_script_keys(tmp_path: Path) -> None:
    registry = load_registry()
    registry["commands"]["suggest"]["equivalent_script"]["extra"] = "unexpected"
    registry_path = tmp_path / "commands.yaml"
    registry_path.write_text(yaml.safe_dump(registry))

    script = """
        import { loadCommandRegistry } from './src/command-registry.js';

        try {
            await loadCommandRegistry(process.argv[1]);
            console.error('expected registry validation to fail');
            process.exit(1);
        } catch (error) {
            console.log(error.message);
        }
    """

    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, str(registry_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "commands.suggest.equivalent_script must contain exactly: location, path" in result.stdout
