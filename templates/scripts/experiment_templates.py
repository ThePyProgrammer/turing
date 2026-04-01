#!/usr/bin/env python3
"""Save, list, and apply reusable experiment configuration templates.

Extract proven configurations from successful experiments, strip
project-specific paths, and save as portable templates. Apply them
to new projects to bootstrap experiments without rediscovering
hyperparameters.

Templates are stored in ~/.turing/templates/ for cross-project reuse.

Usage:
    python scripts/experiment_templates.py save exp-042 "gradient-boost-tuned"
    python scripts/experiment_templates.py save exp-042 "gb-tuned" --description "Tuned GBM config"
    python scripts/experiment_templates.py list
    python scripts/experiment_templates.py show gradient-boost-tuned
    python scripts/experiment_templates.py apply gradient-boost-tuned
    python scripts/experiment_templates.py apply gradient-boost-tuned --output custom-config.yaml
    python scripts/experiment_templates.py delete gradient-boost-tuned
    python scripts/experiment_templates.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
DEFAULT_TEMPLATE_DIR = Path.home() / ".turing" / "templates"

# Keys that are project-specific and should be stripped from templates
STRIP_KEYS = {
    "data_path", "data_dir", "output_dir", "output_path", "log_path",
    "checkpoint_dir", "cache_dir", "model_path", "save_path",
    "experiment_id", "timestamp", "run_id",
}

# Keys that contain path-like values
PATH_PATTERNS = [
    re.compile(r"^(/|~|\.\.?/)"),  # Absolute or relative paths
    re.compile(r"\.(csv|parquet|jsonl|pkl|h5|pt|pth|ckpt)$"),  # File extensions
]


# --- Template Operations ---


def extract_template(
    experiment: dict,
    strip_paths: bool = True,
) -> dict:
    """Extract a portable template from an experiment config.

    Strips project-specific paths and identifiers to make the config
    reusable across projects.

    Args:
        experiment: Full experiment dict from log.
        strip_paths: Whether to remove path-like values.

    Returns:
        Cleaned config dict suitable for a template.
    """
    config = dict(experiment.get("config", {}))

    # Strip known project-specific keys
    for key in STRIP_KEYS:
        config.pop(key, None)

    # Strip values that look like file paths
    if strip_paths:
        config = _strip_path_values(config)

    return config


def _strip_path_values(d: dict) -> dict:
    """Recursively remove values that look like file paths."""
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, str):
            is_path = any(p.search(v) for p in PATH_PATTERNS)
            if is_path:
                cleaned[k] = f"<{k}>"  # Placeholder
            else:
                cleaned[k] = v
        elif isinstance(v, dict):
            cleaned[k] = _strip_path_values(v)
        else:
            cleaned[k] = v
    return cleaned


def save_template(
    name: str,
    config: dict,
    description: str = "",
    source_experiment: str | None = None,
    source_metrics: dict | None = None,
    template_dir: str | None = None,
) -> dict:
    """Save a template to the templates directory.

    Args:
        name: Template name (used as filename).
        config: Extracted configuration dict.
        description: Human-readable description.
        source_experiment: ID of the source experiment.
        source_metrics: Metrics from the source experiment.
        template_dir: Override template directory.

    Returns:
        Result dict with path and metadata.
    """
    tdir = Path(template_dir) if template_dir else DEFAULT_TEMPLATE_DIR
    tdir.mkdir(parents=True, exist_ok=True)

    # Sanitize name for filename
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    path = tdir / f"{safe_name}.yaml"

    template = {
        "name": name,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "experiment_id": source_experiment,
            "metrics": source_metrics or {},
            "project": Path.cwd().name,
        },
        "config": config,
    }

    with open(path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)

    return {
        "name": name,
        "path": str(path),
        "config_keys": list(config.keys()),
        "source_experiment": source_experiment,
    }


def list_templates(template_dir: str | None = None) -> list[dict]:
    """List all saved templates.

    Args:
        template_dir: Override template directory.

    Returns:
        List of template summary dicts.
    """
    tdir = Path(template_dir) if template_dir else DEFAULT_TEMPLATE_DIR
    if not tdir.exists():
        return []

    templates = []
    for path in sorted(tdir.glob("*.yaml")):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            templates.append({
                "name": data.get("name", path.stem),
                "description": data.get("description", ""),
                "created_at": data.get("created_at", ""),
                "source": data.get("source", {}),
                "config_keys": list(data.get("config", {}).keys()),
                "path": str(path),
            })
        except (yaml.YAMLError, OSError):
            continue

    return templates


def show_template(name: str, template_dir: str | None = None) -> dict | None:
    """Load and return a template by name.

    Args:
        name: Template name.
        template_dir: Override template directory.

    Returns:
        Full template dict, or None if not found.
    """
    tdir = Path(template_dir) if template_dir else DEFAULT_TEMPLATE_DIR
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    path = tdir / f"{safe_name}.yaml"

    if not path.exists():
        return None

    with open(path) as f:
        return yaml.safe_load(f)


def apply_template(
    name: str,
    output_path: str = "config.yaml",
    overrides: dict | None = None,
    template_dir: str | None = None,
) -> dict:
    """Apply a template to generate a project config.

    Loads the template, merges with any existing config, applies
    overrides, and fills in project-specific placeholders.

    Args:
        name: Template name to apply.
        output_path: Where to write the generated config.
        overrides: Dict of key=value overrides to apply on top.
        template_dir: Override template directory.

    Returns:
        Result dict with generated config and path.
    """
    template = show_template(name, template_dir)
    if template is None:
        return {"error": f"Template '{name}' not found"}

    template_config = template.get("config", {})

    # Load existing config and merge (template values take precedence)
    existing = load_config(output_path)
    merged = _deep_merge(existing, template_config)

    # Apply overrides
    if overrides:
        merged = _deep_merge(merged, overrides)

    # Replace placeholders with sensible defaults
    merged = _fill_placeholders(merged)

    # Write config
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    return {
        "template": name,
        "output_path": str(out),
        "config_keys": list(merged.keys()),
        "source_description": template.get("description", ""),
        "applied_overrides": list((overrides or {}).keys()),
    }


def delete_template(name: str, template_dir: str | None = None) -> dict:
    """Delete a template by name.

    Args:
        name: Template name.
        template_dir: Override template directory.

    Returns:
        Result dict.
    """
    tdir = Path(template_dir) if template_dir else DEFAULT_TEMPLATE_DIR
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    path = tdir / f"{safe_name}.yaml"

    if not path.exists():
        return {"error": f"Template '{name}' not found"}

    path.unlink()
    return {"deleted": name, "path": str(path)}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Override values take precedence."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _fill_placeholders(config: dict) -> dict:
    """Replace <key> placeholders with reasonable defaults."""
    filled = {}
    for k, v in config.items():
        if isinstance(v, str) and v.startswith("<") and v.endswith(">"):
            # Leave placeholder — user should fill these
            filled[k] = v
        elif isinstance(v, dict):
            filled[k] = _fill_placeholders(v)
        else:
            filled[k] = v
    return filled


# --- Report ---


def format_templates_report(templates: list[dict]) -> str:
    """Format template list as a readable markdown table."""
    if not templates:
        return "No templates found.\n\nSave one: `experiment_templates.py save <exp-id> <name>`"

    lines = [
        "# Experiment Templates",
        "",
        f"*{len(templates)} template(s) in `{DEFAULT_TEMPLATE_DIR}`*",
        "",
        "| Name | Description | Source | Created | Config Keys |",
        "|------|-------------|--------|---------|-------------|",
    ]

    for t in templates:
        name = t.get("name", "?")
        desc = t.get("description", "—") or "—"
        source = t.get("source", {})
        src_exp = source.get("experiment_id", "—") or "—"
        src_proj = source.get("project", "") or ""
        source_display = f"{src_exp}"
        if src_proj:
            source_display += f" ({src_proj})"
        created = t.get("created_at", "?")[:10]
        n_keys = len(t.get("config_keys", []))
        lines.append(f"| {name} | {desc[:40]} | {source_display} | {created} | {n_keys} |")

    return "\n".join(lines)


def format_template_detail(template: dict) -> str:
    """Format a single template in detail."""
    lines = [
        f"# Template: {template.get('name', '?')}",
        "",
        f"**Description:** {template.get('description', '—')}",
        f"**Created:** {template.get('created_at', '?')[:19]} UTC",
    ]

    source = template.get("source", {})
    if source.get("experiment_id"):
        lines.append(f"**Source:** {source['experiment_id']} from {source.get('project', '?')}")
    if source.get("metrics"):
        metrics_str = ", ".join(f"{k}={v}" for k, v in source["metrics"].items())
        lines.append(f"**Source Metrics:** {metrics_str}")

    lines.extend(["", "## Configuration", "", "```yaml"])
    config = template.get("config", {})
    lines.append(yaml.dump(config, default_flow_style=False, sort_keys=False).strip())
    lines.extend(["```", ""])

    return "\n".join(lines)


def save_templates_report(report: dict, path: str = "experiments/templates") -> Path:
    """Save template operation report to YAML."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"template-op-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.yaml"
    with open(out, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return out


# --- Orchestration ---


def run_template(
    action: str,
    name: str | None = None,
    experiment_id: str | None = None,
    description: str = "",
    output_path: str = "config.yaml",
    template_dir: str | None = None,
    log_path: str = DEFAULT_LOG_PATH,
    config_path: str = "config.yaml",
) -> dict:
    """Run template operation.

    Returns:
        Result dict.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if action == "save":
        if not experiment_id or not name:
            return {"error": "Both experiment_id and name are required for save"}

        experiments = load_experiments(log_path)
        exp = next((e for e in experiments
                     if e.get("experiment_id") == experiment_id), None)
        if exp is None:
            return {"error": f"Experiment '{experiment_id}' not found"}

        config = extract_template(exp)
        result = save_template(
            name, config, description,
            source_experiment=experiment_id,
            source_metrics=exp.get("metrics"),
            template_dir=template_dir,
        )
        return {"timestamp": timestamp, "action": "save", **result}

    elif action == "list":
        templates = list_templates(template_dir)
        return {
            "timestamp": timestamp,
            "action": "list",
            "count": len(templates),
            "templates": templates,
        }

    elif action == "show":
        if not name:
            return {"error": "Template name required for show"}
        template = show_template(name, template_dir)
        if template is None:
            return {"error": f"Template '{name}' not found"}
        return {"timestamp": timestamp, "action": "show", "template": template}

    elif action == "apply":
        if not name:
            return {"error": "Template name required for apply"}
        result = apply_template(name, output_path, template_dir=template_dir)
        if "error" in result:
            return {"timestamp": timestamp, **result}
        return {"timestamp": timestamp, "action": "apply", **result}

    elif action == "delete":
        if not name:
            return {"error": "Template name required for delete"}
        result = delete_template(name, template_dir)
        return {"timestamp": timestamp, "action": "delete", **result}

    return {"error": f"Unknown action: {action}"}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Experiment configuration templates")
    parser.add_argument("action", choices=["save", "list", "show", "apply", "delete"],
                        help="Template action")
    parser.add_argument("name", nargs="?", default=None,
                        help="Template name (for save/show/apply/delete)")
    parser.add_argument("--experiment", default=None,
                        help="Source experiment ID (for save)")
    parser.add_argument("--description", default="",
                        help="Template description (for save)")
    parser.add_argument("--output", default="config.yaml",
                        help="Output config path (for apply)")
    parser.add_argument("--template-dir", default=None,
                        help="Override template directory")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    report = run_template(
        action=args.action,
        name=args.name,
        experiment_id=args.experiment,
        description=args.description,
        output_path=args.output,
        template_dir=args.template_dir,
        log_path=args.log,
        config_path=args.config,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        if "error" in report:
            print(f"ERROR: {report['error']}", file=sys.stderr)
            sys.exit(1)

        action = report.get("action")
        if action == "save":
            print(f"Saved template '{report['name']}' to {report['path']}")
            print(f"  Source: {report.get('source_experiment', '?')}")
            print(f"  Config keys: {', '.join(report.get('config_keys', []))}")
        elif action == "list":
            print(format_templates_report(report.get("templates", [])))
        elif action == "show":
            print(format_template_detail(report.get("template", {})))
        elif action == "apply":
            print(f"Applied template '{report['template']}' to {report['output_path']}")
            if report.get("applied_overrides"):
                print(f"  Overrides: {', '.join(report['applied_overrides'])}")
        elif action == "delete":
            print(f"Deleted template '{report.get('deleted', '?')}'")


if __name__ == "__main__":
    main()
