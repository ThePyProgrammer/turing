#!/usr/bin/env python3
"""Unified project scaffolding for the autoresearch pipeline.

Single implementation of project scaffolding used by both:
- /turing:init (Claude Code) — calls with --interactive or pre-filled args
- bin/turing-init.sh (CLI) — calls with explicit --project-name etc.

Eliminates the dual-path divergence where init.md and turing-init.sh
had different capabilities (substitution, hooks, venv, memory).

Usage:
    python scripts/scaffold.py --project-name sentiment --target-metric accuracy \\
        --task-description "Predict sentiment" --ml-dir ml/sentiment \\
        --data-source data/reviews.csv --metric-direction higher

    python scripts/scaffold.py --interactive  # Prompt for each value
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PLACEHOLDER_MAP = {
    "PROJECT_NAME": "project_name",
    "TARGET_METRIC": "target_metric",
    "TASK_DESCRIPTION": "task_description",
    "ML_DIR": "ml_dir",
    "DATA_SOURCE": "data_source",
    "METRIC_DIRECTION": "metric_direction",
}

# Files to copy from templates/ to the ML directory
TEMPLATE_FILES = [
    "prepare.py",
    "evaluate.py",
    "train.py",
    "config.yaml",
    "sweep_config.yaml",
    "program.md",
    "README.md",
    "MEMORY.md",
    "model_registry.yaml",
    "model_contract.md",
    "requirements.txt",
    "pyproject.toml",
]

TEMPLATE_DIRS = {
    "features": ["__init__.py", "featurizers.py"],
    "scripts": [
        "__init__.py",
        "log_experiment.py",
        "show_metrics.py",
        "compare_runs.py",
        "sweep.py",
        "post-train-hook.sh",
        "stop-hook.sh",
        "check_convergence.py",
        "verify_placeholders.py",
        "manage_hypotheses.py",
        "generate_brief.py",
        "show_experiment_tree.py",
        "statistical_compare.py",
        "suggest_next.py",
        "update_state.py",
        "parse_metrics.py",
        "scaffold.py",
        "novelty_guard.py",
        "synthesize_decision.py",
        "show_families.py",
        "critique_hypothesis.py",
        "experiment_index.py",
        "generate_logbook.py",
        "validate_stability.py",
        "show_environment.py",
        "turing_io.py",
        "preflight.py",
        "cleanup.py",
        "generate_model_card.py",
        "cost_frontier.py",
        "leaderboard.py",
        "diff_configs.py",
        "export_results.py",
        "plot_trajectory.py",
        "treequest_suggest.py",
        "seed_runner.py",
        "reproduce_experiment.py",
        "diagnose_errors.py",
        "ablation_study.py",
        "pareto_frontier.py",
        "profile_training.py",
        "checkpoint_manager.py",
        "export_model.py",
        "export_formats.py",
        "equivalence_checker.py",
        "latency_benchmark.py",
        "export_card.py",
        "literature_search.py",
        "draft_paper_sections.py",
        "experiment_queue.py",
        "smart_retry.py",
        "fork_experiment.py",
        "experiment_diff.py",
        "training_monitor.py",
        "regression_gate.py",
        "build_ensemble.py",
        "pipeline_manager.py",
        "warm_start.py",
        "scaling_estimator.py",
        "budget_manager.py",
        "model_distiller.py",
        "knowledge_transfer.py",
        "methodology_audit.py",
        "sanity_checks.py",
        "generate_baselines.py",
        "leakage_detector.py",
        "model_xray.py",
        "sensitivity_analysis.py",
        "calibration.py",
        "feature_intelligence.py",
        "curriculum_optimizer.py",
        "model_pruning.py",
        "model_quantization.py",
        "model_merger.py",
        "architecture_surgery.py",
        "trend_analysis.py",
        "session_flashback.py",
        "experiment_archive.py",
        "experiment_annotations.py",
        "experiment_search.py",
        "experiment_templates.py",
        "experiment_replay.py",
    ],
    "tests": ["__init__.py", "conftest.py"],
}

DIRECTORIES_TO_CREATE = [
    "data/splits",
    "experiments",
    "experiments/seed_studies",
    "experiments/reproductions",
    "experiments/diagnoses",
    "experiments/ablations",
    "experiments/frontiers",
    "experiments/predictions",
    "experiments/profiles",
    "experiments/checkpoints",
    "exports",
    "experiments/literature",
    "paper/sections",
    "experiments/retries",
    "experiments/forks",
    "experiments/diffs",
    "experiments/monitors",
    "experiments/regressions",
    "experiments/ensembles",
    "experiments/cache",
    "experiments/warm_starts",
    "experiments/scaling",
    "experiments/distillations",
    "experiments/transfers",
    "experiments/audits",
    "experiments/sanity",
    "experiments/baselines",
    "experiments/leakage",
    "experiments/xrays",
    "experiments/sensitivity",
    "experiments/calibration",
    "experiments/features",
    "experiments/curriculum",
    "experiments/pruning",
    "experiments/quantization",
    "experiments/merges",
    "experiments/surgery",
    "experiments/trends",
    "experiments/flashbacks",
    "experiments/archive",
    "experiments/searches",
    "experiments/replays",
    "experiments/logs",
    "models/best",
    "models/archive",
]

SHELL_SCRIPTS = [
    "scripts/post-train-hook.sh",
    "scripts/stop-hook.sh",
]


def find_templates_dir() -> Path | None:
    """Locate the templates directory relative to this script or plugin root."""
    # When running from a scaffolded project, templates are local
    script_dir = Path(__file__).parent

    # Check: are we inside the plugin's templates/scripts/ ?
    candidate = script_dir.parent  # templates/
    if (candidate / "prepare.py").exists():
        return candidate

    # Check: plugin root (two levels up from scripts/)
    plugin_root = script_dir.parent.parent
    candidate = plugin_root / "templates"
    if candidate.exists() and (candidate / "prepare.py").exists():
        return candidate

    # Search common plugin locations
    home = Path.home()
    for pattern in [
        home / ".claude" / "plugins" / "*" / "templates",
    ]:
        for match in sorted(pattern.parent.glob(pattern.name)):
            if (match / "prepare.py").exists():
                return match

    return None


def replace_placeholders(text: str, values: dict[str, str]) -> str:
    """Replace all {{PLACEHOLDER}} markers in text with values."""
    for placeholder, arg_name in PLACEHOLDER_MAP.items():
        value = values.get(arg_name, "")
        text = text.replace(f"{{{{{placeholder}}}}}", value)
    return text


def scaffold_project(
    templates_dir: Path,
    ml_dir: str,
    values: dict[str, str],
    setup_venv: bool = True,
    setup_hooks: bool = True,
) -> dict[str, int]:
    """Scaffold a complete ML project.

    Args:
        templates_dir: Path to the templates/ directory.
        ml_dir: Target ML directory (relative to cwd).
        values: Dict mapping arg names to values for placeholder substitution.
        setup_venv: Whether to create and populate a Python venv.
        setup_hooks: Whether to configure Claude Code hooks.

    Returns:
        Dict with counts: files_copied, placeholders_replaced, dirs_created.
    """
    target = Path(ml_dir)
    target.mkdir(parents=True, exist_ok=True)

    stats = {"files_copied": 0, "placeholders_replaced": 0, "dirs_created": 0}

    # Copy and substitute template files
    for filename in TEMPLATE_FILES:
        src = templates_dir / filename
        if not src.exists():
            print(f"  Warning: template {filename} not found, skipping", file=sys.stderr)
            continue
        content = src.read_text(encoding="utf-8")
        content = replace_placeholders(content, values)
        (target / filename).write_text(content, encoding="utf-8")
        stats["files_copied"] += 1

    # Copy and substitute template directories
    for dirname, files in TEMPLATE_DIRS.items():
        dir_target = target / dirname
        dir_target.mkdir(parents=True, exist_ok=True)
        for filename in files:
            src = templates_dir / dirname / filename
            if not src.exists():
                continue
            content = src.read_text(encoding="utf-8")
            content = replace_placeholders(content, values)
            (dir_target / filename).write_text(content, encoding="utf-8")
            stats["files_copied"] += 1

    # Create directories
    for d in DIRECTORIES_TO_CREATE:
        (target / d).mkdir(parents=True, exist_ok=True)
        stats["dirs_created"] += 1

    # Make shell scripts executable
    for script in SHELL_SCRIPTS:
        script_path = target / script
        if script_path.exists():
            script_path.chmod(script_path.stat().st_mode | 0o111)

    # Count placeholders replaced
    placeholder_re = re.compile(r"\{\{[A-Z_]+\}\}")
    for path in target.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".md", ".sh", ".txt", ".toml"):
            try:
                text = path.read_text(encoding="utf-8")
                found = placeholder_re.findall(text)
                # If we find remaining placeholders, they were NOT replaced
                # (this means the template had them but our values didn't cover them)
            except UnicodeDecodeError:
                continue

    # Setup agent memory
    memory_dir = Path(".claude") / "agent-memory" / "ml-researcher"
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory_src = templates_dir / "MEMORY.md"
    if memory_src.exists():
        content = memory_src.read_text(encoding="utf-8")
        content = replace_placeholders(content, values)
        (memory_dir / "MEMORY.md").write_text(content, encoding="utf-8")

    # Setup hooks
    if setup_hooks:
        _setup_hooks(ml_dir)

    # Setup venv
    if setup_venv:
        _setup_venv(target)

    return stats


def _setup_hooks(ml_dir: str) -> None:
    """Configure Claude Code hooks in .claude/settings.local.json."""
    settings_path = Path(".claude") / "settings.local.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            settings = {}

    hooks = settings.get("hooks", {})

    # PostToolUse hook for auto-logging
    post_hooks = hooks.get("PostToolUse", [])
    post_hook_cmd = f"bash {ml_dir}/scripts/post-train-hook.sh"
    if not any(post_hook_cmd in str(h) for h in post_hooks):
        post_hooks.append({
            "matcher": "Bash",
            "hooks": [{"type": "command", "command": post_hook_cmd}],
        })
    hooks["PostToolUse"] = post_hooks

    # Stop hook for convergence
    stop_hooks = hooks.get("Stop", [])
    stop_hook_cmd = f"bash {ml_dir}/scripts/stop-hook.sh"
    if not any(stop_hook_cmd in str(h) for h in stop_hooks):
        stop_hooks.append({
            "type": "command",
            "command": stop_hook_cmd,
        })
    hooks["Stop"] = stop_hooks

    settings["hooks"] = hooks
    settings_path.write_text(json.dumps(settings, indent=2))


def _setup_venv(target: Path) -> None:
    """Create Python venv and install requirements."""
    venv_path = target / ".venv"
    if venv_path.exists():
        print("  Venv already exists, skipping creation.", file=sys.stderr)
        return

    print("  Creating virtual environment...", file=sys.stderr)
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True, capture_output=True,
        )
        pip = str(venv_path / "bin" / "pip")
        req = str(target / "requirements.txt")
        if Path(req).exists():
            print("  Installing requirements...", file=sys.stderr)
            subprocess.run(
                [pip, "install", "-r", req],
                check=True, capture_output=True,
            )
    except subprocess.CalledProcessError as e:
        print(f"  Warning: venv setup failed: {e}", file=sys.stderr)


def verify_placeholders(ml_dir: str) -> list[tuple[str, int, str]]:
    """Check for unreplaced placeholders in scaffolded files.

    Returns list of (filepath, line_number, placeholder) tuples.
    """
    target = Path(ml_dir)
    placeholder_re = re.compile(r"\{\{([A-Z_]+)\}\}")
    known = set(PLACEHOLDER_MAP.keys())
    findings = []

    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in (".py", ".yaml", ".yml", ".md", ".sh", ".txt", ".toml"):
            continue
        if ".venv" in path.parts:
            continue
        try:
            for i, line in enumerate(path.read_text().splitlines(), 1):
                for match in placeholder_re.finditer(line):
                    name = match.group(1)
                    if name in known:
                        findings.append((str(path.relative_to(target)), i, name))
        except UnicodeDecodeError:
            continue

    return findings


def interactive_prompt() -> dict[str, str]:
    """Prompt the user for all scaffold values."""
    print("\nTuring ML Research Harness — Project Setup\n")

    values = {}
    values["project_name"] = input("Project name (e.g., sentiment, churn): ").strip()
    values["target_metric"] = input("Primary metric (accuracy, f1, mae, mse, auc): ").strip()
    values["metric_direction"] = input("Is lower better or higher better? (lower/higher): ").strip()
    values["task_description"] = input("Task description (e.g., Predict customer churn): ").strip()
    values["ml_dir"] = input(f"ML directory (e.g., ml/{values['project_name']}): ").strip()
    values["data_source"] = input("Data source path (e.g., data/training.csv): ").strip()

    return values


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Scaffold an ML project")
    parser.add_argument("--interactive", action="store_true", help="Prompt for values")
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--target-metric", default=None)
    parser.add_argument("--metric-direction", default=None)
    parser.add_argument("--task-description", default=None)
    parser.add_argument("--ml-dir", default=None)
    parser.add_argument("--data-source", default=None)
    parser.add_argument("--no-venv", action="store_true", help="Skip venv creation")
    parser.add_argument("--no-hooks", action="store_true", help="Skip hook configuration")
    parser.add_argument("--templates-dir", default=None, help="Override templates directory")
    args = parser.parse_args()

    # Get values
    if args.interactive:
        values = interactive_prompt()
    else:
        values = {
            "project_name": args.project_name or "my-project",
            "target_metric": args.target_metric or "accuracy",
            "metric_direction": args.metric_direction or "higher",
            "task_description": args.task_description or "ML task",
            "ml_dir": args.ml_dir or ".",
            "data_source": args.data_source or "data/training.csv",
        }

    # Find templates
    if args.templates_dir:
        templates_dir = Path(args.templates_dir)
    else:
        templates_dir = find_templates_dir()

    if templates_dir is None or not templates_dir.exists():
        print("Error: Cannot find templates directory.", file=sys.stderr)
        print("Use --templates-dir to specify the path.", file=sys.stderr)
        sys.exit(1)

    ml_dir = values["ml_dir"]

    print(f"\nScaffolding project: {values['project_name']}")
    print(f"Directory: {ml_dir}")
    print(f"Metric: {values['target_metric']} ({values['metric_direction']} is better)")
    print()

    # Scaffold
    stats = scaffold_project(
        templates_dir=templates_dir,
        ml_dir=ml_dir,
        values=values,
        setup_venv=not args.no_venv,
        setup_hooks=not args.no_hooks,
    )

    print(f"\nScaffolded {stats['files_copied']} files, {stats['dirs_created']} directories.")

    # Verify
    findings = verify_placeholders(ml_dir)
    if findings:
        print(f"\nWarning: {len(findings)} unreplaced placeholder(s):", file=sys.stderr)
        for filepath, line_num, placeholder in findings:
            print(f"  {filepath}:{line_num} — {{{{{placeholder}}}}}", file=sys.stderr)
        sys.exit(1)
    else:
        print("All placeholders replaced successfully.")

    print(f"\nNext steps:")
    print(f"  1. Add training data to {values['data_source']}")
    print(f"  2. cd {ml_dir} && source .venv/bin/activate")
    print(f"  3. python prepare.py")
    print(f"  4. /turing:train  (or: python train.py > run.log 2>&1)")


if __name__ == "__main__":
    main()
