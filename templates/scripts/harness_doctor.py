#!/usr/bin/env python3
"""Harness self-diagnosis for the autoresearch pipeline.

Checks environment health, project integrity, resource availability,
and git state. Identifies common issues and auto-fixes where safe.

Usage:
    python scripts/harness_doctor.py
    python scripts/harness_doctor.py --fix
    python scripts/harness_doctor.py --verbose
    python scripts/harness_doctor.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.turing_io import load_config, load_experiments

DEFAULT_LOG_PATH = "experiments/log.jsonl"
MIN_DISK_MB = 1024  # 1 GB

REQUIRED_SCRIPTS = ["train.py", "prepare.py", "evaluate.py"]
REQUIRED_CONFIG_FIELDS = ["evaluation"]

CHECK_CATEGORIES = ["environment", "dependencies", "config", "experiment_log",
                     "scripts", "disk_space", "git_state"]


# --- Individual Checks ---


def check_environment() -> dict:
    """Check Python environment health."""
    issues = []
    version = sys.version_info

    if version < (3, 10):
        issues.append(f"Python {version.major}.{version.minor} — recommend 3.10+")

    # Check if running in a venv
    in_venv = hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)

    return {
        "name": "Python environment",
        "status": "PASS" if not issues else "WARN",
        "detail": f"Python {version.major}.{version.minor}.{version.micro}, venv={'active' if in_venv else 'not active'}",
        "issues": issues,
    }


def check_dependencies(required: list[str] | None = None) -> dict:
    """Check that required packages are importable."""
    if required is None:
        required = ["yaml", "numpy", "sklearn", "pandas", "scipy"]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        return {
            "name": "Dependencies",
            "status": "FAIL",
            "detail": f"{len(missing)} packages missing: {', '.join(missing)}",
            "issues": [f"Cannot import: {pkg}" for pkg in missing],
            "fix": f"pip install {' '.join(missing)}",
        }

    return {
        "name": "Dependencies",
        "status": "PASS",
        "detail": f"All {len(required)} packages importable",
        "issues": [],
    }


def check_config(config_path: str = "config.yaml") -> dict:
    """Check config.yaml validity and required fields."""
    path = Path(config_path)
    issues = []

    if not path.exists():
        return {
            "name": "Config",
            "status": "FAIL",
            "detail": f"{config_path} not found",
            "issues": [f"{config_path} missing"],
            "fix": "Run /turing:init to scaffold the project",
        }

    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return {
            "name": "Config",
            "status": "FAIL",
            "detail": f"{config_path} has YAML parse error",
            "issues": [str(e)],
        }

    if not isinstance(config, dict):
        return {
            "name": "Config",
            "status": "FAIL",
            "detail": f"{config_path} is not a YAML mapping",
            "issues": ["Config must be a YAML dict"],
        }

    for field in REQUIRED_CONFIG_FIELDS:
        if field not in config:
            issues.append(f"Missing required field: {field}")

    status = "PASS" if not issues else "WARN"
    return {
        "name": "Config",
        "status": status,
        "detail": f"{config_path} valid, {len(config)} top-level keys",
        "issues": issues,
    }


def check_experiment_log(log_path: str = DEFAULT_LOG_PATH) -> dict:
    """Check experiment log integrity."""
    path = Path(log_path)

    if not path.exists():
        return {
            "name": "Experiment log",
            "status": "WARN",
            "detail": "No experiment log yet — run /turing:train first",
            "issues": [],
        }

    issues = []
    total_lines = 0
    valid_lines = 0
    corrupt_lines = []
    missing_fields = []

    with open(path) as f:
        for i, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                valid_lines += 1
                # Check for expected fields
                if "metrics" not in entry:
                    missing_fields.append(i)
            except json.JSONDecodeError:
                corrupt_lines.append(i)

    if corrupt_lines:
        issues.append(f"{len(corrupt_lines)} corrupt lines: {corrupt_lines[:5]}")
    if missing_fields:
        issues.append(f"{len(missing_fields)} entries missing 'metrics' field")

    status = "FAIL" if corrupt_lines else ("WARN" if missing_fields else "PASS")
    return {
        "name": "Experiment log",
        "status": status,
        "detail": f"{valid_lines}/{total_lines} valid entries",
        "issues": issues,
        "corrupt_lines": corrupt_lines,
        "fixable": len(corrupt_lines) > 0,
    }


def check_scripts(script_dir: str = ".") -> dict:
    """Check that required scripts exist and are syntactically valid."""
    issues = []
    checked = 0

    for script in REQUIRED_SCRIPTS:
        path = Path(script_dir) / script
        if not path.exists():
            issues.append(f"{script} not found")
            continue

        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=script)
            checked += 1
        except SyntaxError as e:
            issues.append(f"{script} has syntax error: {e.msg} (line {e.lineno})")

    status = "PASS" if not issues else ("WARN" if checked > 0 else "FAIL")
    return {
        "name": "Scripts",
        "status": status,
        "detail": f"{checked}/{len(REQUIRED_SCRIPTS)} scripts valid",
        "issues": issues,
    }


def check_disk_space(project_dir: str = ".", min_mb: int = MIN_DISK_MB) -> dict:
    """Check available disk space."""
    try:
        usage = shutil.disk_usage(project_dir)
        free_mb = usage.free / (1024 * 1024)
        total_mb = usage.total / (1024 * 1024)

        if free_mb < min_mb:
            return {
                "name": "Disk space",
                "status": "FAIL",
                "detail": f"{free_mb:.0f} MB remaining — below {min_mb} MB threshold",
                "issues": [f"Low disk space: {free_mb:.0f} MB free of {total_mb:.0f} MB"],
                "fix": "Run /turing:archive to reclaim space",
                "free_mb": round(free_mb),
            }

        return {
            "name": "Disk space",
            "status": "PASS",
            "detail": f"{free_mb:.0f} MB free",
            "issues": [],
            "free_mb": round(free_mb),
        }
    except OSError as e:
        return {
            "name": "Disk space",
            "status": "WARN",
            "detail": f"Could not check disk: {e}",
            "issues": [str(e)],
        }


def check_git_state(project_dir: str = ".") -> dict:
    """Check git working tree state."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
            cwd=project_dir,
        )
        if result.returncode != 0:
            return {
                "name": "Git state",
                "status": "WARN",
                "detail": "Not a git repository or git not available",
                "issues": [],
            }

        modified = result.stdout.strip().split("\n") if result.stdout.strip() else []
        issues = []

        # Check if critical files are modified
        critical = {"evaluate.py", "prepare.py"}
        for line in modified:
            if len(line) >= 3:
                filepath = line[3:].strip()
                if any(c in filepath for c in critical):
                    issues.append(f"Uncommitted changes to {filepath} — evaluation integrity at risk")

        status = "WARN" if issues else "PASS"
        detail = "Working tree clean" if not modified else f"{len(modified)} modified files"

        return {
            "name": "Git state",
            "status": status,
            "detail": detail,
            "issues": issues,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {
            "name": "Git state",
            "status": "WARN",
            "detail": "Git check skipped (timeout or not available)",
            "issues": [],
        }


# --- Fix Operations ---


def fix_corrupt_log(log_path: str = DEFAULT_LOG_PATH) -> dict:
    """Remove corrupt lines from experiment log."""
    path = Path(log_path)
    if not path.exists():
        return {"fixed": False, "reason": "Log not found"}

    valid_lines = []
    removed = 0

    with open(path) as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                json.loads(line_stripped)
                valid_lines.append(line)
            except json.JSONDecodeError:
                removed += 1

    if removed > 0:
        # Backup first
        backup = path.with_suffix(".jsonl.bak")
        shutil.copy2(path, backup)
        with open(path, "w") as f:
            f.writelines(valid_lines)
        return {"fixed": True, "removed": removed, "backup": str(backup)}

    return {"fixed": False, "reason": "No corrupt lines found"}


# --- Full Doctor ---


def run_doctor(
    config_path: str = "config.yaml",
    log_path: str = DEFAULT_LOG_PATH,
    fix: bool = False,
    verbose: bool = False,
) -> dict:
    """Run all diagnostic checks.

    Args:
        config_path: Path to config.yaml.
        log_path: Path to experiment log.
        fix: If True, auto-fix safe issues.
        verbose: Include detailed info.

    Returns:
        Doctor report with all check results and score.
    """
    checks = [
        check_environment(),
        check_dependencies(),
        check_config(config_path),
        check_experiment_log(log_path),
        check_scripts(),
        check_disk_space(),
        check_git_state(),
    ]

    # Apply fixes if requested
    fixes_applied = []
    if fix:
        log_check = next((c for c in checks if c["name"] == "Experiment log"), None)
        if log_check and log_check.get("fixable"):
            fix_result = fix_corrupt_log(log_path)
            if fix_result.get("fixed"):
                fixes_applied.append(f"Removed {fix_result['removed']} corrupt log entries (backup: {fix_result['backup']})")
                # Re-run log check
                for i, c in enumerate(checks):
                    if c["name"] == "Experiment log":
                        checks[i] = check_experiment_log(log_path)
                        break

    # Compute score
    passed = sum(1 for c in checks if c["status"] == "PASS")
    warned = sum(1 for c in checks if c["status"] == "WARN")
    failed = sum(1 for c in checks if c["status"] == "FAIL")
    total = len(checks)

    return {
        "checks": checks,
        "score": {"passed": passed, "warned": warned, "failed": failed, "total": total},
        "fixes_applied": fixes_applied,
        "overall": "HEALTHY" if failed == 0 and warned == 0 else ("DEGRADED" if failed == 0 else "UNHEALTHY"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# --- Report Formatting ---


def save_doctor_report(report: dict, output_dir: str = "experiments/doctor") -> Path:
    """Save doctor report to YAML."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filepath = out_path / f"doctor-{ts}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    return filepath


def format_doctor_report(report: dict) -> str:
    """Format doctor report as readable text."""
    lines = ["# Turing Doctor Report", ""]

    status_icons = {"PASS": "✓ PASS ", "WARN": "⚠ WARN ", "FAIL": "✗ FAIL "}

    for check in report.get("checks", []):
        icon = status_icons.get(check["status"], "? ")
        lines.append(f"{icon} {check['name']} ({check.get('detail', '')})")
        for issue in check.get("issues", []):
            lines.append(f"         {issue}")
        fix = check.get("fix")
        if fix:
            lines.append(f"         Fix: {fix}")

    score = report.get("score", {})
    lines.extend([
        "",
        f"Score: {score.get('passed', 0)}/{score.get('total', 0)} pass, "
        f"{score.get('warned', 0)} warning{'s' if score.get('warned', 0) != 1 else ''}, "
        f"{score.get('failed', 0)} failure{'s' if score.get('failed', 0) != 1 else ''}",
        f"Overall: {report.get('overall', 'UNKNOWN')}",
    ])

    fixes = report.get("fixes_applied", [])
    if fixes:
        lines.extend(["", "Fixes applied:"])
        for f in fixes:
            lines.append(f"  - {f}")

    lines.append("")
    lines.append(f"*Generated: {report.get('generated_at', 'N/A')}*")
    return "\n".join(lines)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Harness self-diagnosis — check environment, project, and resource health"
    )
    parser.add_argument("--fix", action="store_true", help="Auto-fix safe issues")
    parser.add_argument("--verbose", action="store_true", help="Show detailed info")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log", default=DEFAULT_LOG_PATH, help="Path to experiment log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    report = run_doctor(
        config_path=args.config,
        log_path=args.log,
        fix=args.fix,
        verbose=args.verbose,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_doctor_report(report))

    saved = save_doctor_report(report)
    if not args.json:
        print(f"\nSaved: {saved}")


if __name__ == "__main__":
    main()
