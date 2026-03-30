#!/usr/bin/env python3
"""Post-scaffolding placeholder verification.

Scans all scaffolded files for unreplaced {{PLACEHOLDER}} markers and
reports any that remain. Returns exit code 1 if any are found.

This converts silent-wrong-behavior failures into loud-and-immediate errors.
A missed placeholder like {{TARGET_METRIC}} produces valid Python that silently
does the wrong thing — this script catches it before the first experiment runs.

Usage:
    python scripts/verify_placeholders.py [directory]

    directory: path to scan (default: current directory)

Exit codes:
    0 = all placeholders replaced
    1 = unreplaced placeholders found
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Files to scan for placeholders (relative to project root)
SCANNABLE_EXTENSIONS = {".py", ".yaml", ".yml", ".md", ".toml", ".sh", ".txt"}

# Known placeholder pattern
PLACEHOLDER_RE = re.compile(r"\{\{([A-Z_]+)\}\}")

# Known valid placeholders (from config/defaults.yaml)
KNOWN_PLACEHOLDERS = {
    "PROJECT_NAME",
    "TARGET_METRIC",
    "TASK_DESCRIPTION",
    "ML_DIR",
    "DATA_SOURCE",
    "METRIC_DIRECTION",
}


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Scan a single file for unreplaced placeholders.

    Returns list of (line_number, placeholder_name, line_content) tuples.
    """
    findings: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError):
        return findings

    for i, line in enumerate(text.splitlines(), start=1):
        for match in PLACEHOLDER_RE.finditer(line):
            placeholder = match.group(1)
            if placeholder in KNOWN_PLACEHOLDERS:
                findings.append((i, placeholder, line.strip()))

    return findings


def scan_directory(root: Path) -> dict[Path, list[tuple[int, str, str]]]:
    """Scan all files in directory for unreplaced placeholders.

    Returns dict mapping file paths to their findings.
    """
    results: dict[Path, list[tuple[int, str, str]]] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in SCANNABLE_EXTENSIONS:
            continue
        # Skip venv, node_modules, git
        parts = path.parts
        if any(p in (".venv", "node_modules", ".git", "__pycache__") for p in parts):
            continue

        findings = scan_file(path)
        if findings:
            results[path] = findings

    return results


def main() -> None:
    """CLI entry point."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    results = scan_directory(root)

    if not results:
        print("All placeholders replaced successfully.")
        sys.exit(0)

    # Report findings
    total = sum(len(findings) for findings in results.values())
    print(f"Found {total} unreplaced placeholder(s) in {len(results)} file(s):\n")

    for path, findings in results.items():
        rel_path = path.relative_to(root) if path.is_relative_to(root) else path
        print(f"  {rel_path}:")
        for line_num, placeholder, line_content in findings:
            print(f"    line {line_num}: {{{{{placeholder}}}}} — {line_content[:80]}")
        print()

    print("Fix: replace all {{PLACEHOLDER}} markers with project-specific values.")
    print("See config.yaml and program.md for placeholder descriptions.")
    sys.exit(1)


if __name__ == "__main__":
    main()
