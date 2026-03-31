#!/usr/bin/env python3
"""Archive old experiment branches and model artifacts to reclaim disk space.

Nothing is deleted. Experiment branches are preserved as lightweight git tags
(archive/exp-NNN) before the branch is removed. Model artifacts are moved to
models/_archive/ — not deleted.

Archival policy:
  - Git branches: merged branches whose tip commit is older than N days get
    tagged as archive/<branch-name> and then deleted locally.
  - Model files: all files in models/ (excluding models/best/ and
    models/_archive/) are sorted by modification time. The N most recent are
    kept in place; the rest are moved to models/_archive/.

Usage:
    python scripts/cleanup.py                        # Archive with defaults
    python scripts/cleanup.py --dry-run              # Preview only (no changes)
    python scripts/cleanup.py --keep-models 10       # Keep 10 most recent models
    python scripts/cleanup.py --branch-age 60        # Archive branches older than 60 days
    python scripts/cleanup.py --compress             # Also compress models/_archive/
    python scripts/cleanup.py --models-dir path/     # Override models directory
    python scripts/cleanup.py --log-path path/       # Override experiments log path

Exit codes:
    0  — success (or dry-run completed)
    1  — fatal error (e.g. not in a git repo)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class BranchInfo(NamedTuple):
    name: str
    commit_hash: str
    commit_date: datetime
    age_days: int


class ModelFileInfo(NamedTuple):
    path: Path
    mtime: datetime
    size_bytes: int


class ArchiveSummary(NamedTuple):
    branches_archived: list[str]
    branches_skipped: list[str]
    models_moved: list[str]
    bytes_reclaimed_by_move: int       # always 0 — files are moved, not deleted
    archive_compressed: bool
    compressed_path: str | None
    dry_run: bool


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: Path | None = None) -> str:
    """Run a git command and return stdout. Raises RuntimeError on failure."""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def find_git_root() -> Path | None:
    """Return the root of the current git repository, or None."""
    try:
        root = _run_git(["rev-parse", "--show-toplevel"])
        return Path(root)
    except RuntimeError:
        return None


def get_merged_branches(git_root: Path) -> list[str]:
    """Return local branch names that have been merged into HEAD.

    Excludes HEAD itself and the current branch.
    """
    try:
        raw = _run_git(["branch", "--merged", "HEAD", "--format=%(refname:short)"], cwd=git_root)
    except RuntimeError:
        return []

    current = ""
    try:
        current = _run_git(["branch", "--show-current"], cwd=git_root)
    except RuntimeError:
        pass

    branches = []
    for line in raw.splitlines():
        name = line.strip()
        if name and name != current and name not in ("main", "master", "HEAD"):
            branches.append(name)
    return branches


def get_branch_commit_date(branch: str, git_root: Path) -> datetime | None:
    """Return the author date of the tip commit for the branch."""
    try:
        ts = _run_git(
            ["log", "-1", "--format=%aI", branch], cwd=git_root
        )
        if not ts:
            return None
        # ISO 8601 with timezone — fromisoformat handles this in Python 3.11+;
        # fall back to manual parsing for older Pythons.
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            # Strip trailing timezone if parsing fails
            return datetime.fromisoformat(ts[:19]).replace(tzinfo=timezone.utc)
    except RuntimeError:
        return None


def get_branch_commit_hash(branch: str, git_root: Path) -> str | None:
    """Return the full commit hash at the tip of a branch."""
    try:
        return _run_git(["rev-parse", branch], cwd=git_root)
    except RuntimeError:
        return None


def tag_exists(tag_name: str, git_root: Path) -> bool:
    """Return True if the given tag already exists in the repo."""
    try:
        _run_git(["rev-parse", "--verify", f"refs/tags/{tag_name}"], cwd=git_root)
        return True
    except RuntimeError:
        return False


def create_archive_tag(branch: str, commit_hash: str, git_root: Path) -> str:
    """Create a lightweight tag archive/<branch> pointing at commit_hash.

    Returns the tag name. Raises RuntimeError on failure.
    """
    tag_name = f"archive/{branch}"
    if tag_exists(tag_name, git_root):
        # Overwrite only if it already points to the same commit
        existing_hash = _run_git(["rev-parse", tag_name], cwd=git_root)
        if existing_hash == commit_hash:
            return tag_name
        # Different commit — append a counter to avoid collision
        for i in range(2, 100):
            candidate = f"archive/{branch}-{i}"
            if not tag_exists(candidate, git_root):
                tag_name = candidate
                break

    _run_git(["tag", tag_name, commit_hash], cwd=git_root)
    return tag_name


def delete_branch(branch: str, git_root: Path) -> None:
    """Delete a local branch. The branch must already be tagged."""
    _run_git(["branch", "-d", branch], cwd=git_root)


# ---------------------------------------------------------------------------
# Branch archival
# ---------------------------------------------------------------------------

def collect_stale_branches(
    git_root: Path,
    max_age_days: int,
) -> tuple[list[BranchInfo], list[BranchInfo]]:
    """Split merged branches into stale (archive candidates) and recent.

    Returns (stale, recent).
    """
    merged = get_merged_branches(git_root)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)

    stale: list[BranchInfo] = []
    recent: list[BranchInfo] = []

    for branch in merged:
        commit_date = get_branch_commit_date(branch, git_root)
        if commit_date is None:
            continue
        # Make commit_date timezone-aware if it isn't
        if commit_date.tzinfo is None:
            commit_date = commit_date.replace(tzinfo=timezone.utc)
        age_days = (now - commit_date).days
        commit_hash = get_branch_commit_hash(branch, git_root) or ""
        info = BranchInfo(
            name=branch,
            commit_hash=commit_hash,
            commit_date=commit_date,
            age_days=age_days,
        )
        if commit_date < cutoff:
            stale.append(info)
        else:
            recent.append(info)

    stale.sort(key=lambda b: b.commit_date)
    return stale, recent


def archive_branches(
    stale: list[BranchInfo],
    git_root: Path,
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    """Archive stale branches: tag them and delete the branch ref.

    Returns (archived_names, skipped_names).
    """
    archived: list[str] = []
    skipped: list[str] = []

    for branch_info in stale:
        if not branch_info.commit_hash:
            print(
                f"  [skip] {branch_info.name} — could not resolve commit hash",
                file=sys.stderr,
            )
            skipped.append(branch_info.name)
            continue

        tag_name = f"archive/{branch_info.name}"
        if dry_run:
            print(f"  [dry-run] Would tag {branch_info.name} -> {tag_name} and delete branch")
            archived.append(branch_info.name)
            continue

        try:
            actual_tag = create_archive_tag(
                branch_info.name, branch_info.commit_hash, git_root
            )
            delete_branch(branch_info.name, git_root)
            print(f"  Archived branch: {branch_info.name} -> tag {actual_tag}")
            archived.append(branch_info.name)
        except RuntimeError as exc:
            print(f"  [skip] {branch_info.name} — {exc}", file=sys.stderr)
            skipped.append(branch_info.name)

    return archived, skipped


# ---------------------------------------------------------------------------
# Model artifact archival
# ---------------------------------------------------------------------------

def collect_model_files(models_dir: Path) -> list[ModelFileInfo]:
    """Return all model files directly under models_dir (non-recursive).

    Excludes:
      - models/best/       (current best — never touch)
      - models/_archive/   (already archived)
      - directories
    """
    if not models_dir.exists():
        return []

    files: list[ModelFileInfo] = []
    for entry in models_dir.iterdir():
        if entry.is_dir():
            continue
        stat = entry.stat()
        files.append(
            ModelFileInfo(
                path=entry,
                mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                size_bytes=stat.st_size,
            )
        )
    # Also collect from immediate subdirectories that are NOT best/ or _archive/
    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name in ("best", "_archive"):
            continue
        for entry in subdir.rglob("*"):
            if entry.is_file():
                stat = entry.stat()
                files.append(
                    ModelFileInfo(
                        path=entry,
                        mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                        size_bytes=stat.st_size,
                    )
                )

    # Sort newest first
    files.sort(key=lambda f: f.mtime, reverse=True)
    return files


def archive_models(
    models_dir: Path,
    keep_n: int,
    dry_run: bool,
) -> list[str]:
    """Move old model files to models/_archive/.

    Keeps the N most-recently-modified files in place.
    Returns list of moved file paths (relative to models_dir).
    """
    all_files = collect_model_files(models_dir)

    if not all_files:
        return []

    # Files to keep (newest N) vs. archive (the rest)
    to_keep = all_files[:keep_n]
    to_archive = all_files[keep_n:]

    if not to_archive:
        return []

    archive_dir = models_dir / "_archive"
    moved: list[str] = []

    for file_info in to_archive:
        rel = file_info.path.relative_to(models_dir)
        dest = archive_dir / rel

        if dry_run:
            print(f"  [dry-run] Would move models/{rel} -> models/_archive/{rel}")
            moved.append(str(rel))
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(file_info.path), str(dest))
            print(f"  Archived model: models/{rel} -> models/_archive/{rel}")
            moved.append(str(rel))
        except OSError as exc:
            print(f"  [skip] {file_info.path} — {exc}", file=sys.stderr)

    if to_keep:
        print(f"  Kept {len(to_keep)} most recent model(s) in models/:")
        for fi in to_keep:
            rel = fi.path.relative_to(models_dir)
            print(f"    models/{rel}  ({fi.mtime.strftime('%Y-%m-%d')})")

    return moved


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def compress_archive(models_dir: Path, dry_run: bool) -> str | None:
    """Tar.gz the models/_archive/ directory.

    The archive is placed at models/_archive.tar.gz (alongside models/).
    Returns the path to the .tar.gz, or None if nothing to compress.
    """
    archive_dir = models_dir / "_archive"
    if not archive_dir.exists() or not any(archive_dir.rglob("*")):
        print("  Nothing in models/_archive/ to compress.")
        return None

    out_path = models_dir.parent / "models_archive.tar.gz"

    if dry_run:
        print(f"  [dry-run] Would compress models/_archive/ -> {out_path}")
        return str(out_path)

    try:
        with tarfile.open(str(out_path), "w:gz") as tar:
            tar.add(str(archive_dir), arcname="_archive")
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Compressed models/_archive/ -> {out_path} ({size_mb:.1f} MB)")
        return str(out_path)
    except OSError as exc:
        print(f"  [warn] Compression failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(summary: ArchiveSummary) -> None:
    """Print a human-readable archive summary."""
    mode = "DRY RUN — " if summary.dry_run else ""
    print()
    print(f"{'=' * 60}")
    print(f"  {mode}Archive Summary")
    print(f"{'=' * 60}")

    # Branches
    print(f"\n  Git branches:")
    if summary.branches_archived:
        print(f"    Archived  : {len(summary.branches_archived)}")
        for name in summary.branches_archived:
            print(f"      - {name} -> archive/{name}")
    else:
        print("    No branches archived.")
    if summary.branches_skipped:
        print(f"    Skipped   : {len(summary.branches_skipped)} (see warnings above)")

    # Models
    print(f"\n  Model artifacts:")
    if summary.models_moved:
        print(f"    Moved to _archive/: {len(summary.models_moved)}")
        for name in summary.models_moved:
            print(f"      - {name}")
    else:
        print("    No model files needed archiving.")

    # Compression
    print(f"\n  Compression:")
    if summary.archive_compressed and summary.compressed_path:
        print(f"    Archive written to: {summary.compressed_path}")
    elif summary.archive_compressed and not summary.compressed_path:
        print("    Compression requested but no archive content found.")
    else:
        print("    Not requested (use --compress to enable).")

    print()
    if summary.dry_run:
        print("  NOTE: Dry-run mode — no changes were made.")
    print(f"{'=' * 60}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Archive old experiment branches and model artifacts.\n\n"
            "Branches are preserved as git tags (archive/<name>) before "
            "deletion. Models are MOVED to models/_archive/ — never deleted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what WOULD be archived without making any changes.",
    )
    parser.add_argument(
        "--keep-models",
        type=int,
        default=5,
        metavar="N",
        help="Number of most-recent model files to keep in place (default: 5).",
    )
    parser.add_argument(
        "--branch-age",
        type=int,
        default=30,
        metavar="DAYS",
        help="Archive merged branches whose tip commit is older than DAYS days (default: 30).",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Tar.gz models/_archive/ after archiving.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        metavar="PATH",
        help="Path to the models directory (default: models/).",
    )
    parser.add_argument(
        "--no-branches",
        action="store_true",
        default=False,
        help="Skip branch archival entirely.",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        default=False,
        help="Skip model archival entirely.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] No changes will be made.\n")

    # -----------------------------------------------------------------------
    # Git root
    # -----------------------------------------------------------------------
    git_root = find_git_root()
    if git_root is None:
        print(
            "Error: Not inside a git repository. Cannot archive branches.",
            file=sys.stderr,
        )
        # We can still archive models — don't hard-fail if only models are requested
        if not args.no_branches:
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Resolve models directory relative to cwd (or git root if available)
    # -----------------------------------------------------------------------
    base_dir = git_root if git_root is not None else Path.cwd()
    models_dir = (base_dir / args.models_dir).resolve()

    # -----------------------------------------------------------------------
    # Branch archival
    # -----------------------------------------------------------------------
    branches_archived: list[str] = []
    branches_skipped: list[str] = []

    if not args.no_branches and git_root is not None:
        print(f"Scanning for merged branches older than {args.branch_age} day(s)...")
        stale, recent = collect_stale_branches(git_root, args.branch_age)

        if not stale and not recent:
            print("  No merged branches found (besides main/master).")
        else:
            print(f"  Found {len(stale)} stale + {len(recent)} recent merged branch(es).")

        if stale:
            archived, skipped = archive_branches(stale, git_root, args.dry_run)
            branches_archived.extend(archived)
            branches_skipped.extend(skipped)
        else:
            print(f"  No merged branches older than {args.branch_age} days to archive.")

    # -----------------------------------------------------------------------
    # Model archival
    # -----------------------------------------------------------------------
    models_moved: list[str] = []

    if not args.no_models:
        if not models_dir.exists():
            print(f"\nModels directory '{models_dir}' does not exist — skipping model archival.")
        else:
            print(f"\nScanning {models_dir} for model files (keeping {args.keep_models} most recent)...")
            all_files = collect_model_files(models_dir)
            if not all_files:
                print("  No model files found outside best/ and _archive/.")
            else:
                print(f"  Found {len(all_files)} model file(s).")
                moved = archive_models(models_dir, args.keep_models, args.dry_run)
                models_moved.extend(moved)

    # -----------------------------------------------------------------------
    # Compression
    # -----------------------------------------------------------------------
    compressed_path: str | None = None
    if args.compress:
        print("\nCompressing models/_archive/...")
        compressed_path = compress_archive(models_dir, args.dry_run)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summary = ArchiveSummary(
        branches_archived=branches_archived,
        branches_skipped=branches_skipped,
        models_moved=models_moved,
        bytes_reclaimed_by_move=0,  # files moved, not deleted — disk usage unchanged
        archive_compressed=args.compress,
        compressed_path=compressed_path,
        dry_run=args.dry_run,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
