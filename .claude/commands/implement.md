---
name: implement
description: Implement a roadmap phase systematically with maximum commit granularity. Each file, test, and integration point gets its own atomic commit.
argument-hint: "<phase-number> <version> (e.g., 16 v2.3.0)"
---

You are the Turing phase implementor. You implement roadmap phases with **maximum commit granularity** — every discrete unit of work gets its own atomic commit.

## Arguments

Parse `$ARGUMENTS` for:
- **Phase number** (e.g., `16`) — the phase to implement
- **Version** (e.g., `v2.3.0` or `2.3.0`) — the target version

If either is missing, stop and ask.

## Step 0: Read the Roadmap

Read `ROADMAP.md` and find the section for the requested phase. Extract:
- Phase title and tagline
- All sub-phases (e.g., 16.1, 16.2, 16.3)
- Per sub-phase: command name, script name, implementation details, dependencies, acceptance criteria
- The implementation order table entries for status updates

If the phase doesn't exist or has no detailed spec, stop and tell the user.

## Step 1: Create Task Breakdown

Use TaskCreate to create one task per deliverable. A typical phase produces 12-20 tasks:

For **each sub-phase** (e.g., 16.1, 16.2, 16.3), create tasks for:
1. Core Python script (`templates/scripts/<name>.py`)
2. Command skill (`commands/<name>.md`)
3. Unit tests (`tests/test_<name>.py`)
4. Edge case tests (`tests/test_<name>_edge_cases.py`)

Then create shared tasks for:
- Router update (`commands/turing.md`)
- Manifest registration (`install.js`, `verify.js`, `scaffold.py`)
- Brief integration (`generate_brief.py`) — if the phase adds data that should appear in `/turing:brief`
- Config files — if the phase requires new config (e.g., `config/watch_alerts.yaml`)
- ROADMAP status update
- README stats update
- Version bump to target version

## Step 2: Implement with Maximum Commit Granularity

**CRITICAL RULE: One commit per file. One commit per logical unit. Never batch multiple files into one commit unless they are fundamentally inseparable.**

The commit pattern for each sub-phase is:

```
1. feat: add <script>.py — <one-line description>
   → commit JUST the script file

2. feat: add /turing:<command> command skill
   → commit JUST the command .md file

3. test: add unit tests for <script>.py
   → commit JUST the test file

4. test: add edge case tests for <script>.py
   → commit JUST the edge case test file
```

Then for shared infrastructure:

```
5. feat: add <command> routes to main router
   → commit JUST turing.md

6. feat: register Phase N commands in installer, verifier, and scaffold
   → commit install.js + verify.js + scaffold.py together (they're inseparable)

7. feat: integrate <feature> into research briefing
   → commit JUST generate_brief.py

8. feat: add <config>.yaml — <description>
   → commit JUST the config file

9. docs: mark Phase N as DONE in roadmap
   → commit JUST ROADMAP.md

10. docs: add Phase N commands and update stats in README
    → commit JUST README.md

11. chore: bump version to vX.Y.Z across all manifests
    → commit pyproject.toml + package.json + plugin.json + README.md (version bump is one logical unit)
```

**This means a typical 3-sub-phase implementation produces 15-25 commits.** That is the goal. Do not combine commits to "save time." Each commit should be independently meaningful and revertable.

### Commit message conventions

- `feat:` for new scripts, commands, config, integrations
- `test:` for test files
- `docs:` for ROADMAP, README updates
- `chore:` for version bumps
- `fix:` for bug fixes found during implementation
- Every commit message ends with: `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`

### Test execution

Run tests with `uv run pytest` (never `python -m pytest`). Run the new tests after each test commit to verify they pass. Run the manifest consistency tests after registering in manifests.

## Step 3: Verify

After all commits:

1. Run the full test suite: `uv run pytest tests/ -q`
2. Run manifest consistency: `uv run pytest tests/test_manifest_consistency.py -v`
3. Count stats: commands (`ls commands/*.md | grep -v turing.md | wc -l`), scripts (`ls templates/scripts/*.py | grep -v __init__.py | wc -l`), tests (`uv run pytest tests/ --co -q 2>/dev/null | tail -1`)
4. Show the commit log: `git log --oneline v<PREV_VERSION>..HEAD`
5. Report the summary to the user

## Step 4: Do NOT Release

Implementation and release are separate operations. After implementing, report the summary and stop. The user will run `/release <version>` separately when ready.

## Conventions

- Follow existing code patterns exactly — read 1-2 existing scripts in the same category before writing new ones
- Use `from scripts.turing_io import load_config, load_experiments` for shared IO
- Save outputs to `experiments/<category>/` directories
- Format reports as markdown with `format_<name>_report()` functions
- CLI entry points use argparse with `--json`, `--config`, `--log` standard flags
- Every script has a `main()` function and `if __name__ == "__main__": main()`
- Test files mirror the structure: `test_<script_name>.py` with sections per function
