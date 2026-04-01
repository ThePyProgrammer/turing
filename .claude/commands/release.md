---
name: release
description: Publish a new version of Turing — bumps versions, commits, tags, creates GitHub release, and updates the claude-plugins marketplace. Run with the new version number as argument.
argument-hint: "<version> (e.g., 1.3.0)"
---

You are the Turing release manager. You automate the full release pipeline for the turing plugin.

## Arguments

`$ARGUMENTS` must contain a semver version number (e.g., `1.3.0`). If empty, stop and ask for one.

Parse the version from `$ARGUMENTS`. Strip any leading `v` prefix — store the bare number (e.g., `1.3.0`) as `NEW_VERSION` and the prefixed form (`v1.3.0`) as `NEW_TAG`.

## Pre-flight Checks

Before doing anything:

1. **Verify clean working tree:** `git status --porcelain` must be empty. If not, stop and tell the user to commit or stash first.
2. **Verify on main branch:** `git branch --show-current` must be `main`. If not, stop.
3. **Verify tag doesn't exist:** `git tag -l v<NEW_VERSION>` must return empty. If the tag exists, stop.
4. **Detect previous version:** grep current version from `pyproject.toml` (line matching `version = "..."`). Store as `OLD_VERSION`.
5. **Show the user what will happen:**
   ```
   Release: v<OLD_VERSION> → v<NEW_VERSION>

   This will:
   1. Update version in pyproject.toml, package.json, .claude-plugin/plugin.json
   2. Update README.md and plugin description
   3. Commit, push to main
   4. Create tag v<NEW_VERSION> and push it
   5. Create GitHub release with generated release notes
   6. Update ../claude-plugins (PLUGINS.md + documentation/turing.md) and push

   Proceed?
   ```
   Wait for user confirmation before continuing.

## Step 1: Gather Changelog

Run `git log --oneline v<OLD_VERSION>..HEAD` to get all commits since the last release.

Categorize commits by their conventional commit prefix:
- `feat:` → Features
- `fix:` → Bug Fixes
- `test:` → Tests
- `docs:` → Documentation
- `chore:` → Maintenance

From the commit history and changed files, determine:
- **Headline feature** — the main thing this release adds (1 sentence)
- **New commands** — any new `/turing:*` commands added
- **New scripts** — any new `templates/scripts/*.py` files
- **New tests** — count of new test files or significant test additions
- **Stats delta** — compute new totals for commands, scripts, tests

To count commands: `ls commands/*.md | grep -v turing.md | wc -l`
To count scripts: `ls templates/scripts/*.py | grep -v __init__.py | wc -l`
To count tests: run `uv run pytest tests/ --co -q 2>/dev/null | tail -1`

## Step 1b: Update CHANGELOG.md

Read `CHANGELOG.md` and prepend a new entry at the top (after the header). Follow the existing format exactly:

```markdown
## [<NEW_VERSION>] — <YYYY-MM-DD> — <Headline>

### Added
- bullet per new command or feature

### Fixed (if applicable)
- bullet per bug fix

### Phase (if applicable)
- **N.M** Phase Name

**<test_count> tests | <command_count> commands | <script_count> scripts | <commit_count> commits**
```

Also add the release link at the bottom of the file:
```markdown
[<NEW_VERSION>]: https://github.com/ThePyProgrammer/turing/releases/tag/v<NEW_VERSION>
```

For patch releases (x.y.Z), keep it minimal — just the fix description and stats.

## Step 2: Update Version in Manifests

Update the version string in all three files. Use the Edit tool for each:

1. **`pyproject.toml`** — `version = "<OLD_VERSION>"` → `version = "<NEW_VERSION>"`
2. **`package.json`** — `"version": "<OLD_VERSION>"` → `"version": "<NEW_VERSION>"`
3. **`.claude-plugin/plugin.json`** — `"version": "<OLD_VERSION>"` → `"version": "<NEW_VERSION>"`

Also update the `description` field in `.claude-plugin/plugin.json` if the headline feature warrants it (e.g., update the command count, mention the new capability).

## Step 3: Update README.md

Read `README.md` and update:
- The architecture stats line (command count, script count, test count) if the numbers changed
- The directory tree comment counts if they changed
- Do NOT rewrite prose sections — only update numbers and add new feature sections if needed

The README uses a specific philosophical, essay-like style with extended metaphors. Any new sections must match this voice. When in doubt, keep changes minimal — numbers and stats only.

## Step 4: Commit and Push

```bash
git add pyproject.toml package.json .claude-plugin/plugin.json README.md CHANGELOG.md
git commit -m "chore: bump version to v<NEW_VERSION> across all manifests

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
git push origin main
```

If additional files were modified (e.g., README sections), include them in the same commit.

## Step 5: Create Tag and GitHub Release

Create the tag:
```bash
git tag v<NEW_VERSION>
git push origin v<NEW_VERSION>
```

Generate release notes in the established style (see previous releases with `gh release view v<OLD_VERSION> --repo ThePyProgrammer/turing` for voice/format reference). The release notes must include:

1. **Title:** `v<NEW_VERSION> — <headline feature>`
2. **"What's new" section** with the headline feature described in 1-2 paragraphs
3. **Subsections** for each significant addition (commands, scripts, integrations)
4. **"Numbers" section** — test count, commit count, command count, script count (with deltas from previous)
5. **"Full changelog" section** — bullet list of all notable changes

Create the release:
```bash
gh release create v<NEW_VERSION> --repo ThePyProgrammer/turing \
    --title "v<NEW_VERSION> — <headline>" \
    --notes "<generated notes>"
```

## Step 6: Update claude-plugins

The plugin marketplace lives at `../claude-plugins` (relative to the turing repo root).

1. **Verify it exists and is clean:**
   ```bash
   git -C ../claude-plugins status --porcelain
   git -C ../claude-plugins branch --show-current
   ```

2. **Update `PLUGINS.md`** — find the turing row and update version + description. Only touch the turing row. The description should be a single-line summary mentioning the headline capability and key stats (command count, agent count).

3. **Update `documentation/turing.md`** — this is the full plugin documentation page. Update:
   - Version number
   - Command count in heading
   - Add new commands to the appropriate table
   - Add a "Key Features" subsection for the headline feature (with version tag)
   - Update test count in Architecture section
   - Update intellectual heritage if new external projects were integrated
   - Keep the same structure and voice as the existing document

4. **Commit and push:**
   ```bash
   cd ../claude-plugins
   git add PLUGINS.md documentation/turing.md
   git commit -m "docs: update turing to v<NEW_VERSION> — <headline>

   Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
   git push origin main
   ```

**CRITICAL:** Do NOT modify any other plugin's files, descriptions, or versions. Only touch the turing entries.

## Step 7: Summary

Print a release summary:

```
Turing v<NEW_VERSION> released.

  Manifests:    pyproject.toml, package.json, plugin.json ✓
  Pushed:       main branch ✓
  Tag:          v<NEW_VERSION> ✓
  Release:      https://github.com/ThePyProgrammer/turing/releases/tag/v<NEW_VERSION>
  Marketplace:  claude-plugins updated ✓

  Changes: <N> commits, <headline>
```

## Error Handling

- If any `git push` fails, stop and report the error. Do not continue to subsequent steps.
- If the GitHub release creation fails, the tag is already pushed — report this and suggest `gh release create` manually.
- If `../claude-plugins` doesn't exist or isn't clean, skip Step 6 and tell the user to update it manually.
- Never force-push. Never amend published commits. Never skip hooks.
