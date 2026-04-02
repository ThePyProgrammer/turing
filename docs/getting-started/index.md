---
title: Installation
description: Install the Turing ML research harness as a Claude Code plugin via npm or local path.
---

# Installation

## Prerequisites

- **Claude Code CLI** -- installed and authenticated
- **Node.js** -- v18 or later
- **Python 3.10+** -- with `pip` or `uv` available on PATH

## Option 1: npm (recommended)

```bash
npm install -g claude-turing && claude-turing install --global && claude-turing verify
```

This does three things:

1. Installs the `claude-turing` CLI globally
2. Registers the plugin with Claude Code (writes to `~/.claude/plugins.json`)
3. Runs verification to confirm all commands, agents, and Python dependencies resolve

## Option 2: Local path

If you are developing Turing or prefer a local install:

```bash
claude plugin add /path/to/turing
```

This registers the plugin directly from a local checkout. Changes to command files take effect immediately -- no reinstall needed.

## Verification

Run the verifier to confirm everything works:

```bash
claude-turing verify
```

Expected output:

```
Turing ML Research Harness v4.4.0
==================================

Commands .................. 74/74 registered
Agents .................... 2/2 available
  @ml-researcher ......... Read/Write/Edit/Bash
  @ml-evaluator .......... Read/Bash (read-only)
Python dependencies ....... OK
  python .................. 3.12.3
  xgboost ................ 2.1.1
  scikit-learn ........... 1.5.2
  pandas ................. 2.2.3
Templates ................. 8/8 present
Plugin registration ....... OK

All checks passed.
```

If any check fails, the verifier prints the specific remediation step. Common issues:

| Symptom | Fix |
|---------|-----|
| `Python dependencies ... FAIL` | Run `pip install -r requirements.txt` or `uv pip install -r requirements.txt` in the Turing directory |
| `Plugin registration ... FAIL` | Run `claude-turing install --global` again |
| `Commands ... N/74 registered` | Update to latest version: `npm update -g claude-turing` |

## Uninstall

```bash
claude-turing uninstall --global
npm uninstall -g claude-turing
```
