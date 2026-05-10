---
name: changelog
description: Model changelog generation — auto-generate human-readable progress narrative from experiment history for stakeholders.
disable-model-invocation: true
argument-hint: "[--since exp-id|date] [--audience technical|stakeholder]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Translate experiment logs into a narrative that PMs and stakeholders can read in 2 minutes.

## Steps
1. **Activate environment:** `source .venv/bin/activate`
2. **Run:** `python scripts/generate_changelog.py $ARGUMENTS`
3. **Audience:** technical (experiment IDs, configs), stakeholder (plain English, percentages)
4. **Saved output:** `paper/CHANGELOG.md`

## Examples
```
/turing:changelog                                # Full changelog
/turing:changelog --audience stakeholder         # Non-technical summary
/turing:changelog --since exp-042                # Since specific experiment
```
