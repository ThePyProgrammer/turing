---
name: doctor
description: Harness self-diagnosis — check environment, project, resources, and git state. Auto-fix common issues.
argument-hint: "[--fix] [--verbose]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Is Turing healthy? Check everything and get a score.

## Steps
1. `source .venv/bin/activate`
2. `python scripts/harness_doctor.py $ARGUMENTS`
3. **Saved:** `experiments/doctor/`

## Checks
- **Environment:** Python version, venv status
- **Dependencies:** all required packages importable
- **Config:** config.yaml valid with required fields
- **Experiment log:** JSONL integrity, corrupt line detection
- **Scripts:** train.py, prepare.py, evaluate.py exist and parse
- **Disk space:** warn if <1GB free
- **Git state:** uncommitted changes to critical files
- **Claude hooks:** `.claude/settings.local.json` hook group schema; `--fix` migrates legacy bare command hooks

## Examples
```
/turing:doctor
/turing:doctor --fix
/turing:doctor --verbose --json
```
