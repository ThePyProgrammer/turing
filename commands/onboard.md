---
name: onboard
description: Project onboarding — generate a walkthrough for new collaborators. Task, history, decisions, next steps.
disable-model-invocation: true
argument-hint: "[--audience researcher|engineer|stakeholder] [--depth brief|full]"
allowed-tools: Read, Bash(*), Grep, Glob
---

5-minute read that replaces a 1-hour onboarding meeting.

## Steps
1. `source .venv/bin/activate`
2. `python scripts/generate_onboarding.py $ARGUMENTS`
3. **Saved:** `ONBOARDING.md`

## Examples
```
/turing:onboard
/turing:onboard --audience engineer --depth brief
```
