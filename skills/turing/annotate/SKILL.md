---
name: annotate
description: Retrospective experiment annotations — add human notes, tags, and context that automated metrics can't capture.
disable-model-invocation: true
argument-hint: "<exp-id> \"note\" [--tag fragile] | --list | --search \"keyword\""
allowed-tools: Read, Bash(*), Grep, Glob
---

Add context that experiment logs can't capture. "This only worked because the data was pre-sorted."

## Steps
1. **Activate environment:** `source .venv/bin/activate`
2. **Run:** `python scripts/experiment_annotations.py $ARGUMENTS`
3. **Operations:** add (text + tags), list (per-experiment or all), search (keyword or tag)
4. **Stored in:** `experiments/annotations.yaml`

## Examples
```
/turing:annotate exp-042 "Fragile — only works with specific preprocessing"
/turing:annotate exp-042 "Reviewer 2 requested this" --tag reviewer-requested
/turing:annotate --list
/turing:annotate --search "fragile"
```
