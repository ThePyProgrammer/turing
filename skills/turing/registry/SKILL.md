---
name: registry
description: Model registry — track, promote, and govern the model lifecycle from candidate to production.
argument-hint: "[list|register|promote|demote|archive|history] [exp-id] [stage]"
allowed-tools: Read, Bash(*), Grep, Glob
---

Track which model is production, staging, candidate, or archived. Promotion requires passing gates.

## Steps
1. `source .venv/bin/activate`
2. `python scripts/model_lifecycle.py $ARGUMENTS`
3. **Registry:** `experiments/registry.yaml`

## Promotion gates
- **candidate → staging:** regression check + seed study must PASS
- **staging → production:** audit + calibration check must PASS
- Use `--force` to skip gate checks

## Examples
```
/turing:registry list
/turing:registry register exp-095 --version v4.1
/turing:registry promote exp-089 staging
/turing:registry promote exp-089 production --force
/turing:registry demote exp-078 staging --reason "latency regression"
/turing:registry archive exp-042 --reason "superseded by v4"
/turing:registry history
/turing:registry history exp-089
```
