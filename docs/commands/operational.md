---
title: "Operational Intelligence"
description: "Failure postmortems, harness self-diagnosis, strategic research planning, and pre-flight resource checks."
---

# Operational Intelligence

Commands for keeping the research harness healthy and the research direction strategic -- diagnosing why experiments stopped improving, verifying the environment is sound, planning the next batch of experiments by ROI, and checking system resources before training.

---

### `/turing:postmortem` -- Failure postmortem

When experiments stop improving, find out why. Diagnoses search space exhaustion (micro-tuning parameters that do not matter), systematic config errors (all experiments share a bad common config), data issues (all model types fail similarly), metric ceilings (near theoretical maximum), and noise floors (improvements within seed variance). Each diagnosis comes with actionable next steps.

**Syntax:** `/turing:postmortem [--window 10] [--auto-trigger 5] [--json]`

**Examples:**
```
/turing:postmortem
/turing:postmortem --window 15
/turing:postmortem --json
```

---

### `/turing:doctor` -- Harness self-diagnosis

Is Turing healthy? Checks the Python environment, dependency imports, config validity, experiment log integrity, script existence, disk space, and git state. Reports a health score and, with `--fix`, auto-repairs common issues. Run this first when something seems off.

**Syntax:** `/turing:doctor [--fix] [--verbose] [--json]`

**Examples:**
```
/turing:doctor
/turing:doctor --fix
/turing:doctor --verbose --json
```

---

### `/turing:plan` -- Research planning assistant

Design the next N experiments strategically, not randomly. Analyzes experiment history to compute per-family ROI, adjusts strategy priorities based on project state and goal, and allocates budget across feature engineering, model search, ensemble, calibration, and verification. Generates a phased plan with specific experiment descriptions.

**Syntax:** `/turing:plan [--budget 20] [--goal "maximize F1 for production"] [--json]`

**Examples:**
```
/turing:plan --budget 20
/turing:plan --budget 10 --goal "maximize F1 for production deployment"
/turing:plan --budget 30 --json
```

---

### `/turing:preflight` -- Pre-flight resource check

Check whether the current system has enough resources to run the planned experiment before hitting an OOM error 3 hours in. Estimates VRAM, RAM, and disk requirements based on model type, parameter count, and batch size, then compares against available system resources. Issues PASS, WARN, or FAIL with specific mitigation suggestions (reduce batch size, use fp16, enable gradient checkpointing, switch to CPU-friendly model).

**Syntax:** `/turing:preflight [--model-type torch|xgboost|...] [--params 10M] [--batch-size 32] [--precision fp16] [--dataset path] [--json]`

**Examples:**
```
/turing:preflight
/turing:preflight --model-type transformer --params 350M --batch-size 16 --precision fp16
/turing:preflight --model-type xgboost --dataset data/train.csv
/turing:preflight --json
```
