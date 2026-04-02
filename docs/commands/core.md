---
title: "Core Loop"
description: "The five commands that form the core experiment loop: init, train, sweep, status, and compare."
---

# Core Loop

These five commands form the backbone of the Turing experiment lifecycle. `init` scaffolds the project, `train` runs the autonomous loop, `sweep` handles systematic hyperparameter search, and `status`/`compare` provide observation without modification.

---

### `/turing:init`: Scaffold a New ML Project

Initialize a new ML project with the Turing autoresearch harness. Scaffolds the full experiment infrastructure: immutable evaluation pipeline, agent-editable training code, structured logging, convergence detection hooks, and a Python virtual environment.

The key architectural idea is the separation between the **measurement apparatus** (read-only `prepare.py`, `evaluate.py`) and the **hypothesis space** (agent-editable `train.py`). This makes autonomous experimentation trustworthy.

**Syntax:** `/turing:init [project_name] [--plan]`

The command runs an interactive setup asking for: project name, target metric, metric direction, task description, ML directory, and data source. All values can also be passed as JSON arguments.

The `--plan` flag generates a `RESEARCH_PLAN.md` after scaffolding, giving the agent strategic direction grounded in literature search for its first 5-10 experiments.

**Examples:**

```
/turing:init
# Interactive setup — prompts for project name, metric, data source, etc.

/turing:init --plan
# Same setup, then generates a literature-grounded research plan

/turing:init
/turing:init
# Scaffold two projects in the same repo (e.g., ml/sentiment and ml/churn)
```

!!! tip
    Use `--plan` on your first project. The research plan gives the agent a structured starting point instead of random exploration, typically saving 3-5 wasted experiments.

---

### `/turing:train`: Run the Autonomous Experiment Loop

Run the autonomous ML experiment loop. Iteratively hypothesizes, trains, evaluates, and decides, keeping only improvements. Implements the autoresearch pattern with formal convergence detection and git-disciplined rollback.

Each iteration follows the experiment lifecycle: `proposed -> running -> evaluating -> kept/discarded`. The agent proposes a hypothesis, executes it, measures the result against the immutable evaluation harness, and decides whether to keep or discard. Only improvements survive in git history.

**Syntax:** `/turing:train [ml/project] [max_iterations]`

- A project path (e.g., `ml/coding`) targets a specific project directory.
- A number sets the maximum iteration count. Without it, the agent runs until convergence (defined by `config.yaml` patience settings).
- Both can be combined.

**Examples:**

```
/turing:train
# Auto-detect project, run until convergence

/turing:train 10
# Run at most 10 iterations

/turing:train ml/sentiment 20
# Target ml/sentiment, max 20 iterations
```

!!! tip
    For fully hands-off training, combine with the `/loop` command: `/loop 5m /turing:train`. The stop hook automatically detects convergence and halts the loop. Use `3m` for small datasets, `5m` for standard runs, `10m` for large models.

---

### `/turing:sweep`: Systematic Hyperparameter Sweep

Generate and run a systematic hyperparameter sweep. Computes the cartesian product of configured parameter ranges and processes the queue sequentially with full experiment logging.

The sweep generates a queue from the configuration, then processes each combination: apply config overrides, create an experiment branch, run training, log results, and merge if improved. Continues until the queue is empty.

**Syntax:** `/turing:sweep [sweep_config.yaml]`

- Accepts a sweep config path, or defaults to `sweep_config.yaml` in the project directory.
- Use `--status` to check queue progress without running.

**Examples:**

```
/turing:sweep
# Run sweep from default sweep_config.yaml

/turing:sweep custom_sweep.yaml
# Use a custom sweep configuration

/turing:sweep --status
# Check how many configurations remain in the queue
```

!!! tip
    Define your sweep config before running. The sweep generates a full cartesian product, so keep parameter ranges small. A 4x4x3 grid produces 48 runs.

---

### `/turing:status`: Show Experiment Status

Show current ML experiment status: best model, recent experiments, convergence state, and trend analysis. This is an observation-only operation; no code is modified. Delegates to the read-only `@ml-evaluator` agent for safety.

**Syntax:** `/turing:status`

No arguments. Auto-detects the project directory.

**Examples:**

```
/turing:status
# Show best model, total experiments, convergence state, and trend

/turing:status
# If no experiments exist, reports the pipeline is ready and suggests /turing:train
```

!!! tip
    Run `/turing:status` before and after every training session. The convergence state tells you whether more iterations will help or whether you have plateaued.

---

### `/turing:compare`: Side-by-Side Experiment Comparison

Compare two ML experiment runs side-by-side: metrics, configuration deltas, and a verdict on which approach is more promising.

Analyzes metric differences across all configured metrics, identifies what changed between the two configurations (model type, hyperparameters, features), performs causal analysis of which changes likely drove the metric difference, and delivers a verdict.

**Syntax:** `/turing:compare <exp-id-1> <exp-id-2>`

Both experiment IDs are required.

**Examples:**

```
/turing:compare exp-003 exp-007
# Full comparison: metrics, config delta, causal analysis, verdict

/turing:compare exp-001 exp-042
# Compare the first experiment against the 42nd
```

!!! tip
    If you are unsure which experiment IDs to compare, run `/turing:status` first to see the recent experiment list with IDs.
