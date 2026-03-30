# Helios

Autonomous ML research harness for Claude Code. Iteratively trains, evaluates, and improves ML models with structured experiment tracking, convergence detection, and safety guardrails.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [snoglobe/helios](https://github.com/snoglobe/helios).

## Installation

```bash
# Install as Claude Code plugin
claude plugin add /path/to/helios

# Or install from npm (when published)
claude plugin add claude-helios
```

## Quick Start

```bash
# 1. Initialize a new ML project in your repo
/helios:init

# 2. Follow the prompts to configure:
#    - Project name
#    - Target metric (accuracy, f1, mae, etc.)
#    - Task description
#    - Data location

# 3. Add your training data

# 4. Start the autonomous training loop
/helios:train

# 5. Check experiment status
/helios:status

# 6. Compare runs
/helios:compare exp-001 exp-002
```

## Commands

| Command | Purpose |
|---------|---------|
| `/helios` | Router -- detects ML intent and routes to sub-commands |
| `/helios:init` | Scaffold a new ML project with autoresearch harness |
| `/helios:train [N]` | Run autonomous experiment loop (optional max iterations) |
| `/helios:status` | Show experiment status, best model, convergence state |
| `/helios:compare <a> <b>` | Side-by-side experiment comparison |
| `/helios:sweep` | Generate and run hyperparameter sweep |

## Agents

| Agent | Purpose | Tools |
|-------|---------|-------|
| `@ml-researcher` | Autonomous training agent. Modifies `train.py`, runs experiments, keeps improvements. | Read, Write, Edit, Bash, Grep, Glob |
| `@ml-evaluator` | Read-only analysis agent. Analyzes results without risk of modifying code. | Read, Bash, Grep, Glob |

## How It Works

Helios implements the **autoresearch pattern**: an AI agent iteratively improves an ML model through a structured experiment loop.

### The Experiment Loop

1. **Read** recent experiment results and agent memory
2. **Propose** a hypothesis (new model, hyperparameters, features)
3. **Modify** `train.py` (the only editable file)
4. **Train** and evaluate using the fixed harness (`prepare.py`, `evaluate.py`)
5. **Keep** improvements, **discard** regressions (via git)
6. **Log** every experiment to `experiments/log.jsonl`
7. **Repeat** until convergence or max iterations

### Safety Guardrails

- **Immutable evaluation**: `prepare.py` and `evaluate.py` are READ-ONLY. The agent cannot game metrics by modifying the evaluation harness.
- **Git discipline**: Every experiment is committed before running. Failed experiments are reverted. Only improvements survive in git history.
- **Convergence detection**: Automatic stop after N consecutive non-improvements (configurable).
- **Read-only evaluator**: The `@ml-evaluator` agent has no Write/Edit tools -- safe for analysis without risk of breaking the pipeline.
- **Structured logging**: Every experiment (kept or discarded) is logged to JSONL with full metrics, config, and description.

### `/loop` Integration

For fully hands-off training:

```
/loop 5m /helios:train
```

The Stop hook automatically detects convergence and halts the loop.

## Project Structure (after `/helios:init`)

```
your-project/
  {{ML_DIR}}/
    prepare.py              # READ-ONLY: Data loading, splitting
    evaluate.py             # READ-ONLY: Evaluation harness
    train.py                # AGENT-EDITABLE: Training code
    config.yaml             # Hyperparameters and settings
    sweep_config.yaml       # Sweep parameter ranges
    program.md              # Agent instructions
    features/
      featurizers.py        # Feature engineering
    scripts/
      log_experiment.py     # Experiment logging
      show_metrics.py       # Metrics display
      compare_runs.py       # Run comparison
      sweep.py              # Hyperparameter sweep
      post-train-hook.sh    # PostToolUse hook
      stop-hook.sh          # Convergence detection hook
    experiments/
      log.jsonl             # Experiment log (append-only)
      results.tsv           # Quick-reference summary
    models/
      best/                 # Current best model
      archive/              # Previous best models
    data/
      splits/               # Train/val/test splits
    tests/
      conftest.py           # Test fixtures
  .claude/
    agent-memory/
      ml-researcher/
        MEMORY.md           # Agent memory (persists across sessions)
    settings.local.json     # Hooks for auto-logging and convergence
```

## Configuration

### config.yaml

Controls all experiment parameters:

- `data.source` -- path to training data
- `data.target_column` -- column name for the prediction target
- `evaluation.primary_metric` -- metric to optimize
- `evaluation.lower_is_better` -- direction of optimization
- `convergence.patience` -- non-improvements before stopping
- `convergence.improvement_threshold` -- minimum relative improvement
- `model.type` -- model framework
- `model.hyperparams` -- model hyperparameters

### sweep_config.yaml

Defines hyperparameter ranges for grid search:

```yaml
sweep:
  model.hyperparams.n_estimators: [50, 100, 200]
  model.hyperparams.max_depth: [3, 4, 6, 8]
  model.hyperparams.learning_rate: [0.01, 0.05, 0.1]
```

## Placeholders

Templates use `{{PLACEHOLDER}}` markers that `/helios:init` replaces:

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `{{PROJECT_NAME}}` | ML project name | `sentiment` |
| `{{TARGET_METRIC}}` | Primary metric to optimize | `accuracy` |
| `{{TASK_DESCRIPTION}}` | What the model does | `Predict sentiment from reviews` |
| `{{ML_DIR}}` | Directory for ML files | `ml/sentiment` |
| `{{DATA_SOURCE}}` | Training data location | `data/reviews.csv` |
| `{{METRIC_DIRECTION}}` | Optimization direction | `higher` |

## CLI Usage (outside Claude Code)

```bash
# Scaffold a project from the command line
helios-init my-project ml/my-project
```

## License

MIT
