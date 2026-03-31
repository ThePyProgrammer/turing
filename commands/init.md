---
name: init
description: Initialize a new ML project with the Turing autoresearch harness. Scaffolds the full experiment infrastructure — immutable evaluation pipeline, agent-editable training code, structured logging, convergence detection hooks, and a Python virtual environment. Use --plan to generate a research plan.
disable-model-invocation: true
argument-hint: "[project_name] [--plan]"
allowed-tools: Read, Write, Edit, Bash(*), Grep, Glob, WebSearch, WebFetch
---

Scaffold a new ML project with the Turing autoresearch harness. This creates the separation between the measurement apparatus (READ-ONLY) and the hypothesis space (AGENT-EDITABLE) that makes autonomous experimentation trustworthy.

## Interactive Setup

Ask the user for the following (or accept from `$ARGUMENTS` if provided as JSON):

1. **Project name** (`{{PROJECT_NAME}}`): Name of the ML project (e.g., "sentiment", "churn", "fraud-detection")
2. **Target metric** (`{{TARGET_METRIC}}`): Primary metric to optimize (e.g., "accuracy", "f1", "mae", "mse", "auc")
3. **Metric direction**: Is lower better (mae, mse, loss) or higher better (accuracy, f1, auc)?
4. **Task description** (`{{TASK_DESCRIPTION}}`): What the model does (e.g., "Predict customer churn from usage data")
5. **ML directory** (`{{ML_DIR}}`): Where ML files go relative to project root (e.g., "ml/sentiment")
6. **Data source** (`{{DATA_SOURCE}}`): Where training data comes from (e.g., "data/reviews.csv")

## Scaffolding

Once you have all 6 values, delegate to the unified scaffolding script:

```bash
python3 <templates_dir>/scripts/scaffold.py \
    --project-name "<project_name>" \
    --target-metric "<target_metric>" \
    --metric-direction "<metric_direction>" \
    --task-description "<task_description>" \
    --ml-dir "<ml_dir>" \
    --data-source "<data_source>" \
    --templates-dir "<templates_dir>"
```

The scaffold script handles everything in a single atomic operation:
- Copies all template files with placeholder substitution
- Creates data/, experiments/, models/ directories
- Sets up agent memory at `.claude/agent-memory/ml-researcher/MEMORY.md`
- Configures Claude Code hooks in `.claude/settings.local.json`
- Creates Python virtual environment and installs requirements
- Verifies all placeholders were replaced (fails loudly if any remain)

## Locating Templates

Find the templates directory using Glob:
```
~/.claude/plugins/*/templates/
```
Or check if installed via npm by looking for `node_modules/claude-turing/templates/`.

## After Scaffolding

Report what was created:
- The separation: READ-ONLY (`prepare.py`, `evaluate.py`) vs AGENT-EDITABLE (`train.py`)
- Next steps: add data to the configured data source path, run `python prepare.py`, then `/turing:train`
- The taste-leverage loop: `/turing:try` to inject hypotheses, `/turing:brief` for intelligence reports

## Research Plan Generation (--plan flag)

If `$ARGUMENTS` contains `--plan`, generate a research plan AFTER scaffolding. This gives the agent strategic direction for its first 5-10 experiments rather than ad-hoc exploration.

### Steps:

1. **Read the task context** from the just-created `config.yaml`: task description, model type, target metric, data source.

2. **Search literature** with `WebSearch` for the task domain:
   - "state of the art <task description> machine learning 2024 2025"
   - "best model <target metric> <data type> benchmark"
   - "<task description> common approaches survey"

   Use `WebFetch` on top 2-3 results to extract: dominant model families, typical metric ranges, known challenges.

3. **Generate `RESEARCH_PLAN.md`** in the ML project directory with this structure:

   ```markdown
   # Research Plan: <task description>

   Generated: <date>

   ## Task Summary
   <one paragraph describing the task, data, and success criteria>

   ## Model Families to Explore
   Ordered by expected relevance based on literature:
   1. **<family 1>** — <why, with citation>
   2. **<family 2>** — <why, with citation>
   3. **<family 3>** — <why, with citation>

   ## Evaluation Strategy
   - Primary metric: <metric> (<higher/lower> is better)
   - Multi-run recommendation: <yes/no, based on expected variance>
   - Baseline target: <realistic first-pass metric from literature>

   ## Search Budget
   - <N> experiments per model family before moving on
   - Total budget: <N> experiments before first convergence check

   ## Success Criteria
   - Target metric: <value from literature benchmarks>
   - Convergence: <patience> consecutive non-improvements

   ## Known Challenges
   - <challenge 1 from literature, e.g., "class imbalance common in this domain">
   - <challenge 2>

   ## Sources
   - <citation 1>
   - <citation 2>
   ```

4. **Self-critique the plan** (one round):
   - Are the model families ordered by evidence strength?
   - Is the budget realistic?
   - Are the success criteria grounded in benchmark data?
   Revise if any section is vague or unsupported.

5. **Report:** "Research plan generated at `<ml_dir>/RESEARCH_PLAN.md`. The agent will read this during `/turing:train` for strategic direction."

### Integration

The agent's `program.md` OBSERVE step reads `RESEARCH_PLAN.md` (if it exists) for strategic direction. The plan is advisory — the agent can deviate but should note why in `experiment_state.yaml`.
