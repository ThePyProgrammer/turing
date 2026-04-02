---
title: "Taste-Leverage Interface"
description: "Six commands for injecting human judgment into the autonomous loop: try, brief, suggest, explore, design, and mode."
---

# Taste-Leverage Interface

These commands bridge human research taste and machine execution leverage. You decide *what* is worth trying; the agent decides *how* to try it rigorously. This is the core interaction pattern: the human selects which coins to flip, the agent flips them.

---

### `/turing:try`: Inject a Hypothesis

Inject a hypothesis into the agent's experiment queue. This is the primary taste-leverage mechanism: you provide judgment about what is worth trying, the agent provides disciplined execution.

Human-injected hypotheses are marked as **high priority** and tested before the agent generates its own. After testing, each hypothesis is marked as `tested`, `promising`, or `dead-end` with a link to the resulting experiment.

**Syntax:** `/turing:try <hypothesis description>` or `/turing:try archetype:<name>`

- Free-text hypotheses are queued directly.
- Archetype syntax (e.g., `archetype:model_comparison`) expands into structured multi-experiment strategies.

Available archetypes: `model_comparison` (~5 experiments), `hyperparameter_sweep` (15-36), `feature_sweep` (6-10), `regularization_search` (4-6), `ensemble_construction` (4-6), `learning_rate_schedule` (4-5), `data_quality_audit` (3-5), `ablation_study` (N+1).

**Examples:**

```
/turing:try switch to LightGBM with dart boosting and lower learning rate
# Free-text hypothesis, queued as high priority

/turing:try add polynomial features for the numeric columns
# Another free-text hypothesis

/turing:try archetype:model_comparison
# Structured strategy: compare XGBoost, LightGBM, RF, LR, MLP with statistical tests

/turing:try archetype:ensemble_construction
# Build voting, stacking, and blending ensembles of top models
```

!!! tip
    Use archetypes for systematic exploration (e.g., `model_comparison` early on) and free-text for targeted refinements (e.g., "increase regularization, the train/val gap suggests overfitting").

---

### `/turing:brief`: Research Intelligence Report

Generate a structured research intelligence report from experiment history: what has been learned, what is promising, what is exhausted, and what the human should consider next. Designed to be read in 2 minutes.

The briefing has 6 sections: Campaign Summary, Current Best, Improvement Trajectory, Model Types Explored, Hypothesis Queue, and Recommendations. The `--deep` flag adds a 7th section with Literature-Grounded Suggestions that searches recent papers and auto-queues hypotheses.

**Syntax:** `/turing:brief [ml/project] [--deep]`

- Accepts an optional project path. Auto-detects from `config.yaml` otherwise.
- `--deep` adds literature-grounded suggestions by searching recent papers and distilling 3-5 actionable techniques.

**Examples:**

```
/turing:brief
# 2-minute intelligence briefing on the current project

/turing:brief ml/sentiment
# Brief for a specific project

/turing:brief --deep
# Full briefing plus literature-grounded suggestions with citations, auto-queued as hypotheses
```

!!! tip
    Run `/turing:brief` every time you return to a project after time away. It tells you exactly where things stand and what to try next. Use `--deep` when the agent seems stuck and you want evidence-based direction.

---

### `/turing:suggest`: Literature-Grounded Model Selection

Read the ML task context, search recent literature, and suggest model architectures worth trying, with citations. Suggestions are auto-queued as hypotheses for the next `/turing:train` iteration.

Supports two strategies: **literature** (default) searches the web for recent papers and synthesizes grounded suggestions, and **treequest** uses AB-MCTS tree search over the critique scoring function to explore refinement chains that linear search cannot find.

**Syntax:** `/turing:suggest [task description override] [--strategy literature|treequest]`

- Default strategy is `literature` (requires internet).
- `--strategy treequest` uses tree-search-guided exploration (works offline).
- TreeQuest options: `--iterations N`, `--top N`, `--greedy`.

**Examples:**

```
/turing:suggest
# Literature search for model suggestions based on config.yaml task

/turing:suggest "tabular classification with heavy class imbalance"
# Override the task description for more targeted literature search

/turing:suggest --strategy treequest --iterations 50 --top 10
# Tree-search exploration with 50 iterations, top 10 results
```

!!! tip
    Use the literature strategy early in a project to find strong baselines. Switch to treequest mid-experiment when you want to explore non-obvious combinations of techniques that papers do not explicitly describe.

---

### `/turing:explore`: Tree-Search Hypothesis Exploration

Explore the hypothesis space using AB-MCTS tree search. Instead of suggesting independent ideas, this builds and searches a tree of refinement chains; each node is a hypothesis scored by novelty, feasibility, and expected impact. Discovers non-obvious combinations that linear suggestion cannot find.

**Syntax:** `/turing:explore [ml/project] [--iterations N] [--top N] [--strategy abmcts-a|abmcts-m|greedy]`

- `--iterations N` sets search depth (default: 30).
- `--top N` sets number of results (default: 5).
- `--strategy` selects the algorithm: `abmcts-a` (default), `abmcts-m` (Bayesian, requires PyMC), or `greedy`.
- `--seeds-only` shows generated seeds without running search.
- `--json` outputs JSON for programmatic use.

**Examples:**

```
/turing:explore
# Default: 30 iterations, top 5, AB-MCTS-A strategy

/turing:explore --iterations 50 --top 10
# Deeper search, more results

/turing:explore --strategy greedy
# Fast greedy search without TreeQuest dependency

/turing:explore ml/churn --json
# JSON output for a specific project
```

!!! tip
    `/turing:explore` differs from `/turing:suggest` in a key way: suggest finds what papers recommend (independent suggestions), while explore finds what combinations score well (refinement chains). Use suggest for early exploration, explore for mid-experiment optimization.

---

### `/turing:design`: Experiment Design Document

Generate a structured experiment design for a hypothesis before running it. Reads experiment history, searches literature for methodology, and produces a scored design document. Front-loads the thinking before the coding.

The design includes a falsifiable objective, literature-grounded method, concrete implementation plan for `train.py`, expected outcomes with metric thresholds, risks from the literature, and a self-critique score.

**Syntax:** `/turing:design <hypothesis-id or description>`

- Accepts a hypothesis ID (e.g., `hyp-003`) or free-text description.
- Design documents are saved to `experiments/designs/`.

**Examples:**

```
/turing:design hyp-003
# Design document for a queued hypothesis

/turing:design "switch to CatBoost with ordered boosting for categorical features"
# Design from a free-text description

/turing:design "add target encoding for high-cardinality categorical features"
# Literature-grounded design with implementation plan
```

!!! tip
    Use `/turing:design` for high-risk or complex hypotheses. The literature search often surfaces pitfalls and hyperparameter recommendations that save wasted iterations.

---

### `/turing:mode`: Set Research Strategy

Set the research strategy mode: explore (try new things), exploit (refine what works), or replicate (verify results). The mode drives the novelty guard policy and agent behavior during `/turing:train`.

| Mode | Novelty Guard | Agent Behavior |
|------|---------------|----------------|
| **explore** | Allow novel ideas, block repeats | Try fundamentally different approaches |
| **exploit** | Allow follow-ups, block repeats | Refine the current best configuration |
| **replicate** | Allow duplicates, block novel ideas | Re-run best experiments with different seeds |

**Syntax:** `/turing:mode <explore|exploit|replicate>`

**Examples:**

```
/turing:mode explore
# Switch to exploration — best when the current approach feels exhausted

/turing:mode exploit
# Switch to exploitation — best when you have a promising direction to refine

/turing:mode replicate
# Switch to replication — best before declaring a winner
```

!!! tip
    The default mode is `exploit`. Switch to `explore` when you are plateauing, and to `replicate` before making final decisions about which model to ship.
