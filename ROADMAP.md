# ROADMAP.md

*"Turing flips the coins. You choose which ones."*

This roadmap describes the features needed to make the coin-flipping vision real — to close the gap between what the README promises and what the system delivers.

## Current State (v2.0.0)

The scaffolding works. The experiment loop runs. Evaluation is immutable. Convergence is detected. 68 tests pass. 16 ADRs document the architecture. But the actual research intelligence is thin: grid search, free-text memory, single-run comparisons, and no structured mechanism for the human to steer the agent or the agent to report back to the human.

## Vision

A system where:
- The **human** selects the search region (taste, judgment, problem selection)
- The **agent** exhaustively explores it with perfect memory, statistical rigor, and intelligent prioritization
- The **interface** between them is structured, bidirectional, and actionable

---

## Phase 1: Human-Agent Interface (Tier 4)

*Make the taste-leverage loop functional, not aspirational.*

### 1.1 Hypothesis Injection — `/turing:try`

**What:** A command that lets the human inject a specific hypothesis into the agent's work queue with a structured description.

**Why:** Currently the human can only say `/turing:train` and hope the agent picks the right direction. The taste-leverage loop requires the human to steer: "try LightGBM with leaf-wise growth" or "increase regularization, I think we're overfitting." Without this, the human's taste has no mechanism to reach the agent.

**Implementation:**
1. Create `commands/try.md` — a new skill that accepts a hypothesis description from `$ARGUMENTS`
2. Create `templates/hypotheses.yaml` — a structured queue file with fields:
   - `id`: sequential hypothesis ID (hyp-001, hyp-002, ...)
   - `description`: what to try and why (from the human)
   - `source`: "human" or "agent"
   - `status`: "queued" | "in-progress" | "tested" | "promising" | "dead-end"
   - `priority`: "high" | "medium" | "low" (human-injected defaults to high)
   - `parent_experiment`: optional link to the experiment that motivated this
   - `result_experiment`: filled after testing, links to the experiment log entry
   - `created_at`: ISO timestamp
3. Update `templates/program.md` — the experiment loop's OBSERVE step checks `hypotheses.yaml` for queued items before generating its own hypotheses. Human-injected hypotheses take priority.
4. Update `agents/ml-researcher.md` — document that the agent should check the hypothesis queue at each iteration start
5. Update `commands/turing.md` router — add "try", "inject", "test this" to routing table
6. Add `templates/scripts/manage_hypotheses.py` — CLI tool:
   - `python scripts/manage_hypotheses.py add "description" [--priority high]` — add hypothesis
   - `python scripts/manage_hypotheses.py list` — show queue with status
   - `python scripts/manage_hypotheses.py next` — get next queued hypothesis
   - `python scripts/manage_hypotheses.py mark <id> <status>` — update status
7. Add tests for the hypothesis queue management
8. Update `docs/adr/README.md` index and `config/state.toml`

**Acceptance:** The human can type `/turing:try "switch to LightGBM with dart boosting"` and the next `/turing:train` iteration will prioritize that hypothesis.

### 1.2 Research Briefing — `/turing:brief`

**What:** A command that generates a structured intelligence report from experiment history — what's been learned, what's promising, what's exhausted, and what the human should consider next.

**Why:** The agent accumulates knowledge across experiments but currently has no structured way to report it back. The human must read MEMORY.md (free-text, agent-written) or scan the experiment log manually. A briefing command closes the taste-leverage loop: the agent reports intelligence, the human applies taste, the human injects hypotheses, the agent executes them.

**Implementation:**
1. Create `commands/brief.md` — a skill that delegates to `@ml-evaluator` for read-only analysis
2. The briefing must produce a structured report with these sections:
   - **Campaign Summary**: total experiments, kept/discarded ratio, time span
   - **Current Best**: model type, metrics, experiment ID, configuration
   - **Improvement Trajectory**: metric over time (text sparkline or table), rate of improvement
   - **Exhausted Directions**: approaches that were tried and showed no promise (from MEMORY.md + log analysis)
   - **Promising Directions**: approaches that showed improvement but weren't fully explored
   - **Hypothesis Queue Status**: pending hypotheses from `hypotheses.yaml`
   - **Recommendations**: 3-5 concrete next hypotheses ranked by expected impact, with rationale
   - **Convergence Assessment**: how close to convergence, remaining budget estimate
3. Add `templates/scripts/generate_brief.py` — generates the structured report from `log.jsonl`, `MEMORY.md`, and `hypotheses.yaml`
4. The brief should be saveable: `python scripts/generate_brief.py > briefs/brief-YYYY-MM-DD.md`
5. Add tests for brief generation
6. Update router

**Acceptance:** `/turing:brief` produces a 1-page report that a researcher can read in 2 minutes and immediately decide what to inject next.

### 1.3 Experiment Dependency Graph

**What:** Track which experiments inspired which. Each experiment records an optional `parent_experiment` and `hypothesis_id`, creating a tree of reasoning.

**Implementation:**
1. Extend the JSONL log schema: add `parent_experiment` (optional experiment ID) and `hypothesis_id` (optional hypothesis ID) fields to `log_experiment.py`
2. Update `templates/scripts/log_experiment.py` — accept optional `--parent` and `--hypothesis` CLI args
3. Update `templates/program.md` — the RECORD step includes parent linkage
4. Add `templates/scripts/show_experiment_tree.py` — visualize the experiment dependency tree as text:
   ```
   exp-001 (xgboost baseline, accuracy=0.82)
   ├── exp-002 (increase depth, accuracy=0.84) ✓ kept
   │   ├── exp-004 (increase estimators, accuracy=0.85) ✓ kept
   │   └── exp-005 (add regularization, accuracy=0.83) ✗ discarded
   └── exp-003 (lightgbm switch, accuracy=0.81) ✗ discarded
   ```
5. Update `/turing:brief` to include the tree in its output
6. Add tests for tree generation and parent linkage

**Acceptance:** `python scripts/show_experiment_tree.py` displays the full experiment lineage, making the agent's reasoning chain visible to the human.

---

## Phase 2: Intelligent Search (Tier 1)

*Make the coin-flipping smarter, not just faster.*

### 2.1 Multi-Run Statistical Significance

**What:** Run each configuration N times with different random seeds and compare distributions, not point estimates.

**Implementation:**
1. Add `evaluation.n_runs` config key (default: 1, settable to 3-5 for statistical mode)
2. Update `templates/train.py` — accept `--seed` override argument
3. Add `templates/scripts/statistical_compare.py` — runs N training passes, collects metrics, computes:
   - Mean and standard deviation per metric
   - Confidence interval (95%)
   - Mann-Whitney U test p-value between current and prior best
   - Verdict: "significantly better" / "not significant" / "significantly worse"
4. Update convergence detection to use mean performance, not single-run
5. Update `templates/program.md` to use statistical comparison when `n_runs > 1`
6. Add tests

**Acceptance:** With `n_runs: 3`, the agent runs each experiment 3 times and only keeps configurations that are *statistically* better, not just numerically different.

### 2.2 Bayesian-Guided Hypothesis Generation

**What:** Use experiment history to build a surrogate model that predicts which configurations are most likely to improve, guiding exploration toward promising regions.

**Implementation:**
1. Add `templates/scripts/suggest_next.py` — reads `log.jsonl`, fits a simple surrogate (Random Forest over hyperparameter space → predicted metric), suggests the configuration with highest expected improvement
2. This is advisory, not mandatory — the agent can use it as one input alongside its own reasoning
3. Update `templates/program.md` OBSERVE step: "optionally run `python scripts/suggest_next.py` for data-driven suggestions"
4. No additional dependencies (sklearn's RandomForestRegressor is already available)
5. Add tests

**Acceptance:** `python scripts/suggest_next.py` outputs 3 suggested configurations ranked by expected improvement, with uncertainty estimates.

---

## Phase 3: Smarter Measurement (Tier 3)

*Make the immutable evaluation apparatus more informative.*

### 3.1 Automatic Metric Decomposition

**What:** Extend `evaluate.py` to produce per-class metrics, confusion matrix, and train/val gap alongside aggregate metrics.

**Implementation:**
1. Add `evaluate_detailed()` function to `evaluate.py` — produces per-class precision/recall/F1, confusion matrix as dict
2. Add `format_detailed_metrics()` — outputs detailed metrics to a separate section in `run.log` (after the `---` delimited aggregate block)
3. Update `train.py` to call `evaluate_detailed()` and write results to `experiments/details/exp-NNN.json`
4. The `@ml-evaluator` agent can read detailed metrics for deeper analysis
5. Add tests for per-class metrics

**Acceptance:** After each experiment, per-class performance and confusion matrix are available for analysis without modifying the aggregate metric contract.

### 3.2 Train/Val Gap Monitoring

**What:** Automatically log both training and validation metrics, and flag overfitting when the gap exceeds a threshold.

**Implementation:**
1. Update `train.py` template to evaluate on both train and val sets
2. Add `train_metrics` alongside existing `metrics` in the log entry
3. Add `overfitting_gap` computed metric: `train_metric - val_metric`
4. Update convergence detection: if `overfitting_gap > threshold`, flag as overfitting (not just "no improvement")
5. Update `taxonomy.toml` — the `overfitting` failure mode gets a quantitative definition
6. Add tests

**Acceptance:** The experiment log includes both train and val metrics, and the agent can distinguish "didn't improve" from "overfit" in its reasoning.

---

## Phase 4: Structured Memory (Tier 2)

*Make the agent's memory machine-readable, not just human-readable.*

### 4.1 Structured Experiment Embeddings

**What:** Replace free-text MEMORY.md with a structured `experiment_state.yaml` that the agent reads and writes in a schema-validated format.

**Implementation:**
1. Define schema: `best_result` (structured), `failed_approaches` (list of {description, experiment_id, reason}), `promising_directions` (list of {description, priority, evidence}), `session_history` (list of {date, experiments_run, best_metric})
2. Create `templates/scripts/update_state.py` — reads/writes `experiment_state.yaml` with validation
3. Update `agents/ml-researcher.md` — read `experiment_state.yaml` instead of (or alongside) MEMORY.md
4. Keep MEMORY.md as human-readable companion, generated from the structured state
5. Add tests for state serialization and validation

**Acceptance:** The agent's memory is a validated YAML file that can be programmatically queried, not just prose that must be parsed by an LLM.

---

## Implementation Order

| # | Feature | Phase | Priority | Depends On |
|---|---------|-------|----------|------------|
| 1 | Hypothesis injection `/turing:try` | 1.1 | **Critical** | — |
| 2 | Research briefing `/turing:brief` | 1.2 | **Critical** | 1.1 (reads hypothesis queue) |
| 3 | Experiment dependency graph | 1.3 | **High** | — |
| 4 | Multi-run statistical significance | 2.1 | **High** | — |
| 5 | Bayesian-guided suggestions | 2.2 | **Medium** | — |
| 6 | Automatic metric decomposition | 3.1 | **Medium** | — |
| 7 | Train/val gap monitoring | 3.2 | **Medium** | 3.1 |
| 8 | Structured experiment state | 4.1 | **Medium** | 1.1 (hypothesis queue is first structured state) |

Phases 1.1 and 1.2 are the v2.1.0 release. They close the taste-leverage loop.
