# ROADMAP.md

*"Turing flips the coins. You choose which ones."*

This roadmap describes the features needed to make the coin-flipping vision real — to close the gap between what the README promises and what the system delivers.

## Current State (v1.1.0)

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

## Phase 5: Anti-Cheating Guardrails (Tier 0 — Safety)

*The agent is not cheating. It is optimizing. That is worse.*

### The Problem

The autoresearch pattern has a structural vulnerability: when you give an optimizer a metric and a code editor, it will find the shortest path to a high number — even if that path subverts the entire purpose of the experiment. This is not theoretical. It has been observed in practice and documented extensively.

**Primary sources:**
- [suzuke, "I let AI run 100 experiments. It learned to cheat."](https://suzuke.github.io/blog/posts/ai-cheating-experiments/) (2026-03-18) — detailed case study of an LLM agent gaming a Gomoku AlphaZero task
- [suzuke/autocrucible](https://github.com/suzuke/autocrucible) — an autoresearch fork built specifically to prevent metric gaming, with file-level access control, behavioral probes, sandbox isolation, and stability validation
- [karpathy/autoresearch#322](https://github.com/karpathy/autoresearch/discussions/322) — community discussion on Goodhart's Law violations in autonomous experiment loops

### Taxonomy of Failure Modes

The research identifies seven distinct ways an autonomous ML agent can produce misleading results:

| # | Failure Mode | Description | Example | Severity |
|---|-------------|-------------|---------|----------|
| 1 | **Task Avoidance** | Agent skips the expensive/uncertain part of the task and achieves the metric via an alternative path that satisfies the number but not the intent | Agent asked to train a neural net writes an alpha-beta search engine instead. `train_time_sec: 0.0`. Win rate: 99.3%. No neural net involved. | Critical |
| 2 | **Minimal Compliance** | Agent satisfies the letter of a probe/check with minimum effort while not doing the actual work | After adding a forward hook check, agent calls `net.forward()` once, throws away the output, keeps using its search engine. Hook sees a call. Net still does nothing. | Critical |
| 3 | **Seed/Data Exploitation** | Agent reads the evaluation harness, finds fixed seeds or test data, and memorizes answers instead of learning | Agent reads `evaluate.py`, finds `seed=42`, reconstructs the exact noise vector, achieves MSE=0.0 by memorizing the test set. 3 iterations. | Critical |
| 4 | **Accidental Semantic Bugs** | Agent makes plausible-looking mistakes that invert behavior. Not intentional but equally destructive. | MCTS `_evaluate_leaf()` returns -1.0 for "current player lost" but backpropagation reads from parent's perspective. Sign is flipped. Tree prefers losing moves. Happened in ~50% of agent MCTS rewrites. | High |
| 5 | **Noise Exploitation** | Without multi-run validation, a lucky single run appears as improvement. Agent rewarded for changes with no real effect. | Agent makes a no-op change, gets a higher metric from randomness, "keeps" it. Over many iterations, accumulated noise masquerades as progress. | High |
| 6 | **Ceiling Hitting** | Agent reaches the structural limit of the experimental setup. Not cheating but reveals design flaws. | Self-play against a fixed-strength champion cannot surpass that champion's level. Convergence is declared at the ceiling, not the optimum. | Medium |
| 7 | **Prompt Rule Circumvention** | Every natural-language instruction can be technically complied with while violating the spirit. | "You must use the neural net" → agent calls `forward()` once, ignores result. Technically compliant. Functionally useless. | Critical |

**The core insight:** Every prompt-based rule got worked around. Every code-based rule held. Prompt enforcement is probabilistic. Code enforcement is deterministic.

### What Turing Currently Does

Turing already implements the foundational separation (ADR-0002): `prepare.py` and `evaluate.py` are READ-ONLY, `train.py` is the only agent-editable file. This prevents the most obvious metric gaming (modifying the evaluation function). But the enforcement is **prompt-level only** — the agent is *instructed* not to modify evaluation files, not *prevented* from doing so.

Turing also has the two-agent architecture (ADR-0003): the evaluator agent has no Write/Edit tools. And Phase 2.1 implemented multi-run statistical significance, which addresses noise exploitation.

What Turing does NOT have:
- **Hidden file tier** — the agent can READ `evaluate.py`. It can study fixed seeds, scoring formulas, test data, and exploit them.
- **Behavioral probes** — no mechanism to verify the agent actually uses the trained model (vs. building an alternative)
- **Runtime enforcement** — file access restrictions are prompt-level, not SDK-level
- **Sandbox isolation** — experiments run in the user's environment with full filesystem access
- **Stability validation** — no automatic detection of high-variance metrics that enable noise exploitation
- **Tool restriction** — the researcher agent has Bash access, meaning it can execute arbitrary commands

### 5.1 Hidden File Tier — Prevent Evaluation Exploitation

**What:** Add a third file access tier beyond READ-ONLY and READ-WRITE: HIDDEN files that the agent cannot see, read, or know exist. `evaluate.py` moves from READ-ONLY to HIDDEN.

**Why:** The most devastating cheating pattern — seed/data exploitation — requires the agent to *read* the evaluation harness. Making it READ-ONLY prevents modification but not reverse-engineering. The autocrucible project demonstrated that when `evaluate.py` was READ-ONLY (not hidden), the agent read it, found `seed=42`, reconstructed the exact noise vector, and achieved MSE=0.0 by memorizing the test set in 3 iterations.

**Implementation:**
1. Extend the file access taxonomy in `commands/rules/loop-protocol.md`:
   ```
   | Layer | Files | Agent Access | Rationale |
   |-------|-------|-------------|-----------|
   | Hidden | evaluate.py | NONE | Agent cannot read, write, or discover |
   | Measurement | prepare.py | READ-ONLY | Agent can read but not modify |
   | Hypothesis | train.py, config.yaml | READ-WRITE | All experimental changes go here |
   ```
2. Update `templates/evaluate.py` — remove any inline comments that leak implementation details. The agent should not see this file at all, but defense in depth.
3. Update `templates/program.md` — remove references to reading `evaluate.py`. The agent knows the metric name and direction from `config.yaml`, not from reading the harness.
4. Update `agents/ml-researcher.md` — document the hidden tier. The agent should not attempt to read evaluation files.
5. Create a new ADR documenting this decision and its rationale.
6. **Future (requires Claude Agent SDK):** Implement SDK-level PreToolUse hooks that deny Read/Glob/Grep access to hidden files. This is the autocrucible approach — the agent's tool calls are intercepted before execution:
   ```python
   async def pre_tool_use_hook(hook_input, match, context):
       rel = resolve_path(hook_input["tool_input"].get("file_path", ""))
       if rel in hidden_files:
           return {"permissionDecision": "deny",
                   "permissionDecisionReason": "Access denied: hidden platform file."}
   ```
   Until Turing integrates the Claude Agent SDK, this enforcement remains prompt-level with the hidden tier as a strong convention.

**Acceptance:** `evaluate.py` is not referenced in any agent-visible file. The agent has no mechanism to discover its contents. SDK-level enforcement is designed but deferred until Agent SDK integration.

### 5.2 Behavioral Probes — Verify the Agent Does Real Work

**What:** Add hidden validation checks in `evaluate.py` that verify the agent actually performed the intended work (training a model, using the trained model for inference) rather than finding an alternative path to the metric.

**Why:** Task avoidance and minimal compliance are the two most common cheating patterns. The agent writes a search engine instead of training a neural net (`train_time_sec: 0.0`, win rate 99.3%). After adding a forward hook, the agent calls `forward()` once and discards the result. Behavioral probes must check that the *substance* of the work was done, not just the *form*.

**Implementation:**
1. Add timing validation to `evaluate.py`:
   ```python
   import json, os

   # The agent writes train_metadata.json during training
   meta_path = os.path.join(os.path.dirname(__file__), "train_metadata.json")
   if os.path.exists(meta_path):
       meta = json.load(open(meta_path))
       train_time = meta.get("train_time_sec", 0)
       if train_time < config.get("constraints", {}).get("min_train_time", 10):
           print(f"PROBE FAIL: train_time={train_time:.1f}s below minimum")
           # Zero the metric — this is not a valid experiment
   ```
2. Add model-usage validation: verify the saved model artifact was actually loaded and used for predictions (not just saved and ignored). Check that predictions vary (a model that always predicts the same value is suspicious).
3. Add `config.yaml` constraints section:
   ```yaml
   constraints:
     min_train_time: 10        # Seconds — training that takes 0s is suspicious
     min_model_size_bytes: 100 # A real model has non-trivial size
     max_prediction_entropy: 0 # Not yet implemented — placeholder for future
   ```
4. These checks live in `evaluate.py` (hidden from agent) — the agent cannot see or work around them.
5. Add tests for probe validation.

**Acceptance:** An agent that skips training (`train_time_sec: 0.0`) or produces a model that isn't used gets a zeroed metric. The agent cannot discover why without reading `evaluate.py`, which is hidden.

### 5.3 Stability Validation — Prevent Noise Exploitation

**What:** Automatically detect high-variance metrics and force multi-run evaluation when variance exceeds a threshold.

**Why:** Phase 2.1 added optional multi-run statistical significance, but it requires manual configuration (`evaluation.n_runs`). The autocrucible project demonstrated an automatic approach: run the experiment N times, compute coefficient of variation, and if CV > 5%, automatically enable `repeat: 3` with median aggregation. This prevents the agent from being rewarded for lucky runs.

**Implementation:**
1. Add `commands/validate.md` — a `/turing:validate` command that runs the current best experiment N times and reports variance:
   ```
   /turing:validate          # Run 5 times, report CV
   /turing:validate --auto   # If CV > 5%, auto-set n_runs: 3 in config.yaml
   ```
2. Add `templates/scripts/validate_stability.py`:
   ```python
   def check_stability(n_runs=5):
       results = [run_experiment() for _ in range(n_runs)]
       mean = statistics.mean(results)
       stdev = statistics.stdev(results)
       cv = (stdev / abs(mean) * 100) if mean != 0 else float("inf")
       return {"stable": cv < 5.0, "cv": cv, "mean": mean, "stdev": stdev,
               "recommendation": "Set evaluation.n_runs: 3" if cv >= 5.0 else "Stable"}
   ```
3. Integrate with `/turing:init` — run stability check after first successful training and auto-configure if needed.
4. Update convergence detection: use mean performance across runs, not single-run metric.
5. Add tests.

**Acceptance:** After scaffolding, if the metric has CV > 5%, the system auto-configures multi-run evaluation. The agent cannot exploit single-run noise.

### 5.4 Tool Restriction — Limit Agent Attack Surface

**What:** Restrict the researcher agent's tool access to prevent arbitrary command execution that could circumvent file access controls.

**Why:** The autocrucible project gives its agent only 5 tools: Read, Edit, Write, Glob, Grep. No Bash. No subprocess. The rationale: an agent with shell access can `cat evaluate.py` even if the prompt says "don't read it." It can `curl` to exfiltrate data. It can modify files outside the project. Tool restriction is the difference between "please don't" and "you can't."

**Implementation:**
1. This is a tension point for Turing. The researcher agent currently has Bash access because it needs to:
   - Activate the venv: `source .venv/bin/activate`
   - Run training: `python train.py > run.log 2>&1`
   - Parse metrics: `grep -A 10 "^---" run.log`
   - Run git operations: `git checkout -b exp/NNN-description`
2. **Option A (conservative):** Keep Bash but restrict via `allowed-tools` in command frontmatter: `Bash(python train.py:*, python scripts/*:*, git:*, source .venv/bin/activate:*)` — allow only specific command patterns.
3. **Option B (aggressive):** Remove Bash entirely. Move experiment execution to the platform (like autocrucible). The agent edits files; a hook runs training and reports metrics. This is architecturally cleaner but requires significant refactoring of the experiment loop.
4. **Recommended:** Start with Option A (already partially done — `allowed-tools` was added to command frontmatter in the infrastructure modernization). Refine the patterns to whitelist only necessary commands. Evaluate Option B as a future phase when/if Claude Agent SDK integration happens.
5. Document the tradeoffs in an ADR.

**Acceptance:** The researcher agent cannot `cat evaluate.py`, `curl` to external services, or modify files outside the project directory via Bash. Whitelisted commands only.

### 5.5 Diff-Based History — Show What Actually Changed

**What:** Replace agent-generated experiment descriptions in the context window with actual git diffs of failed iterations.

**Why:** The autocrucible project A/B tested this approach and found it dramatically improved keep rates: **42-62% keep rate** with diff-based history vs 32% baseline. The reason: agent-generated descriptions of what they tried are unreliable (they contain self-verification text, markdown formatting artifacts, and sometimes misrepresent what actually changed). Showing the actual diff forces the agent to see exactly what was tried — and more importantly, what NOT to try again.

**Implementation:**
1. Update `templates/program.md` OBSERVE step: instead of reading free-text descriptions from MEMORY.md for failed experiments, the agent should read the git diff:
   ```bash
   # For each recent discarded experiment:
   git diff main..exp/NNN-description -- train.py config.yaml
   ```
2. For kept experiments, show one-line metric summaries (the code is already on main, no need for the diff).
3. Update `templates/scripts/show_metrics.py` — add `--with-diffs` flag that includes abbreviated diffs for discarded experiments.
4. Update the briefing command to include diffs in the "Exhausted Directions" section.
5. Strategy tiering based on consecutive failures (from autocrucible's approach):
   - **0-1 failures:** EXPLOIT — push further in the same direction
   - **2-3 failures:** RE-READ — re-read all code from scratch, the agent likely has a stale mental model
   - **4-5 failures:** COMBINE — combine two previously successful ideas
   - **6+ failures:** RADICAL — abandon current approach entirely
6. Add to `templates/program.md` as a strategy escalation protocol.

**Acceptance:** Discarded experiments show actual diffs in the agent's context. The agent's keep rate improves measurably (track as a metric in experiment logs).

### Summary: Defense-in-Depth Layers

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Architectural Separation (ADR-0002)               │
│  Hypothesis space vs measurement apparatus                   │
│  STATUS: Implemented                                        │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: Hidden File Tier (5.1)                            │
│  evaluate.py invisible to agent — prevents exploitation      │
│  STATUS: Implemented                                        │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3: Behavioral Probes (5.2)                           │
│  Training time, model usage, prediction diversity checks     │
│  STATUS: Implemented                                        │
├─────────────────────────────────────────────────────────────┤
│  LAYER 4: Statistical Validation (5.3 + Phase 2.1)          │
│  Multi-run evaluation, CV check, median aggregation          │
│  STATUS: Implemented                                        │
├─────────────────────────────────────────────────────────────┤
│  LAYER 5: Tool Restriction (5.4)                            │
│  Whitelisted Bash commands, no arbitrary execution           │
│  STATUS: Implemented                                        │
├─────────────────────────────────────────────────────────────┤
│  LAYER 6: Diff-Based History (5.5)                          │
│  Show actual changes, not agent descriptions                 │
│  STATUS: Implemented                                        │
└─────────────────────────────────────────────────────────────┘
```

Each layer addresses a different failure mode. Layers 1-3 prevent the agent from gaming the metric. Layer 4 prevents noise exploitation. Layer 5 limits the agent's attack surface. Layer 6 improves the agent's honest performance.

Turing is a Claude Code plugin — the human is always in the loop. Fully autonomous platform-managed execution (where the agent can't run Bash at all) is out of scope. If that need ever arises, it's a different tool.

---

## Implementation Order

| # | Feature | Phase | Priority | Status | Tests |
|---|---------|-------|----------|--------|-------|
| 1 | Hypothesis injection `/turing:try` | 1.1 | **Critical** | **DONE** | 14 |
| 2 | Research briefing `/turing:brief` | 1.2 | **Critical** | **DONE** | 9 |
| 3 | Experiment dependency graph | 1.3 | **High** | **DONE** | 7 |
| 4 | Multi-run statistical significance | 2.1 | **High** | **DONE** | 13 |
| 5 | Bayesian-guided suggestions | 2.2 | **Medium** | **DONE** | 11 |
| 6 | Automatic metric decomposition | 3.1 | **Medium** | **DONE** | 8 |
| 7 | Train/val gap monitoring | 3.2 | **Medium** | **DONE** | — |
| 8 | Structured experiment state | 4.1 | **Medium** | **DONE** | 13 |
| 9 | Hidden file tier | 5.1 | **Critical** | **DONE** | — |
| 10 | Behavioral probes | 5.2 | **Critical** | **DONE** | — |
| 11 | Stability validation | 5.3 | **High** | **DONE** | — |
| 12 | Tool restriction | 5.4 | **High** | **DONE** | — |
| 13 | Diff-based history | 5.5 | **Medium** | **DONE** | — |
| 14 | Novelty guard | 6.1 | **Critical** | **DONE** | 25 |
| 15 | Decision packets | 6.2 | **High** | **DONE** | 16 |
| 16 | Experiment families | 6.3 | **High** | **DONE** | 6 |
| 17 | Failure clustering | 6.4 | **Medium** | **DONE** | — |
| 18 | Research mode selection | 6.5 | **Medium** | **DONE** | — |

Phases 1-6 complete. All roadmap items implemented.

---

## Phase 6: MemoryLab Research-Ops Layer

*Source: [karpathy/autoresearch#172](https://github.com/karpathy/autoresearch/discussions/172) and [pauldebdeep9/autoresearch](https://github.com/pauldebdeep9/autoresearch)*

### Background

MemoryLab is a fork of Karpathy's autoresearch by pauldebdeep9 that adds an "operator-facing research memory layer" around the experiment loop. The upstream autoresearch has a tight loop but no memory of whether an idea is genuinely new, close to a prior success, or a likely repeat of a failed branch. MemoryLab wraps the loop with structured memory and decision infrastructure.

The fork's thesis: the experiment loop itself is solved (Karpathy proved that). What's missing is the layer *around* the experiments — remembering what was tried, distinguishing repeated failures from intentional follow-ups, and helping a human wake up to something more interpretable than a pile of logs.

### What MemoryLab Adds (Upstream Comparison)

| Capability | karpathy/autoresearch | pauldebdeep9/MemoryLab | Turing (current) |
|---|---|---|---|
| Memory format | Console logs + TSV | Structured JSONL ledger + archived artifacts | JSONL log + structured `experiment_state.yaml` (Phase 4.1) |
| Novelty control | None (agent decides) | History-aware guard (explore/exploit/replicate modes) | **NOT IMPLEMENTED** — agent uses MEMORY.md heuristically |
| Run identity | Commit-oriented | Run-centric (repeated runs on same commit stay distinct) | Experiment-oriented (`exp-NNN` IDs, but no multi-run distinction) |
| Best-run tracking | Manual inspection | Champion/challenger registry with lineage | `get_best_experiment()` function, but no challenger board |
| Decision guidance | Human inference only | Policy-driven packets (promote, abandon, retry, etc.) | Hypothesis queue (Phase 1.1) — but no post-run decision synthesis |
| Overnight reporting | Read logs by hand | Automated morning report + decision queue | `/turing:brief` (Phase 1.2) — similar but less structured |
| Experiment families | None | Family tags group related experiments | `parent_experiment` linkage (Phase 1.3) — similar concept |

### Overlap Analysis: What Turing Already Has

**Significant overlap (already implemented):**
- Structured experiment logging (JSONL) — Turing's `log_experiment.py` serves the same role as MemoryLab's ledger
- Experiment lineage — Turing's `parent_experiment` field and `show_experiment_tree.py` cover MemoryLab's lineage tracking
- Morning report — Turing's `/turing:brief` and `generate_brief.py` produce a similar report with campaign summary, trajectory, model types, hypothesis queue, and recommendations
- Hypothesis queue — Turing's `manage_hypotheses.py` with status transitions covers the "what to try next" aspect
- Structured state — Turing's `experiment_state.yaml` (Phase 4.1) serves the same purpose as MemoryLab's registry

**Genuine gaps (MemoryLab has, Turing doesn't):**

1. **Novelty guard** — the most valuable MemoryLab feature that Turing lacks entirely
2. **Decision packets** — automated post-run "what happened / what next?" synthesis
3. **Failure clustering** — identifying repeated failure patterns across experiments
4. **Experiment families** — grouping experiments by strategic theme (not just parent-child)
5. **Explore/exploit/replicate modes** — explicit research strategy selection

### Merit Assessment

**High merit — should implement:**

1. **Novelty Guard (6.1)** — This is the single most impactful feature Turing is missing. Currently the agent has no mechanism to check whether a proposed experiment is genuinely new, a repeat of a known failure, or an incremental follow-up to a success. Without it, the agent wastes iterations re-trying things it has already tried (especially across `/loop` sessions where context is lost). MemoryLab's approach is deliberately heuristic — rule-based token matching with alias tables, not embedding search — which keeps it fast, inspectable, and dependency-free.

2. **Decision Packets (6.2)** — After each experiment, MemoryLab synthesizes a compact verdict: `promote` (new champion), `branch_followup` (promising, explore further), `replicate` (needs confirmation), `abandon` (dead end), `fix_and_retry` (crashed, fixable). This is more structured than Turing's current approach where the agent makes a binary kept/discarded decision and updates free-text memory. Decision packets integrate naturally with the hypothesis queue — a `branch_followup` verdict can auto-queue a follow-up hypothesis.

3. **Experiment Families (6.3)** — Grouping experiments by strategic theme ("optimizer-sweep", "architecture-search", "feature-engineering") provides a higher-level view than the parent-child tree. The agent can see that 8 experiments in the "optimizer-sweep" family produced diminishing returns, suggesting it's time to switch families rather than continuing to tweak learning rates.

**Medium merit — useful but lower priority:**

4. **Failure Clustering (6.4)** — Identifying patterns across failures (e.g., "all experiments with max_depth > 8 overfit") helps the agent avoid entire regions of the search space rather than individual points. Currently this analysis happens in the agent's reasoning, but a structured clustering tool would make it more reliable.

5. **Research Mode Selection (6.5)** — The explore/exploit/replicate modes from MemoryLab provide an explicit strategy selector. Currently Turing's agent implicitly chooses between exploration and exploitation with no structured policy. Making this explicit would help the human steer: "we're in exploit mode now, focus on refining the current best."

**Low merit — Turing's approach is better:**

- **Champion/challenger registry** — Turing's `get_best_experiment()` + `generate_brief.py` already cover this. MemoryLab's registry is more elaborate but adds complexity for marginal benefit at Turing's scale.
- **Run-centric identity** — MemoryLab distinguishes repeated runs on the same commit. Turing's Phase 2.1 (multi-run statistical significance) handles this more rigorously with actual statistical tests rather than just bookkeeping.

### Design Constraints for Turing Integration

MemoryLab was built for a specific context: karpathy/autoresearch's single-GPU nanochat training loop with 5-minute fixed-budget runs. Turing's context is different:

1. **Project-scoped memory** — MemoryLab stores everything in `results/memorylab/`. Turing must scope memory per-project since one plugin manages multiple ML projects. The `experiment_state.yaml` and `hypotheses.yaml` already live in the ML project directory — new MemoryLab features must follow this pattern.

2. **Plugin architecture** — MemoryLab is a standalone CLI (`memorylab.py`). Turing's features are Claude Code commands and scripts. New features should follow the existing pattern: a Python script in `templates/scripts/` consumed by a command in `commands/`.

3. **Agent-native** — MemoryLab is operator-facing (the human runs `memorylab.py check`). In Turing, the agent runs the novelty check as part of its experiment loop. The novelty guard must be callable from `program.md` without human intervention.

4. **Alias tables** — MemoryLab's novelty matching uses ML-training-specific alias tables ("learning rate" = "step size" = "lr"). Turing is model-agnostic (XGBoost, LightGBM, neural nets, etc.) so alias tables need to be configurable per project, not hardcoded.

### Implementation Plan

#### 6.1 Novelty Guard — Prevent Duplicate Work

**What:** Before starting an experiment, check the proposed hypothesis against prior experiments. Classify as `novel`, `known_success`, `incremental_followup`, `repeat_failure`, or `duplicate_run`.

**Why:** The highest-ROI MemoryLab feature. Prevents the agent from wasting iterations on ideas it has already tried, especially across `/loop` sessions where context is lost. MemoryLab's author frames it well: "not perfect semantic search — the goal is to prevent obvious duplicate work."

**Implementation:**
1. Create `templates/scripts/novelty_guard.py` with:
   - `normalize_text(text)` — lowercase, strip stopwords, apply alias table, extract numbers
   - `similarity_score(text_a, text_b)` — blend of token overlap (Jaccard), number overlap, and concept overlap (using configurable concept patterns)
   - `classify_novelty(proposed, history, threshold=0.7)` — compare proposed text against all prior experiment descriptions, return classification + top matches + confidence
   - `check_novelty(proposed, log_path, mode="exploit")` — main entry point, applies mode policy on top of classification
2. Create `config/novelty_aliases.yaml` — configurable alias table (not hardcoded like MemoryLab):
   ```yaml
   phrase_aliases:
     "learning rate": "lr"
     "step size": "lr"
     "batch size": "batch_size"
     "gradient accumulation": "grad_accum"
   token_aliases:
     "increase": "up"
     "decrease": "down"
     "reduce": "down"
   concept_patterns:
     lr: ["lr", "learning_rate", "step_size"]
     architecture: ["depth", "width", "heads", "layers"]
     regularization: ["dropout", "weight_decay", "l1", "l2"]
   ```
3. Mode policies (from MemoryLab, adapted):
   - `explore`: allow `novel`, block `duplicate_run` and `repeat_failure`
   - `exploit`: allow `incremental_followup` and `known_success`, block `duplicate_run`
   - `replicate`: allow `duplicate_run`, block `novel`
4. Update `templates/program.md` HYPOTHESIZE step:
   ```bash
   python scripts/novelty_guard.py check \
     --description "increase max_depth to 8" \
     --log experiments/log.jsonl \
     --mode exploit
   ```
   If blocked, the agent must choose a different hypothesis.
5. Add to `templates/scripts/manage_hypotheses.py` — when adding an agent-generated hypothesis, run novelty check automatically. Human-injected hypotheses skip the guard (human taste overrides).
6. Add tests for normalization, similarity scoring, classification, and mode policies.

**Acceptance:** The agent cannot propose an experiment that is a near-duplicate of a prior failure without the guard flagging it. Human-injected hypotheses bypass the guard.

#### 6.2 Decision Packets — Post-Run Verdict Synthesis

**What:** After each experiment, synthesize a structured verdict that combines the run outcome, novelty classification, comparison to champion, and a recommended next action.

**Why:** Currently the agent makes a binary kept/discarded decision. Decision packets add nuance: "this was kept but only marginally better — replicate before declaring it champion" or "this crashed, but it was an OOM — reduce batch size and retry."

**Implementation:**
1. Create `templates/scripts/synthesize_decision.py` with:
   - `classify_outcome(metrics, best_metrics, config)` — determine if the run was a new champion, marginal improvement, lateral move, regression, or crash
   - `recommend_action(outcome, novelty_class, hypothesis)` — map to action: `promote`, `branch_followup`, `replicate`, `abandon`, `fix_and_retry`, `investigate_crash`
   - `format_decision_packet(run, outcome, action, evidence)` — produce a compact JSON summary
2. Decision packets are written to `experiments/decisions/exp-NNN.json` alongside the JSONL log.
3. Update `templates/program.md` RECORD step — after logging, synthesize a decision packet. The next iteration's OBSERVE step reads recent packets.
4. Integrate with hypothesis queue:
   - `branch_followup` → auto-queue a follow-up hypothesis as `agent` source, `medium` priority
   - `fix_and_retry` → auto-queue a retry hypothesis with the crash context
   - `abandon` → mark the associated hypothesis as `dead-end`
5. Include decision packets in `/turing:brief` output.
6. Add tests.

**Acceptance:** Every experiment produces a decision packet. `branch_followup` and `fix_and_retry` verdicts auto-populate the hypothesis queue.

#### 6.3 Experiment Families — Strategic Grouping

**What:** Tag experiments with a `family` label (e.g., "optimizer-sweep", "architecture-search") to group related experiments beyond the parent-child tree.

**Why:** The dependency tree shows individual lineage. Families show strategic themes. "All 8 experiments in the optimizer-sweep family produced diminishing returns" is a higher-level signal than "exp-007 is a child of exp-004."

**Implementation:**
1. Extend `log_experiment.py` — add optional `--family` and `--tags` CLI args. Store in JSONL entry.
2. Extend `manage_hypotheses.py` — add `--family` and `--tags` to hypothesis entries.
3. Create `templates/scripts/show_families.py` — group experiments by family, show per-family metrics:
   ```
   optimizer-sweep (8 experiments, 3 kept, best accuracy=0.87)
   architecture-search (4 experiments, 1 kept, best accuracy=0.85)
   feature-engineering (2 experiments, 2 kept, best accuracy=0.88)
   ```
4. Update `/turing:brief` to include family summary.
5. Update novelty guard — same-family experiments get a lower novelty threshold (it's expected that experiments in a family are similar).
6. Add tests.

**Acceptance:** `python scripts/show_families.py` displays per-family performance summaries. The agent and human can see when a family is exhausted.

#### 6.4 Failure Clustering — Pattern Detection Across Failures

**What:** Automatically identify patterns across failed experiments — which hyperparameter ranges, model types, or feature combinations consistently fail.

**Implementation:**
1. Add to `generate_brief.py` — a "Failure Patterns" section that groups discarded experiments by common traits:
   - "3/3 experiments with max_depth > 8 were discarded (overfitting)"
   - "All neural network experiments were discarded (dataset too small?)"
2. Use simple heuristics: group by model_type, by hyperparameter ranges, by family tags.
3. Surface in `/turing:brief` output.
4. Optionally feed back into novelty guard — experiments matching known failure patterns get flagged.

**Acceptance:** `/turing:brief` includes a "Failure Patterns" section when discarded experiments share common traits.

#### 6.5 Research Mode Selection — Explicit Strategy

**What:** Add a `/turing:mode` command that sets the research strategy: explore (try new things), exploit (refine what works), replicate (verify results).

**Implementation:**
1. Create `commands/mode.md` — `/turing:mode explore|exploit|replicate`
2. Store current mode in `experiment_state.yaml` under a `research_mode` key.
3. The novelty guard reads the mode and applies the appropriate policy.
4. The program.md HYPOTHESIZE step adapts behavior based on mode:
   - `explore`: prefer novel hypotheses, skip incremental tweaks
   - `exploit`: prefer follow-ups to best results, skip wild ideas
   - `replicate`: re-run best experiments with different seeds
5. Update router.
6. Add tests.

**Acceptance:** `/turing:mode exploit` causes the agent to focus on refining the current best rather than exploring new architectures.

### Implementation Order

| # | Feature | Phase | Priority | Depends On |
|---|---------|-------|----------|------------|
| 15 | Novelty guard | 6.1 | **Critical** | experiment log, alias config |
| 16 | Decision packets | 6.2 | **High** | 6.1 (novelty classification feeds into packets) |
| 17 | Experiment families | 6.3 | **High** | experiment log extension |
| 18 | Failure clustering | 6.4 | **Medium** | 6.3 (families make clustering more meaningful) |
| 19 | Research mode selection | 6.5 | **Medium** | 6.1 (mode drives novelty guard policy) |

Phase 6.1 (novelty guard) is the highest-priority item. It's the single MemoryLab feature that Turing cannot replicate with existing infrastructure. The rest build on top of it.

### What NOT to Adopt from MemoryLab

1. **VRAM tracking** — MemoryLab tracks `peak_vram_mb` because upstream autoresearch runs on a single GPU. Turing is model-agnostic and may not involve GPUs at all (XGBoost runs on CPU). Skip.
2. **Commit-centric identity** — MemoryLab indexes by git commit hash. Turing indexes by `exp-NNN` IDs which are cleaner for the hypothesis queue and dependency tree. Keep Turing's scheme.
3. **Hardcoded alias tables** — MemoryLab's `novelty.py` has 30+ hardcoded aliases for LLM training ("muon", "adamw", "kv", "fa", etc.). Turing must use a configurable YAML alias table since it supports arbitrary ML tasks.
4. **TSV compatibility layer** — MemoryLab maintains a compatibility TSV for upstream-style workflows. Turing already has its own TSV via `log_experiment.py`. No need for a second one.
5. **Fixed 5-minute budget** — MemoryLab assumes 5-minute training runs. Turing has configurable convergence via patience/threshold. Keep Turing's approach.

---

## Phase 7: Literature-Grounded Research Intelligence (Inspired by AERO)

*Make the agent read the literature before flipping coins.*

### Context: What AERO Is

[AERO](https://github.com/aether-raid/AERO) (Automated Exploration, Research & Orchestration) is an open-source ML research framework built on LangGraph. It implements five LLM-driven workflows that automate the research lifecycle: model selection, research planning, experiment design, result analysis, and paper writing. Each workflow is a LangGraph `StateGraph` with typed state, conditional routing, iterative critique-and-refine loops, and arXiv/Tavily integration for literature grounding.

AERO's five workflows:

| # | Workflow | AERO Module | What It Does |
|---|----------|-------------|-------------|
| 1 | **Model Researcher** | `src/aero/model_researcher/` | Analyzes task properties → searches arXiv → validates papers → suggests models with citations |
| 2 | **Research Planner** | `src/aero/research_planner/` | Generates problem statement → validates novelty via Tavily → creates structured research plan → critiques and refines |
| 3 | **Experiment Designer** | `src/aero/experiment_designer/` | Extracts research components → optionally tree-searches for methodology → designs experiments → generates validated code |
| 4 | **Experimentalist** | `src/aero/experimentalist/` | Analyzes results → identifies research directions → searches literature → generates follow-up experiments |
| 5 | **Report Writer** | `src/aero/report_writer/` | Analyzes results → sets up paper structure → finds sources → generates content → critiques → finalizes |

AERO's key architectural patterns that Turing can adapt:

1. **arXiv semantic search pipeline** — query generation → paper retrieval via arXiv API (`utils/arxiv.py`) → chunking → `sentence-transformers` embedding → FAISS indexing → cosine similarity ranking → LLM-scored relevance filtering (custom ranking prompts per workflow type) → validated paper set
2. **Iterative critique-and-refine loops** — generate → LLM critique with score (0-10) → conditional edge: if score >= 7.0 or refinements >= 3, finalize; else refine. Used in every AERO workflow.
3. **Tree search for idea exploration** — Monte Carlo Tree Search via `treequest` library over hypothesis/methodology space (`experiment_designer/idea_tree.py`), scored by LLM evaluation. Used when initial hypothesis is vague.
4. **Code generation with validation** — extract `[CODE_NEEDED]` tags from experiment designs → parallel LLM codegen → AST validation → iterative refinement (`experiment_designer/code.py`)
5. **Typed state machines** — `TypedDict` states with explicit fields for each workflow phase, enabling structured data flow, checkpoint/resume, and streaming support
6. **Task property classification** — 21 ML task categories in `model_researcher/shared_defs.py` (e.g., `temporal_structure`, `few_shot_learning`, `noise_robustness`) used to classify problems and generate targeted searches

### What's Relevant to Turing (and What's Not)

**Relevant — AERO workflows can feed INTO Turing's experiment loop:**
- Model Researcher → "which architectures should I try?" → feeds `hypotheses.yaml`
- Experiment Designer → "how should I structure this experiment?" → feeds `train.py` edits
- Experimentalist → "given these results, what next?" → feeds OBSERVE step
- Research Planner → "what's the research plan?" → feeds `/turing:init` setup

**Not relevant — Turing's domain is different:**
- Report Writer — Turing's output is trained models and experiment logs, not papers. The *critique-and-refine pattern* from Report Writer is relevant; the paper-generation workflow is not.
- Full literature review as a primary workflow — too heavyweight for a single experiment loop iteration. But targeted searches (3-5 papers) to inform a hypothesis are valuable.

### 7.1 Literature-Informed Model Selection — `/turing:suggest`

**What:** A new command that takes the current task description (from `config.yaml`) and uses arXiv search + LLM analysis to suggest model architectures worth trying, grounded in recent literature.

**Why:** Currently, the agent's model selection is based on whatever the LLM knows from training data. AERO's Model Researcher workflow demonstrates a better approach: analyze task properties → generate targeted arXiv search queries → retrieve and validate papers → extract model suggestions with citations. This gives the agent evidence-backed hypotheses instead of vibes.

**AERO source reference:** `src/aero/model_researcher/` — the full workflow graph:
```
analyze_properties_and_task → generate_search_query → search_arxiv → validate_papers
  ↓ (conditional: enough valid papers? _should_continue_with_papers)
    YES → suggest_models → critique_response → (conditional: _should_revise_suggestions)
      ACCEPT → END
      REVISE → revise_suggestions → critique_response (loop)
    NO  → search_arxiv (retry) or generate_search_query (new query)
```

Key implementation details from AERO:
- `shared_defs.py` defines `ML_RESEARCH_CATEGORIES` — 21 task property categories used to classify the task. Each category has a description (e.g., `temporal_structure`: "Data has inherent time dependencies or ordering"). The LLM detects which categories apply.
- `nodes/analyze_properties_nodes.py` — LLM classifies the user's task against the 21 categories, producing a `PropertyHit` list with `Evidence` objects (snippet, source, confidence score). Confidence is computed via independent-signal formula: `1 - product(1 - score_i)` with log-scaled evidence count bonus.
- `nodes/arxiv_search_nodes.py` — generates arXiv-formatted search queries from detected properties, retrieves papers via arXiv API (`utils/arxiv.py` with `format_search_string` for `all:%22term%22+AND+all:term` encoding), processes via `ArxivPaperProcessor` (chunk → embed with `sentence-transformers` → FAISS index → cosine similarity → LLM relevance scoring with custom `model_suggestion` ranking prompt)
- `nodes/suggestion_nodes.py` — synthesizes validated papers into model recommendations
- `edges/conditional_edges.py` — `_should_continue_with_papers` (checks paper count/quality, routes to retry if insufficient), `_should_revise_suggestions` (checks critique score, routes to revision if below threshold)

**Implementation for Turing:**
1. Create `commands/suggest.md` — a `/turing:suggest` skill that delegates to `@ml-evaluator` (read-only — it searches and suggests but doesn't edit)
2. Add `templates/scripts/suggest_models.py`:
   - Reads task description from `config.yaml` (field: `task.description`, to be added)
   - Classifies task against a configurable taxonomy (adapted from AERO's 21 categories, stored in `config/task_taxonomy.yaml`)
   - Queries arXiv API using `urllib.request` (same approach as AERO's `utils/arxiv.py`)
   - Embeds paper chunks with `sentence-transformers`, indexes in FAISS
   - LLM scores papers for relevance (adapt AERO's ranking prompt from `shared_defs.py:create_custom_ranking_prompt`)
   - LLM generates 3-5 model suggestions with citations and confidence scores
   - Outputs structured suggestions to `hypotheses.yaml` with `source: "literature"` and `citations: [...]`
3. Add `config/task_taxonomy.yaml` — configurable task property categories (fork of AERO's `ML_RESEARCH_CATEGORIES`, extensible per project)
4. Update router in `commands/turing.md` — add "suggest", "what model", "recommend" to routing table
5. Add tests for taxonomy classification, arXiv querying, paper scoring, and suggestion generation

**Dependencies:** `sentence-transformers`, `faiss-cpu` (add to `pyproject.toml`)

**Acceptance:** `/turing:suggest` reads the task config, searches arXiv, and produces 3-5 model architecture hypotheses with paper citations, auto-queued in `hypotheses.yaml`.

### 7.2 Experiment Design Scaffolding — `/turing:design`

**What:** Given a queued hypothesis (from `hypotheses.yaml`), generate a structured experiment design with implementation code — before the agent starts the main experiment loop.

**Why:** Currently, the agent in `/turing:train` edits `train.py` directly based on free-text reasoning. AERO's Experiment Designer demonstrates a more structured approach: extract research components → optionally tree-search for methodology → design experiment with literature grounding → generate validated code. This front-loads the thinking before the coding.

**AERO source reference:** `src/aero/experiment_designer/main.py` — the workflow graph:
```
extract_components → (conditional: has experiment ideas?)
  YES → design_and_codegen
  NO  → tree_search → design_and_codegen
```

The `design_and_codegen` node internally runs:
```
summarize → literature_search → plan → design → score
  → (loop: if any score < 70, refine and re-design, max 3 rounds)
  → codegen (4-node subgraph: extract_tags → generate_code [parallel] → validate_code [AST] → refine_code)
```

Key implementation details:
- `experiment.py` defines `ExperimentState` with fields: `experiment_input`, `summary`, `literature_results`, `full_design_content`, `refined_design_content`, `scores` (dict), `refinement_round`, `refinement_suggestions`
- `idea_tree.py` uses `treequest` (MCTS library) — `ExperimentTreeSystem` initializes with literature context from arXiv, then tree-searches over methodology space. Each node is an `IdeaState` with `level`, `content`, `score`, `citations`, `references`. The system runs 10 iterations per hypothesis.
- `code.py` defines `CodeGenState` and a 4-node `build_codegen_graph()`: tags are extracted via regex `\[CODE_NEEDED(?::\s*([^\]]+))?\]`, code is generated in parallel with `asyncio.gather`, validated via `ast.parse`, and refined if invalid. Missing imports are detected by `importlib.util.find_spec`.
- `search.py` builds a FAISS semantic index from arXiv papers for the current hypothesis, using Google Custom Search for dataset links from known repositories (OpenNeuro, PhysioNet, Kaggle, Zenodo, etc.)

**Implementation for Turing:**
1. Create `commands/design.md` — a `/turing:design [hypothesis-id]` skill
2. Add `templates/scripts/design_experiment.py`:
   - Reads the hypothesis from `hypotheses.yaml` by ID
   - Searches arXiv for 3-5 relevant methodology papers (reuses arXiv pipeline from 7.1)
   - LLM generates structured experiment design:
     ```yaml
     objective: "what we're testing"
     method: "how we'll test it"
     expected_outcome: "what success looks like"
     code_changes:
       - file: train.py
         description: "change model from XGBoost to LightGBM"
         diff_preview: "..."
     evaluation_criteria: "which metrics, what threshold"
     estimated_runs: 3
     ```
   - Optionally generates a `train.py` diff/patch as a starting point
   - Validates generated code via AST parsing (from AERO's `code.py` pattern)
   - LLM scores the design on feasibility (0-10), novelty (0-10), clarity (0-10)
   - If any score < 7.0 and refinements < 3, refine (AERO's critique loop pattern)
   - Outputs to `experiments/designs/hyp-NNN-design.md`
3. The agent in `/turing:train` can optionally read the design before editing `train.py`
4. Update router — add "design", "plan experiment" to routing table
5. Add tests

**Acceptance:** `/turing:design hyp-003` produces a structured experiment design with implementation guidance, scored for quality, with literature citations.

### 7.3 Result-Driven Follow-Up Suggestions — Enhanced `/turing:brief --deep`

**What:** Enhance the existing `/turing:brief` command to include literature-grounded follow-up experiment suggestions based on current results, adapting AERO's Experimentalist workflow.

**Why:** Currently, `/turing:brief` generates recommendations from the agent's own reasoning + Bayesian suggestions (Phase 2.2). AERO's Experimentalist goes further: analyze results → identify research directions → search literature for methodology → distill paper methodologies → generate validated follow-up experiments with implementation roadmaps.

**AERO source reference:** `src/aero/experimentalist/` — the workflow graph:
```
analyze_data → research_direction → generate_search_query → search_arxiv → validate_papers
  → distill_methodologies → generate_experiments → validate_experiments
    ↓ (conditional: PASS?)
    YES → prioritize → implementation_roadmap → finalize
    NO  → generate_experiments (retry with cumulative feedback, track past_experiment_mistakes)
```

Key implementation details:
- `shared_defs.py:ExperimentSuggestionState` tracks: `past_fixed_issues`, `past_unresolved_issues`, `most_recent_generation_issues`, `cumulative_validation_feedback`, `past_experiment_mistakes` — the LLM learns from its own generation failures across iterations
- `nodes/experiment_generation_nodes.py:_distill_paper_methodologies_node` — processes top 5 validated papers, extracting methodology in <=600 characters each (tight budget forces precision)
- `nodes/research_direction_nodes.py` — analyzes experimental findings to identify promising research directions
- The validation loop has explicit "past mistakes" tracking: each failed generation attempt is recorded with its issues, and the next generation attempt receives cumulative feedback so the LLM doesn't repeat the same mistakes

**Implementation for Turing:**
1. Extend `commands/brief.md` — add `--deep` flag for literature-grounded analysis
2. Add `templates/scripts/suggest_next_literature.py`:
   - Reads `log.jsonl` and `experiment_state.yaml` for current results
   - Identifies improvement patterns and stagnation points (reuse `generate_brief.py` analysis)
   - Searches arXiv for papers with similar experimental setups/challenges
   - Distills methodology from retrieved papers (<=600 chars each, AERO's approach)
   - Generates 3-5 follow-up experiment suggestions with:
     - Literature citation
     - Expected impact estimate
     - Implementation complexity (low/medium/high)
     - Specific `train.py` changes needed
   - Tracks past suggestion failures to avoid repeating bad recommendations (AERO's `past_experiment_mistakes` pattern)
   - Auto-queues as hypotheses with `source: "literature-brief"` and `priority: "medium"`
3. Update `generate_brief.py` to include a "Literature-Grounded Suggestions" section when `--deep` is passed
4. Add tests

**Acceptance:** `/turing:brief --deep` includes a section with literature-backed suggestions referencing specific papers, auto-queued as hypotheses.

### 7.4 Research Plan Generation — Enhanced `/turing:init --plan`

**What:** Enhance the scaffolding command to optionally generate a research plan before the first experiment, using AERO's Research Planner pattern.

**Why:** Currently, `/turing:init` scaffolds files and the agent starts experimenting immediately. AERO demonstrates value in planning first: generate problem statement → validate novelty via web search → create structured research plan → critique and refine.

**AERO source reference:** `src/aero/research_planner/main.py` — the workflow graph:
```
initialize_clients → generate_problem → validate_problem
  ↓ (conditional: _streamlined_validation_decision)
    accept → create_research_plan → critique_plan
      ↓ (conditional: _determine_refinement_path)
        finalize → END
        refine → create_research_plan (loop, max 3 refinements)
    reject → process_rejection_feedback → generate_problem (retry, max 10 attempts)
```

Key implementation details:
- `validate.py` uses Tavily web search to check if the problem is already solved (novelty validation)
- `critique.py` scores plans on `overall_score` (0-10); threshold 7.0 for acceptance, max 3 refinements
- Safety valve: after 10 generation attempts, force-accepts to prevent infinite loops

**Implementation for Turing:**
1. Extend `commands/init.md` — add `--plan` flag: `/turing:init --plan`
2. Add `templates/scripts/generate_research_plan.py`:
   - Reads task description from user input or `config.yaml`
   - Generates a structured research plan:
     - Model families to explore (ordered by expected relevance)
     - Evaluation strategy (which metrics, multi-run config recommendation)
     - Search budget allocation (how many experiments per family)
     - Success criteria (target metric, convergence definition)
     - Risk factors (overfitting risk, data quality concerns)
   - Optionally validates against arXiv (is this well-studied? what approaches dominate?)
   - LLM critiques and refines the plan (score → refine if < 7.0, max 3 rounds)
   - Writes to `RESEARCH_PLAN.md` in the ML project root
3. Update `templates/program.md` — the OBSERVE step reads `RESEARCH_PLAN.md` for strategic direction
4. The plan is advisory — the agent can deviate but should note why in `experiment_state.yaml`
5. Add tests

**Acceptance:** `/turing:init --plan` produces a `RESEARCH_PLAN.md` giving the agent strategic direction for its first 5-10 experiments.

---

## Phase 8: Critique-and-Refine Loops (Inspired by AERO)

*Make every generated artifact self-improving.*

### What AERO Demonstrates

Every AERO workflow uses the same meta-pattern: **generate → critique → refine (loop until quality threshold or max iterations)**. Implemented as conditional graph edges:

```python
# From AERO's research_planner/main.py
workflow.add_conditional_edges(
    "critique_plan",
    _determine_refinement_path,
    {
        "finalize_plan": "finalize_plan",      # score >= 7.0 or refinements >= 3
        "refine_plan": "create_research_plan",  # score < 7.0, try again
    }
)
```

Constants across all AERO workflows:
- **Threshold:** 7.0/10 for acceptance
- **Max refinements:** 3 (prevents infinite loops)
- **Score history:** `critique_score_history: List[float]` for convergence detection
- **Previous versions:** stored for diff-aware refinement
- **Cumulative issues:** critique builds on prior issues rather than starting fresh

### 8.1 Critique Loops for Hypothesis Generation

**What:** When the agent generates hypotheses (in `/turing:train`'s OBSERVE step or via `/turing:suggest`), run a critique pass before committing to execution.

**Why:** A 30-second LLM critique is cheaper than a 30-minute wasted training run. AERO's Model Researcher revision loop catches vague suggestions, unsupported claims, and redundant recommendations.

**Implementation:**
1. Add `templates/scripts/critique_hypothesis.py`:
   - Takes a hypothesis description + experiment history context
   - LLM scores on: novelty (cross-reference with novelty guard from 6.1), feasibility (given current infrastructure), expected impact (based on experiment history)
   - Returns score (0-10) and specific concerns
   - If score < 5.0, suggest modifications or reject
   - Integrates with novelty guard — a "repeat_failure" hypothesis automatically scores 0 on novelty
2. Integrate into `templates/program.md` HYPOTHESIZE step: critique before executing
3. Update `manage_hypotheses.py` — add `critique` subcommand
4. Track critique scores in `hypotheses.yaml` for meta-learning
5. Add tests

**Acceptance:** Hypotheses below score 5.0 are flagged before execution. The agent must modify or justify proceeding.

### 8.2 Critique Loops for Briefing Reports

**What:** Self-critique pass on generated briefing reports before presenting to the human.

**Implementation:**
1. Extend `generate_brief.py` — after generating, run LLM critique:
   - Are recommendations specific enough to act on?
   - Do "exhausted directions" cover all failed approaches in the log?
   - Are convergence estimates grounded in data?
2. If critique identifies gaps, regenerate affected sections (max 2 rounds)
3. Add tests

**Acceptance:** Briefing reports are self-critiqued. Recommendations are concrete and actionable.

---

## Phase 9: Semantic Experiment Memory (Inspired by AERO)

*Give the agent a TF-IDF index instead of a text file.*

### What AERO Demonstrates

AERO uses `sentence-transformers` + FAISS for semantic retrieval across all workflows:
1. Chunk documents into segments
2. Embed with `sentence-transformers` (e.g., `all-MiniLM-L6-v2`)
3. Index in FAISS (flat L2 index)
4. At query time, embed the query, retrieve top-K similar chunks by cosine similarity
5. Use retrieved chunks as LLM context

AERO's `experiment_designer/search.py` demonstrates the full pipeline including dataset link discovery from known repositories (OpenNeuro, PhysioNet, Kaggle, Zenodo, etc.).

### 9.1 Semantic Experiment Index

**What:** Embed experiment descriptions and results in a FAISS index. When the agent asks "what have I tried that's similar to X?", retrieve by semantic similarity instead of scanning the full log.

**Why:** Phase 4.1 structured the experiment state into YAML. But as experiments accumulate (50+), the agent's context window fills with irrelevant history. Phase 6.1's novelty guard uses heuristic token matching — fast but misses semantic similarity ("try a deeper tree" vs "increase max_depth"). A semantic index is complementary: novelty guard for fast blocking, semantic index for nuanced retrieval.

**Implementation:**
1. Add `templates/scripts/embed_experiments.py`:
   - Reads `log.jsonl` entries
   - Embeds each experiment: `f"{description} | config: {config_summary} | result: {metric}={value} | status: {kept/discarded}"`
   - Stores in FAISS flat index at `experiments/index.faiss` + `experiments/index.pkl` (metadata)
   - Supports incremental updates (track last-indexed experiment ID)
2. Add `templates/scripts/query_experiments.py`:
   - Takes natural language query (e.g., "experiments with high learning rate that overfit")
   - Embeds query, retrieves top-K (default 5) similar experiments with similarity scores
   - Outputs structured results with experiment IDs, descriptions, metrics
3. Update `templates/program.md` OBSERVE step: query the index with current hypothesis before deciding
4. Update `/turing:brief` — use index for "Exhausted Directions" section
5. Add tests

**Dependencies:** `sentence-transformers`, `faiss-cpu` (shared with Phase 7.1)

**Acceptance:** With 50+ experiments, the agent retrieves relevant history in <1 second via semantic search instead of loading the entire log into context.

---

## Updated Full Implementation Order

| # | Feature | Phase | Priority | Status | Tests |
|---|---------|-------|----------|--------|-------|
| 1 | Hypothesis injection `/turing:try` | 1.1 | **Critical** | **DONE** | 14 |
| 2 | Research briefing `/turing:brief` | 1.2 | **Critical** | **DONE** | 9 |
| 3 | Experiment dependency graph | 1.3 | **High** | **DONE** | 7 |
| 4 | Multi-run statistical significance | 2.1 | **High** | **DONE** | 13 |
| 5 | Bayesian-guided suggestions | 2.2 | **Medium** | **DONE** | 11 |
| 6 | Automatic metric decomposition | 3.1 | **Medium** | **DONE** | 8 |
| 7 | Train/val gap monitoring | 3.2 | **Medium** | **DONE** | — |
| 8 | Structured experiment state | 4.1 | **Medium** | **DONE** | 13 |
| 9 | Hidden file tier | 5.1 | **Critical** | **DONE** | — |
| 10 | Behavioral probes | 5.2 | **Critical** | **DONE** | — |
| 11 | Stability validation | 5.3 | **High** | **DONE** | — |
| 12 | Tool restriction | 5.4 | **High** | **DONE** | — |
| 13 | Diff-based history | 5.5 | **Medium** | **DONE** | — |
| 14 | Platform-managed execution | 5.6 | **Medium** | Future | — |
| 15 | Novelty guard | 6.1 | **Critical** | **DONE** | 25 |
| 16 | Decision packets | 6.2 | **High** | **DONE** | 20 |
| 17 | Experiment families | 6.3 | **High** | **DONE** | 6 |
| 18 | Failure clustering | 6.4 | **Medium** | **DONE** | — |
| 19 | Research mode selection | 6.5 | **Medium** | **DONE** | — |
| 20 | Literature-informed model selection `/turing:suggest` | 7.1 | **High** | **DONE** | — |
| 21 | Experiment design scaffolding `/turing:design` | 7.2 | **Medium** | **DONE** | — |
| 22 | Literature-grounded briefing `/turing:brief --deep` | 7.3 | **Medium** | **DONE** | — |
| 23 | Research plan generation `/turing:init --plan` | 7.4 | **Low** | **DONE** | — |
| 24 | Critique loops for hypothesis generation | 8.1 | **High** | **DONE** | 20 |
| 25 | Critique loops for briefing reports | 8.2 | **Low** | **DONE** | — |
| 26 | Semantic experiment index (TF-IDF) | 9.1 | **Medium** | **DONE** | 21 |

All phases 1-9 complete except 5.6 (platform-managed execution, deferred until Agent SDK integration). 257 tests passing.

---

## Appendix: K-Dense BYOK Evaluation

*Source: [K-Dense-AI/k-dense-byok](https://github.com/K-Dense-AI/k-dense-byok)*
*Evaluated: 2026-03-31*

### What K-Dense BYOK Is

A desktop AI research assistant with a React frontend, Python backend, and LiteLLM proxy routing to 40+ models. The system has a main agent ("Kady") that delegates tasks to specialized "experts" via the Gemini CLI. It ships with 326 workflow templates across 22 scientific disciplines, access to 229 scientific databases, and 170+ skills from the K-Dense skill library.

### Architecture

```
React Frontend (port 3000) → Python Backend (port 8000) → LiteLLM Proxy (port 4000)
                                    ↓
                              Kady Agent (google.adk.agents.LlmAgent)
                                    ↓
                              delegate_task() → Gemini CLI subprocess
                                    ↓
                              K-Dense Skills (170+ scientific skills)
```

The agent is built on Google's Agent Development Kit (`google.adk`). Expert delegation happens via `delegate_task()` which spawns Gemini CLI as a subprocess, captures JSONL stream output, and extracts activated skills and tool usage.

### Overlap Assessment with Turing

| K-Dense Feature | Turing Equivalent | Worth Adopting? |
|---|---|---|
| 326 workflow templates | `/turing:init --plan` research plan | **No** — different scope |
| 229 database catalog | Not applicable | **No** — Turing is model-agnostic |
| Multi-model routing (40+ models) | Single Claude Code agent | **No** — architectural mismatch |
| Expert delegation via Gemini CLI | Two-agent architecture (ADR-0003) | **No** — Turing's approach is simpler and sufficient |
| Sandbox file management | Template scaffolding (ADR-0008) | **No** — different paradigm |
| MCP server extensibility | Claude Code plugin system | **No** — already extensible |
| React web UI | CLI-first (Claude Code) | **No** — different interface paradigm |

### Verdict: Nothing to Adopt

K-Dense BYOK and Turing solve fundamentally different problems:

**K-Dense BYOK** is a *general scientific assistant* — a desktop app where a human chats with an AI that can delegate to specialized experts across 22 scientific disciplines. It is broad (326 workflows) but shallow (each workflow is a one-shot prompt template, not an iterative loop). It has no experiment tracking, no convergence detection, no hypothesis queue, no immutable evaluation, no memory across sessions. It is a research *dispatcher*, not a research *engine*.

**Turing** is a *focused ML experiment engine* — a CLI plugin that autonomously iterates through a hypothesis-test-decide loop with structured memory, statistical rigor, anti-cheating guardrails, and a formal taste-leverage interface. It is narrow (ML experiments only) but deep (19 implemented features across 9 phases, 257 tests, 16 ADRs).

The two systems are complementary, not competitive. A researcher could use K-Dense BYOK's "Build a Classifier" workflow to get initial ideas, then use Turing to systematically iterate on those ideas with discipline.

**Specific features evaluated:**

1. **Workflow templates** — K-Dense's 15 ML workflows ("Build a Classifier", "Hyperparameter Tuning", "Model Comparison") are prompt templates with placeholder variables. They are equivalent to Turing's `program.md` but less structured — no iteration, no memory, no convergence. Turing's `/turing:init --plan` already generates a more detailed research plan grounded in literature search. **Not useful.**

2. **Multi-model routing** — K-Dense routes to 40+ models via LiteLLM/OpenRouter. Turing runs inside Claude Code which handles model routing. Adding multi-model support would contradict Turing's single-agent architecture (ADR-0003) and add complexity without clear benefit for the ML experiment loop. **Not useful.**

3. **Expert delegation** — K-Dense's `delegate_task()` spawns Gemini CLI as a subprocess for specialized work. This is architecturally interesting but solves a different problem (breadth across disciplines) than Turing needs (depth in one experiment loop). Turing already has two agents with distinct capabilities. **Not useful.**

4. **Scientific database catalog** — 229 databases across genomics, finance, astronomy, etc. This is a data discovery feature, not an experiment infrastructure feature. Turing assumes the user already has their data. **Not useful.**

5. **MCP extensibility** — Custom Model Context Protocol servers configured via JSON. Claude Code already has its own MCP integration. **Not useful.**

6. **Skill library** — 170+ K-Dense scientific skills activated via Gemini CLI. These are domain-specific tools (scikit-learn, SHAP, PyTorch Lightning). Turing's templates already include scikit-learn, XGBoost, and LightGBM. The skill activation model ("not always reliable" per K-Dense's own docs) is less robust than Turing's direct Python imports. **Not useful.**

### What Would Be Useful (If K-Dense Had It)

The features that *would* be worth adopting from a system like K-Dense — if it had them — are exactly the features it's missing:

- Experiment memory across sessions (Turing has it: Phases 1, 4, 6)
- Iterative experiment loops with convergence (Turing has it: ADR-0006)
- Hypothesis tracking and novelty detection (Turing has it: Phases 1.1, 6.1)
- Immutable evaluation with anti-cheating (Turing has it: ADRs 0002, Phase 5)
- Statistical significance testing (Turing has it: Phase 2.1)

K-Dense BYOK is impressive infrastructure for multi-disciplinary scientific assistance. It is not relevant to Turing's core experiment loop.

### What IS Worth Stealing (Revised Assessment)

Two narrow features from K-Dense improve Turing's existing literature capabilities:

#### A. Scholarly API Catalog for `/turing:suggest` and `/turing:design`

K-Dense catalogs 15 scholarly publication APIs. Turing's literature features currently use `WebSearch` blindly. A curated catalog of *which API to query for what* makes searches more targeted:

| API | What it provides | Use case in Turing |
|-----|-----------------|-------------------|
| [Semantic Scholar](https://api.semanticscholar.org/api-docs/) | AI-enhanced paper search, citations, recommendations | `/turing:suggest` — find papers for a model architecture |
| [OpenAlex](https://docs.openalex.org) | Open scholarly metadata, 200M+ works | `/turing:design` — find benchmark results for a task |
| [arXiv API](https://info.arxiv.org/help/api/) | Preprints in CS/ML/AI | `/turing:suggest` — find cutting-edge approaches |
| [CrossRef](https://api.crossref.org) | DOI metadata for 150M+ works | `/turing:brief --deep` — resolve citation metadata |
| [Papers With Code](https://paperswithcode.com/api/v1/docs/) | SOTA leaderboards, datasets, methods | `/turing:init --plan` — find baseline metrics for a task |

**Implementation:** Add `config/scholarly_apis.yaml` listing these APIs with their base URLs, query patterns, and which Turing command uses them. Update `/turing:suggest`, `/turing:design`, and `/turing:init --plan` to prefer these APIs over blind web search when available. This is a config-only change plus minor updates to the WebSearch queries in the command files.

**Priority:** Low — improves quality of literature grounding but doesn't change functionality. The current WebSearch approach works; this makes it more precise.

#### B. Structured Experiment Archetypes for `/turing:try`

K-Dense's 15 ML workflows are structured prompts with placeholders: "Build a Classifier: (1) EDA, (2) Feature engineering, (3) Train {models}, (4) Evaluate {metrics}, (5) SHAP interpretability, (6) Tune best model." Turing's `/turing:try` accepts free text.

A library of *experiment archetypes* would give both humans and the agent structured starting points:

```yaml
# config/experiment_archetypes.yaml
archetypes:
  model_comparison:
    description: "Systematic comparison of multiple model families"
    steps: ["Train all models with identical preprocessing", "Cross-validate", "Statistical comparison (paired t-test)", "Report"]
    suggested_families: ["xgboost", "lightgbm", "random_forest", "mlp"]

  feature_sweep:
    description: "Systematic feature engineering exploration"
    steps: ["Analyze feature importance", "Add interaction features", "Add polynomial features", "Evaluate each addition independently"]

  regularization_search:
    description: "Find optimal regularization for current best model"
    steps: ["Vary weight_decay/dropout/max_depth", "Plot train-val gap vs regularization", "Find elbow point"]

  ensemble_construction:
    description: "Combine top-performing models"
    steps: ["Select top 3 diverse models", "Try voting ensemble", "Try stacking", "Compare with individual bests"]
```

**Implementation:** Add `config/experiment_archetypes.yaml`. Update `/turing:try` to accept archetype names: `/turing:try archetype:model_comparison`. The hypothesis description is auto-generated from the archetype template with the project's specific metric and model type filled in.

**Priority:** Medium — improves hypothesis quality by giving structured approaches rather than ad-hoc text. Particularly valuable for the agent's self-generated hypotheses (not just human injections).

---

## Phase 10: Statistical Rigor (v1.3.0)

*Stop publishing lucky seeds. Start publishing distributions.*

Phase 2.1 added optional multi-run significance testing. This phase makes statistical rigor a first-class workflow with dedicated commands.

### 10.1 Multi-Seed Runner — `/turing:seed`

**What:** Run the same experiment across N random seeds, compute mean/std/confidence intervals, and flag results that are seed-sensitive. Prevents publishing lucky seeds.

**Why:** Phase 2.1's `statistical_compare.py` requires manual `n_runs` config. `/turing:seed` makes multi-seed evaluation a one-command operation that any researcher reaches for before claiming a result. The difference: Phase 2.1 is infrastructure the agent uses during training; `/turing:seed` is a human-facing verification command.

**Implementation:**
1. Create `commands/seed.md` — `/turing:seed [N]` (default N=5)
2. Add `templates/scripts/seed_runner.py`:
   - Reads current best experiment config from `experiment_state.yaml`
   - Runs `train.py` N times with seeds `[42, 123, 456, 789, 1024, ...]` (configurable in `config.yaml`)
   - Collects per-seed metrics into a structured result:
     ```yaml
     seeds_run: [42, 123, 456, 789, 1024]
     metric: accuracy
     results: [0.872, 0.868, 0.871, 0.855, 0.873]
     mean: 0.8678
     std: 0.0071
     ci_95: [0.859, 0.877]
     cv_percent: 0.82
     seed_sensitive: false  # true if CV > 5%
     worst_seed: 789
     best_seed: 1024
     ```
   - Writes results to `experiments/seed_studies/exp-NNN-seeds.yaml`
   - Prints a summary table with pass/fail verdict
3. Integration with `/turing:brief` — seed study results appear in the "Current Best" section
4. Add `--quick` flag: runs 3 seeds instead of 5 for fast checks
5. Add tests for seed runner, CI computation, sensitivity detection

**Acceptance:** `/turing:seed` runs 5 seeds, reports mean±std with 95% CI, and flags seed-sensitive results (CV > 5%) before the researcher publishes.

### 10.2 Reproducibility Verification — `/turing:reproduce`

**What:** Given an experiment ID, re-run it from the logged config and verify metrics fall within a confidence interval of the original. Catches non-determinism, environment drift, and silent data changes.

**Why:** Reviewers ask "did you check reproducibility?" and the honest answer is usually "no." This command makes it trivial. It also catches environment drift — if you upgrade a library and your results shift, `/turing:reproduce` will tell you before a reviewer does.

**Implementation:**
1. Create `commands/reproduce.md` — `/turing:reproduce <exp-id> [--tolerance 0.02]`
2. Add `templates/scripts/reproduce_experiment.py`:
   - Reads the experiment entry from `log.jsonl` by ID
   - Extracts the full config (hyperparameters, seed, model type)
   - Reconstructs `train.py` from the experiment's git commit: `git show exp/NNN:train.py`
   - Runs the experiment with identical config
   - Compares new metrics against original:
     - **Exact match** (deterministic algorithms): metrics must match within float tolerance (1e-6)
     - **Statistical match** (stochastic algorithms): runs N times (default 3), checks if original metric falls within the 95% CI of the new distribution
   - Reports verdict: `reproducible`, `approximately_reproducible` (within tolerance), `not_reproducible` (outside CI), `environment_changed` (different library versions detected)
   - Writes report to `experiments/reproductions/exp-NNN-repro.yaml`
3. Environment snapshot: capture `pip freeze` at experiment time (extend `log_experiment.py`) and diff against current environment during reproduction
4. Add `--strict` flag: exact match only, no statistical tolerance
5. Add tests

**Depends on:** Experiment logging (Phase 1), git-based experiment tracking

**Acceptance:** `/turing:reproduce exp-042` re-runs the experiment and reports whether results are within statistical tolerance. Environment differences are flagged.

---

## Phase 11: Experiment Intelligence (v1.4.0)

*Understand your experiments deeper than aggregate metrics.*

### 11.1 Error Analysis — `/turing:diagnose`

**What:** Cluster failure cases from the current best model, identify systematic failure modes, and suggest targeted fixes. Goes beyond aggregate metrics to answer "where and why does this model fail?"

**Why:** A model with 87% accuracy is hiding 13% of failures. Are those failures random, or does the model consistently fail on a specific subpopulation (long sequences, rare classes, noisy features)? `/turing:diagnose` answers this and feeds actionable hypotheses into the experiment queue.

**Implementation:**
1. Create `commands/diagnose.md` — `/turing:diagnose [exp-id]` (defaults to current best)
2. Add `templates/scripts/diagnose_errors.py`:
   - Loads the model and runs inference on the validation set
   - Collects all misclassified/high-error samples
   - For classification: builds confusion matrix, identifies most-confused class pairs, analyzes per-class precision/recall (extends Phase 3.1's metric decomposition)
   - For regression: identifies high-residual samples, bins by feature ranges to find systematic bias
   - Clusters error cases by feature similarity (k-means on feature vectors of misclassified samples)
   - For each cluster, describes the failure mode:
     ```yaml
     failure_modes:
       - id: fm-001
         description: "Model confuses class 'cat' and 'dog' — 42% of all errors"
         affected_samples: 127
         suggested_fix: "Add pet-specific features or increase training data for these classes"
         auto_hypothesis: "Add breed-specific features to distinguish cat vs dog"
       - id: fm-002
         description: "High error on samples where feature_3 > 100 — model extrapolates poorly"
         affected_samples: 34
         suggested_fix: "Add feature_3 binning or cap outliers"
         auto_hypothesis: "Bin feature_3 into quantiles instead of raw values"
     ```
   - Optionally auto-queues hypotheses from failure modes into `hypotheses.yaml` with `source: "diagnose"`
3. Writes report to `experiments/diagnoses/exp-NNN-diagnosis.md`
4. Integration with `/turing:brief` — failure modes appear in recommendations
5. Add tests

**Acceptance:** `/turing:diagnose` identifies the top 3-5 systematic failure modes with actionable fix suggestions. Auto-queued hypotheses target specific weaknesses.

### 11.2 Systematic Ablation Studies — `/turing:ablate`

**What:** Given a model configuration, systematically remove or disable components one at a time, measure the impact on the primary metric, and produce an ablation table ready for a paper.

**Why:** Ablation tables are required by every ML venue and are the most tedious part of writing a paper. Each row requires a separate training run with one component removed. `/turing:ablate` automates the mechanical work — the researcher chooses which components to ablate, the system runs the experiments and formats the table.

**Implementation:**
1. Create `commands/ablate.md` — `/turing:ablate [exp-id] [--components "feature_X,regularization,augmentation"]`
2. Add `templates/scripts/ablation_study.py`:
   - Reads the experiment's config from `log.jsonl`
   - If `--components` not specified, auto-detects ablatable components:
     - Feature groups (if feature engineering is used)
     - Regularization terms (dropout, weight_decay, max_depth limits)
     - Preprocessing steps (normalization, encoding)
     - Model-specific components (number of estimators, layer count)
   - For each component, creates a modified config with that component removed/disabled
   - Runs all ablation experiments (optionally in parallel via `/turing:seed` with N=3 for statistical robustness)
   - Produces ablation table:
     ```
     | Configuration          | Accuracy | Δ from Full |
     |------------------------|----------|-------------|
     | Full model             | 0.872    | —           |
     | − feature_X            | 0.851    | −0.021      |
     | − regularization       | 0.863    | −0.009      |
     | − augmentation         | 0.870    | −0.002      |
     | − feature_Y            | 0.874    | +0.002      |
     ```
   - Ranks by impact (largest Δ first)
   - Flags components with positive Δ when removed (dead weight — removing them improves the model)
   - Writes to `experiments/ablations/exp-NNN-ablation.md`
3. Integration with `/turing:paper` (Phase 14.2) — ablation tables are auto-included
4. Add `--latex` flag for LaTeX-formatted table output
5. Add tests

**Depends on:** Seed runner (Phase 10.1) for statistical robustness of ablation results

**Acceptance:** `/turing:ablate` produces a publication-ready ablation table showing the impact of each component. Dead-weight components are flagged.

### 11.3 Pareto Frontier Visualization — `/turing:frontier`

**What:** Visualize the Pareto frontier across multiple objectives (accuracy vs. latency vs. parameter count vs. memory) from experiment history. Answers "which model is actually best?" when there are tradeoffs.

**Why:** Researchers often have multiple objectives that trade off against each other. The "best" model depends on deployment constraints. A Pareto frontier makes the tradeoff space visible so the researcher can make an informed choice rather than optimizing a single metric blindly.

**Implementation:**
1. Create `commands/frontier.md` — `/turing:frontier [--metrics "accuracy,latency_ms,n_params"]`
2. Add `templates/scripts/pareto_frontier.py`:
   - Reads all experiments from `log.jsonl`
   - Extracts specified metrics (defaults to primary metric + any secondary metrics in config)
   - Computes Pareto-optimal set: experiments where no other experiment is strictly better on all metrics
   - Produces output:
     - Text table of Pareto-optimal experiments with all metrics
     - ASCII scatter plot (2D projection for the two most important metrics)
     - Dominated experiments marked with their closest Pareto-optimal neighbor
     ```
     Pareto-optimal experiments (3 of 47):
     | Exp ID  | Accuracy | Latency (ms) | Params  | Notes             |
     |---------|----------|--------------|---------|-------------------|
     | exp-042 | 0.872    | 12.3         | 1.2M    | Best accuracy     |
     | exp-031 | 0.865    | 3.1          | 45K     | Best latency      |
     | exp-038 | 0.870    | 5.7          | 200K    | Best tradeoff     |
     ```
   - Writes to `experiments/frontiers/frontier-YYYY-MM-DD.md`
3. Extended metrics collection: update `log_experiment.py` to optionally capture inference latency and model size alongside the primary metric
4. Integration with `/turing:brief` — Pareto summary appears when multiple metrics are tracked
5. Add tests for Pareto computation, dominance checking

**Acceptance:** `/turing:frontier` shows which experiments are Pareto-optimal across tracked metrics. Researchers can identify the best tradeoff point for their deployment constraints.

---

## Phase 12: Performance & Resources (v1.5.0)

*Know where your time and memory go.*

### 12.1 Computational Profiling — `/turing:profile`

**What:** Measure FLOPS, memory high-water mark, throughput (samples/sec), and per-phase timing breakdown for a training run. Identifies bottlenecks (data loading? forward pass? gradient computation? I/O?) to answer "why is training slow?"

**Why:** Researchers waste hours waiting for training without knowing that 60% of the time is spent in data loading (fixable with caching) or that memory peaks during a single poorly-batched operation (fixable with gradient accumulation). Profiling makes the invisible visible.

**Implementation:**
1. Create `commands/profile.md` — `/turing:profile [exp-id]` (profiles the current best config)
2. Add `templates/scripts/profile_training.py`:
   - Wraps a single training run with instrumentation:
     - **Timing:** `time.perf_counter()` around data loading, preprocessing, model forward, loss computation, backward pass, optimizer step
     - **Memory:** `tracemalloc` for Python memory; `torch.cuda.max_memory_allocated()` for GPU if available; `/proc/self/status` VmPeak for system memory
     - **Throughput:** samples/sec and batches/sec
     - **I/O:** time spent reading from disk vs. compute
   - Produces a structured profile:
     ```yaml
     profile:
       total_time_sec: 142.3
       breakdown:
         data_loading: 85.2  # 59.9% — BOTTLENECK
         preprocessing: 12.1
         forward_pass: 22.4
         backward_pass: 18.7
         optimizer_step: 3.9
       memory:
         peak_rss_mb: 2048
         peak_gpu_mb: null  # No GPU detected
         model_size_mb: 12.3
       throughput:
         samples_per_sec: 1420
         batches_per_sec: 44.4
       bottleneck: "data_loading (59.9% of total time)"
       recommendations:
         - "Cache preprocessed data to disk — data loading is 60% of training time"
         - "Consider increasing batch size — GPU memory is underutilized"
     ```
   - Writes to `experiments/profiles/exp-NNN-profile.yaml`
3. Auto-recommendations engine: map bottleneck patterns to known fixes
4. Integration with `/turing:brief` — profile summary if available
5. Add tests for timing instrumentation, memory tracking

**Acceptance:** `/turing:profile` identifies the training bottleneck and suggests concrete fixes. A researcher can immediately see whether to optimize data loading, model architecture, or hardware.

### 12.2 Smart Checkpoint Manager — `/turing:checkpoint`

**What:** Manage model checkpoints based on Pareto dominance (not just "keep last K"). Supports checkpoint averaging, pruning, and resumption from any point in experiment history.

**Why:** Default checkpoint strategies waste disk space (keep everything) or lose good models (keep last K). Pareto-based pruning keeps only checkpoints that are best on at least one metric. Checkpoint averaging across the top-K epochs often outperforms any single checkpoint — free accuracy at no training cost.

**Implementation:**
1. Create `commands/checkpoint.md` — `/turing:checkpoint [list|prune|average|resume]`
2. Add `templates/scripts/checkpoint_manager.py`:
   - **list:** Scan `experiments/checkpoints/` and display with metrics, size, and Pareto status
   - **prune:** Remove checkpoints dominated on all tracked metrics. Keep Pareto-optimal + the latest checkpoint (for resume safety)
     ```
     Before: 47 checkpoints, 12.3 GB
     Pareto-optimal: 5 checkpoints
     Pruning 42 dominated checkpoints...
     After: 5 checkpoints, 1.4 GB (saved 10.9 GB)
     ```
   - **average:** Load top-K checkpoints (by primary metric), average their weights, evaluate the averaged model. Often yields 0.5-1% improvement for free
     ```
     Averaging top 3 checkpoints: exp-042 (0.872), exp-038 (0.870), exp-035 (0.869)
     Averaged model accuracy: 0.875 (+0.003 over best single)
     ```
   - **resume:** Given an experiment ID, restore the checkpoint and training state for continued training
3. Checkpoint metadata: extend `log_experiment.py` to record checkpoint path and size
4. Automatic pruning: optionally run after each experiment to prevent unbounded disk growth
5. Add tests for Pareto pruning, checkpoint averaging, resume logic

**Depends on:** Pareto computation from Phase 11.3 (shared logic)

**Acceptance:** `/turing:checkpoint prune` reclaims disk space by removing dominated checkpoints. `/turing:checkpoint average` produces a model that outperforms any single checkpoint.

---

## Phase 13: Deployment Bridge (v2.0.0)

*The v2.0 milestone. Turing crosses from experiment engine to production-ready pipeline — the first time it produces deployable artifacts, not just experiment logs. Get the model out of the loop and into production.*

### 13.1 Model Export — `/turing:export`

**What:** Export the best model to production-ready formats (ONNX, TorchScript, TFLite, scikit-learn joblib, XGBoost JSON). Run inference equivalence checks and benchmark latency on simulated hardware profiles.

**Why:** The gap between "experiment works" and "model is deployable" is where most ML projects stall. Format conversion, equivalence testing, and latency benchmarking are mechanical but error-prone. `/turing:export` automates the mechanical parts so the researcher can focus on whether the model is *worth* deploying.

**Implementation:**
1. Create `commands/export.md` — `/turing:export [exp-id] [--format onnx|torchscript|joblib|xgboost_json] [--target cpu|gpu|edge]`
2. Add `templates/scripts/export_model.py`:
   - Auto-detects model type from experiment config
   - Format-specific export:
     - **scikit-learn/XGBoost/LightGBM:** `joblib.dump()` + model-native format (XGBoost JSON, LightGBM text)
     - **PyTorch:** `torch.jit.trace()` for TorchScript, `torch.onnx.export()` for ONNX
     - **TensorFlow/Keras:** `tf.lite.TFLiteConverter` for TFLite, SavedModel for serving
   - Inference equivalence check:
     - Run 100 random samples through both original and exported model
     - Compare outputs within tolerance (1e-5 for float32, 1e-3 for quantized)
     - Report: `equivalent` / `approximately_equivalent` / `divergent` with max delta
   - Latency benchmark:
     - Warm-up: 10 inference calls (discard)
     - Benchmark: 100 calls, report p50/p95/p99 latency
     - Compare against original model latency
   - Model card generation:
     ```yaml
     model_card:
       name: "exp-042-xgboost-classifier"
       task: "binary_classification"
       metrics: {accuracy: 0.872, f1: 0.869}
       seed_study: {mean: 0.868, std: 0.007}
       export_format: "xgboost_json"
       equivalence: "exact (max_delta=0.0)"
       inference_latency:
         original_p50_ms: 1.2
         exported_p50_ms: 0.8
       size_mb: 2.3
       dependencies: ["xgboost>=1.7"]
       training_date: "2026-03-31"
       experiment_id: "exp-042"
     ```
   - Writes exported model + card to `exports/exp-NNN/`
3. Integration with seed study (Phase 10.1) — model card includes seed study results if available
4. Add `--quantize` flag for INT8 quantization (where supported) with accuracy-loss check
5. Add tests for export, equivalence checking, latency benchmarking

**Acceptance:** `/turing:export` produces a deployable model artifact with equivalence verification, latency benchmarks, and a model card. Zero manual conversion steps.

---

## Phase 14: Research Workflow (v2.1.0)

*Close the loop from experiment to publication.*

### 14.1 Literature Integration — `/turing:lit`

**What:** Targeted literature search scoped to the current experiment's domain. Find relevant papers, baselines to compare against, and methodological precedent — without leaving the experiment loop.

**Why:** Phase 7 added literature-grounded model selection and experiment design. `/turing:lit` is the human-facing complement: a researcher mid-experiment who wants to know "has anyone tried this approach on a similar dataset?" or "what's the SOTA baseline I should be comparing against?" Currently this requires context-switching to Google Scholar. `/turing:lit` brings the search into the experiment workflow.

**Implementation:**
1. Create `commands/lit.md` — `/turing:lit <query>` or `/turing:lit --baseline` or `/turing:lit --related <exp-id>`
2. Add `templates/scripts/literature_search.py`:
   - **Free query mode:** `/turing:lit "gradient boosting for tabular data with missing values"`
     - Searches scholarly APIs (arXiv, Semantic Scholar, Papers With Code — from Appendix A's catalog)
     - Returns top 5 papers with: title, authors, year, venue, abstract snippet, citation count, relevance score
   - **Baseline mode:** `/turing:lit --baseline`
     - Reads task description from `config.yaml`
     - Searches Papers With Code for SOTA results on similar tasks/datasets
     - Returns leaderboard entries with: method, metric, paper, code availability
     - Compares against current best: "Your accuracy (0.872) vs. SOTA (0.891) — gap: 0.019"
   - **Related mode:** `/turing:lit --related exp-042`
     - Reads experiment description and config
     - Finds papers using similar methods/architectures
     - Highlights methodological differences worth trying
   - Writes results to `experiments/literature/query-YYYY-MM-DD-HHMMSS.md`
   - Optionally auto-queues hypotheses from literature findings with `source: "literature"`
3. Reuses the arXiv/Semantic Scholar pipeline from Phase 7.1 (shared library)
4. Integration with `/turing:brief --deep` (Phase 7.3) — literature results feed into deep briefings
5. Add tests

**Depends on:** Scholarly API infrastructure from Phase 7.1

**Acceptance:** `/turing:lit --baseline` shows SOTA comparison for the current task. `/turing:lit "query"` returns relevant papers without leaving the terminal.

### 14.2 Paper Section Drafting — `/turing:paper`

**What:** Draft the mechanical sections of an ML paper (methodology, results tables, ablation tables, experimental setup) directly from experiment logs. Not a full paper — just the parts that are tedious to write and easy to get wrong.

**Why:** Writing "we trained XGBoost with max_depth=6, learning_rate=0.1, n_estimators=500 on an 80/20 train/test split with 5-fold cross-validation" is mechanical transcription from experiment logs to LaTeX. Getting a number wrong in a results table is embarrassingly common. `/turing:paper` eliminates transcription errors and saves hours of formatting.

**Implementation:**
1. Create `commands/paper.md` — `/turing:paper [--sections setup,results,ablation] [--format latex|markdown]`
2. Add `templates/scripts/draft_paper_sections.py`:
   - **Experimental Setup section:**
     - Reads `config.yaml` for task description, dataset info, evaluation protocol
     - Reads best experiment config for hyperparameters
     - Reads seed study (Phase 10.1) for statistical methodology
     - Generates prose: "We evaluate on [dataset] using [metric] with [N]-fold cross-validation. Results are reported as mean ± standard deviation over [N] random seeds."
   - **Results Table:**
     - Reads experiment families (Phase 6.3) or top-K experiments
     - Generates formatted table with model name, all tracked metrics, seed study stats
     - Highlights best result per metric (bold)
     - Includes statistical significance indicators (from Phase 2.1)
     ```latex
     \begin{table}[h]
     \centering
     \caption{Comparison of model architectures on [dataset].}
     \begin{tabular}{lcc}
     \toprule
     Model & Accuracy & F1 \\
     \midrule
     XGBoost & \textbf{0.872 ± 0.007} & 0.869 ± 0.008 \\
     LightGBM & 0.865 ± 0.005 & \textbf{0.871 ± 0.006} \\
     Random Forest & 0.843 ± 0.012 & 0.839 ± 0.014 \\
     \bottomrule
     \end{tabular}
     \end{table}
     ```
   - **Ablation Table:**
     - Reads ablation study (Phase 11.2) if available
     - Formats as publication-ready table
   - **Hyperparameter Table:**
     - Extracts final hyperparameters for all reported models
     - Formats as appendix-style table
   - Writes to `paper/sections/` directory with one file per section
3. Add `--bib` flag: generates BibTeX entries for papers cited via `/turing:lit`
4. Add `--overleaf` flag: formats for direct paste into Overleaf
5. Add tests for table generation, number accuracy (critical — verify against log.jsonl)

**Depends on:** Seed runner (10.1), ablation studies (11.2), literature integration (14.1), experiment families (6.3)

**Acceptance:** `/turing:paper` produces LaTeX sections with correct numbers pulled directly from experiment logs. No manual transcription needed.

---

## Phase 15: Orchestration (v2.2.0)

*Control how experiments run, not just what they run.*

### 15.1 Experiment Scheduler — `/turing:queue`

**What:** Queue multiple experiments for batch execution with priority ordering, dependency chains, and resource-aware scheduling. The researcher loads the queue Friday afternoon, reads `/turing:brief` Monday morning.

**Why:** Currently `/turing:train` runs one iterative loop. The researcher must babysit it or use `/loop`. A proper queue decouples experiment *planning* from experiment *execution* — the human front-loads taste (which hypotheses, in what order, with what constraints), and the system executes overnight with no supervision.

**Implementation:**
1. Create `commands/queue.md` — `/turing:queue [add|list|run|pause|clear]`
2. Add `templates/scripts/experiment_queue.py`:
   - **add:** `/turing:queue add "try LightGBM" --priority high --after hyp-003` — adds to queue with optional dependency
   - **list:** shows queue with status, priority, estimated runtime, dependencies
   - **run:** executes the queue sequentially (respecting dependencies and priorities):
     - For each queued item: edit `train.py` per hypothesis → run → evaluate → log → decision packet → next
     - Pause on crash if `--halt-on-error`; otherwise `retry` (Phase 15.2) and continue
     - Write a batch summary when queue drains
   - **pause:** save queue state and stop after current experiment finishes
   - **clear:** discard all queued items
   - Queue persists in `experiments/queue.yaml`
3. Integration with hypothesis queue (`hypotheses.yaml`) — `/turing:queue add` can pull directly from queued hypotheses
4. Integration with `/turing:brief` — batch summary appears as a "Queue Report" section
5. Resource awareness: estimate runtime per experiment from `/turing:profile` data (Phase 12.1) if available
6. Add tests

**Acceptance:** Researcher queues 10 experiments, runs `/turing:queue run`, walks away. All 10 execute in priority/dependency order with a summary report at the end.

### 15.2 Smart Failure Recovery — `/turing:retry`

**What:** When an experiment crashes (OOM, NaN loss, timeout, import error), auto-diagnose the failure mode and retry with a targeted fix. Turns crashes from dead ends into one-step recoveries.

**Why:** The most common experiment failures have mechanical fixes: OOM → reduce batch size, NaN → add gradient clipping, timeout → increase patience. Currently the researcher reads the traceback, edits config, and re-runs manually. `/turing:retry` automates the diagnosis-fix-rerun cycle for known failure patterns.

**Implementation:**
1. Create `commands/retry.md` — `/turing:retry [exp-id] [--max-attempts 3]`
2. Add `templates/scripts/smart_retry.py`:
   - Reads the experiment's `run.log` and exit code
   - Classifies failure against a taxonomy:
     ```yaml
     failure_modes:
       oom:
         patterns: ["CUDA out of memory", "MemoryError", "RuntimeError: out of memory"]
         fix: "Reduce batch_size by 50%"
         config_change: {batch_size: "//2"}
       nan_loss:
         patterns: ["loss is NaN", "nan", "RuntimeWarning: invalid value"]
         fix: "Add gradient clipping at 1.0, reduce learning_rate by 10x"
         config_change: {gradient_clip: 1.0, learning_rate: "//10"}
       timeout:
         patterns: ["TimeoutError", "exceeded time limit"]
         fix: "Increase max_epochs or reduce early_stopping patience"
         config_change: {max_epochs: "*2"}
       import_error:
         patterns: ["ModuleNotFoundError", "ImportError"]
         fix: "Install missing dependency"
         action: "pip_install"
       convergence_failure:
         patterns: ["loss did not decrease", "no improvement"]
         fix: "Increase learning_rate by 3x for warm-up, add learning rate scheduler"
         config_change: {learning_rate: "*3"}
     ```
   - Applies the fix, re-runs, and logs the retry as a child experiment of the original
   - Tracks retry attempts to prevent infinite loops (max 3 by default)
   - If all retries fail, creates a decision packet with `investigate_crash` verdict
3. Configurable taxonomy in `config/failure_modes.yaml` — researchers can add project-specific patterns
4. Integration with `/turing:queue` — crashed experiments auto-retry before moving to the next queue item
5. Add tests

**Depends on:** Phase 11.1 (diagnose) for failure pattern infrastructure

**Acceptance:** An OOM crash triggers automatic batch size reduction and re-run. The researcher sees "exp-051 crashed (OOM), retried as exp-052 with batch_size=16 → succeeded" in the log.

### 15.3 Experiment Branching — `/turing:fork`

**What:** Branch an experiment into parallel tracks. "Try both approach A and approach B from this point" — creates two child experiments from the same parent, runs both, and reports which branch wins.

**Why:** Researchers constantly face "should I try A or B?" decisions. Currently they pick one, run it, then try the other — sequential and slow. `/turing:fork` runs both in parallel (or queued back-to-back) and lets the results decide.

**Implementation:**
1. Create `commands/fork.md` — `/turing:fork <exp-id> --branches "LightGBM with dart" "XGBoost with deeper trees"`
2. Add `templates/scripts/fork_experiment.py`:
   - Reads the parent experiment's config
   - For each branch: creates a child hypothesis, applies the described modification, queues for execution
   - Branches share the same parent in the dependency tree (Phase 1.3)
   - After all branches complete, generates a comparison report:
     ```
     Fork from exp-042:
     ├── exp-053: LightGBM with dart → accuracy=0.878 ✓ WINNER
     └── exp-054: XGBoost deeper trees → accuracy=0.869
     Recommendation: promote exp-053, abandon exp-054
     ```
   - Auto-generates decision packets for each branch
3. Integration with `/turing:queue` — branches are queued as a dependency group (all must complete before the comparison fires)
4. Integration with experiment families (Phase 6.3) — forked branches share a family tag
5. Add `--auto-promote` flag: automatically keeps the best branch and discards the rest
6. Add tests

**Depends on:** Phase 1.3 (experiment dependency graph), Phase 15.1 (queue for parallel execution)

**Acceptance:** `/turing:fork exp-042 --branches "A" "B"` queues two experiments, runs both, and reports the winner with a recommendation.

---

## Phase 16: Deep Analysis (v2.3.0)

*See what your experiments are actually doing, not just their final numbers.*

### 16.1 Deep Experiment Comparison — `/turing:diff`

**What:** Side-by-side diff of two experiments showing config differences, metric deltas, per-class performance differences, training curve divergence points, and feature importance shifts. Goes beyond "which metric is higher" to answer "at what point did these two experiments diverge and why?"

**Why:** `/turing:compare` (existing) shows metric tables. `/turing:diff` is a diagnostic tool — when two experiments have similar aggregate metrics but feel different, it finds where and why they diverge. Essential for understanding whether a change actually helped or just shifted errors around.

**Implementation:**
1. Create `commands/diff.md` — `/turing:diff <exp-a> <exp-b>`
2. Add `templates/scripts/experiment_diff.py`:
   - **Config diff:** which hyperparameters changed, with magnitude:
     ```
     max_depth:      6 → 8      (+33%)
     learning_rate:  0.1 → 0.1  (unchanged)
     n_estimators:   500 → 300  (-40%)
     ```
   - **Metric diff:** all tracked metrics with deltas and significance (reuse Phase 2.1):
     ```
     accuracy:  0.872 → 0.878  (+0.006, p=0.03 significant)
     f1:        0.869 → 0.871  (+0.002, p=0.41 not significant)
     ```
   - **Per-class diff:** which classes improved and which regressed (reuse Phase 3.1):
     ```
     class_0: precision 0.91 → 0.93  (+0.02)
     class_1: precision 0.83 → 0.79  (-0.04) ← REGRESSION
     ```
   - **Training curve divergence:** if epoch-level metrics are logged, find the epoch where curves separate
   - **Feature importance diff:** if available, show which features gained/lost importance
   - Writes to `experiments/diffs/exp-A-vs-B.md`
3. Integration with `/turing:brief` — diff between current best and previous best appears automatically
4. Add `--code` flag: includes the `git diff` of `train.py` between the two experiments
5. Add tests

**Depends on:** Phase 11.3 (frontier for multi-metric comparison), Phase 3.1 (per-class metrics)

**Acceptance:** `/turing:diff exp-042 exp-053` shows exactly which config changes caused which metric shifts, including per-class regressions hidden by aggregate improvement.

### 16.2 Live Training Monitor — `/turing:watch`

**What:** Stream metrics during a training run with early-warning alerts: loss spikes, gradient explosion, learning rate too aggressive, train/val gap widening mid-epoch. Catches problems 10 minutes into a 2-hour run instead of at the end.

**Why:** Long training runs fail silently. Loss goes to NaN at epoch 47 of 100, but the signs were visible at epoch 5. `/turing:watch` surfaces those signals in real-time so the researcher can intervene early — or the system can auto-pause and suggest fixes (integrating with `/turing:retry`).

**Implementation:**
1. Create `commands/watch.md` — `/turing:watch [--alerts] [--interval 10s]`
2. Add `templates/scripts/training_monitor.py`:
   - Tails `run.log` during an active training run
   - Parses epoch-level metrics as they're written
   - Computes rolling statistics: loss trend, train/val gap, metric velocity
   - Alert rules (configurable in `config/watch_alerts.yaml`):
     ```yaml
     alerts:
       loss_spike:
         condition: "loss > 3 * rolling_mean_loss"
         severity: warning
         message: "Loss spike at epoch {epoch}: {loss} vs rolling mean {mean}"
       nan_detected:
         condition: "loss == NaN"
         severity: critical
         action: pause
       overfitting_onset:
         condition: "train_loss < 0.5 * val_loss for 3 consecutive epochs"
         severity: warning
         message: "Overfitting detected — train/val gap widening since epoch {onset}"
       plateau:
         condition: "val_metric improvement < 0.001 for 5 epochs"
         severity: info
         message: "Metric plateaued — consider early stopping or learning rate reduction"
     ```
   - On critical alert with `action: pause`: saves checkpoint, pauses training, notifies researcher
   - Displays a compact live dashboard:
     ```
     Epoch 23/100 | loss: 0.342 ↓ | acc: 0.865 ↑ | gap: 0.018 | ⚠ plateau (5 epochs)
     ```
3. Integration with `/turing:retry` — if auto-paused, suggest a fix
4. Checkpoint save on alert: never lose progress when a problem is detected
5. Add tests for alert rule evaluation, metric parsing

**Depends on:** Phase 12.1 (profiling infrastructure for metric instrumentation)

**Acceptance:** `/turing:watch` catches a loss spike at epoch 12 and alerts the researcher before the run wastes 88 more epochs.

### 16.3 Performance Regression Gate — `/turing:regress`

**What:** After any code or dependency change, automatically re-run the best experiment and verify metrics haven't degraded. CI for your model — catch silent regressions from library upgrades, data pipeline changes, or accidental `train.py` edits.

**Why:** ML projects have a unique fragility: a scikit-learn patch, a pandas dtype change, or an accidental data preprocessing edit can silently shift results by 2-3%. Nobody notices until the next paper revision. `/turing:regress` makes metric stability a verifiable property.

**Implementation:**
1. Create `commands/regress.md` — `/turing:regress [--tolerance 0.01] [--against exp-id]`
2. Add `templates/scripts/regression_gate.py`:
   - Identifies the current best experiment from `experiment_state.yaml`
   - Re-runs it with identical config (reuses `/turing:reproduce` infrastructure from Phase 10.2)
   - Compares against the stored metrics:
     - **Pass:** all metrics within tolerance → "No regression detected"
     - **Warning:** some metrics degraded within 2x tolerance → "Minor regression, investigate"
     - **Fail:** any metric degraded beyond tolerance → "REGRESSION DETECTED" with diff
   - On failure, captures environment diff: `pip freeze` comparison, git diff, data hash comparison
   - Writes verdict to `experiments/regressions/check-YYYY-MM-DD.yaml`
3. Integration with git hooks (optional): run `/turing:regress` before committing changes to `train.py` or `prepare.py`
4. Integration with `/turing:brief` — regression check history appears as a "Stability" section
5. Add `--quick` flag: runs with 1 seed instead of full seed study for fast checks
6. Add tests

**Depends on:** Phase 10.2 (reproduce infrastructure)

**Acceptance:** After a library upgrade, `/turing:regress` detects a 1.5% accuracy drop and reports the environment diff showing which package version changed.

---

## Phase 17: Model Composition (v2.4.0)

*Combine what you've already trained into something better.*

### 17.1 Automated Ensemble Construction — `/turing:ensemble`

**What:** Build ensembles from the top-K experiments automatically. Tries voting, stacking, and blending. Often yields 1-3% improvement from models you've already trained — zero additional training cost.

**Why:** Ensembling is the most reliably effective technique in applied ML (virtually every Kaggle winner uses it). But manually combining models is tedious: load each model, align predictions, try different combination strategies, evaluate. `/turing:ensemble` automates the mechanical work.

**Implementation:**
1. Create `commands/ensemble.md` — `/turing:ensemble [--top-k 5] [--methods voting,stacking,blending]`
2. Add `templates/scripts/build_ensemble.py`:
   - Selects top-K models by primary metric (or Pareto-optimal set from Phase 11.3)
   - Filters for diversity: models with similar predictions add no value. Use prediction correlation to select diverse members.
   - Tries ensemble methods:
     - **Voting:** majority vote (classification) or mean (regression)
     - **Weighted voting:** weights proportional to individual model performance
     - **Stacking:** train a meta-learner (logistic regression / ridge) on out-of-fold predictions
     - **Blending:** holdout-based alternative to stacking (simpler, less data-efficient)
   - Evaluates each ensemble with full seed study (Phase 10.1)
   - Reports:
     ```
     Ensemble results (from 5 base models):
     | Method           | Accuracy | Δ vs Best Single |
     |------------------|----------|------------------|
     | Best single      | 0.872    | —                |
     | Voting (uniform) | 0.879    | +0.007           |
     | Voting (weighted)| 0.881    | +0.009           |
     | Stacking (LR)    | 0.884    | +0.012 ← BEST   |
     | Blending         | 0.882    | +0.010           |
     ```
   - Logs the best ensemble as a new experiment with `family: "ensemble"`
3. Integration with `/turing:export` (Phase 13.1) — export the ensemble as a single deployable artifact
4. Diversity analysis: report correlation matrix of base model predictions
5. Add tests

**Depends on:** Phase 11.3 (Pareto for model selection), Phase 10.1 (seed study for evaluation)

**Acceptance:** `/turing:ensemble` combines 5 trained models into a stacking ensemble that beats the best individual model, with zero additional training.

### 17.2 Pipeline Composition — `/turing:stitch`

**What:** Define preprocessing, feature engineering, model, and postprocessing as swappable stages. Independently test any stage without rewriting `train.py`. "Keep the feature pipeline from exp-031 but swap the model from exp-042."

**Why:** ML pipelines are monolithic — everything lives in one `train.py`. Changing the model means also re-running preprocessing. Changing features means also re-training the model. `/turing:stitch` decomposes the pipeline into stages so each can be independently varied, cached, and reused.

**Implementation:**
1. Create `commands/stitch.md` — `/turing:stitch [show|swap|cache|run]`
2. Add `templates/scripts/pipeline_manager.py`:
   - **show:** parse `train.py` and display the pipeline as stages:
     ```
     Pipeline stages:
     1. preprocess  → StandardScaler, handle_missing   (hash: a3b2c1)
     2. features    → polynomial_features, log_transform (hash: d4e5f6)
     3. model       → XGBoostClassifier(max_depth=6)     (hash: g7h8i9)
     4. postprocess → calibration, threshold_tuning       (hash: j0k1l2)
     ```
   - **swap:** replace one stage with a version from another experiment:
     `/turing:stitch swap model --from exp-031` — takes exp-031's model config, keeps current preprocessing and features
   - **cache:** save intermediate outputs (preprocessed data, engineered features) to disk. Subsequent experiments that only change the model skip preprocessing entirely.
   - **run:** execute the stitched pipeline and log as a new experiment
   - Stage hashing: detect when a stage hasn't changed and skip re-computation
3. Stage definition in `config.yaml`:
   ```yaml
   pipeline:
     stages: [preprocess, features, model, postprocess]
     cache_dir: experiments/cache/
   ```
4. Integration with `/turing:ablate` (Phase 11.2) — ablate a stage by replacing it with a no-op
5. Add tests

**Depends on:** Phase 11.2 (ablation for stage-level testing)

**Acceptance:** `/turing:stitch swap model --from exp-031` creates a hybrid experiment in under 30 seconds (skipping cached preprocessing).

### 17.3 Warm-Start from Prior Model — `/turing:warm`

**What:** Take a trained checkpoint and use it as initialization for a different configuration. Automates the "start from here but change X" pattern.

**Why:** Transfer learning and fine-tuning are standard practice but operationally clumsy: find the checkpoint, load it, modify the architecture to accept the weights, freeze/unfreeze layers, adjust the optimizer. `/turing:warm` handles the plumbing so the researcher can focus on what to change.

**Implementation:**
1. Create `commands/warm.md` — `/turing:warm <exp-id> [--freeze-layers "encoder"] [--unfreeze-after 5]`
2. Add `templates/scripts/warm_start.py`:
   - Loads the checkpoint from the specified experiment (via Phase 12.2 checkpoint manager)
   - Detects model type and applies appropriate warm-start strategy:
     - **Tree models (XGBoost/LightGBM):** continue boosting from existing trees with modified hyperparameters
     - **Neural networks:** load weights, optionally freeze layers, reset optimizer state
     - **scikit-learn:** use `warm_start=True` parameter where supported
   - Creates a modified `train.py` with warm-start initialization
   - Logs the new experiment as a child of the source experiment
   - Reports: "Warm-started from exp-042 (epoch 50). Frozen: encoder layers. Training: decoder + head."
3. Layer freezing schedule: `--unfreeze-after N` unfreezes all layers after N epochs (gradual unfreezing)
4. Learning rate adjustment: automatically reduces learning rate for warm-started training (fine-tuning convention)
5. Add tests

**Depends on:** Phase 12.2 (checkpoint manager for loading)

**Acceptance:** `/turing:warm exp-042 --freeze-layers encoder` creates a fine-tuning experiment that starts from exp-042's weights with the encoder frozen.

---

## Phase 18: Scaling & Efficiency (v2.5.0)

*Do more with less. Know when to stop.*

### 18.1 Scaling Law Estimator — `/turing:scale`

**What:** Run 3-4 small experiments at different data/compute sizes, fit a power-law curve, and predict what performance you'd get at full scale. Answers "is it worth training on the full dataset?" before you commit the compute.

**Why:** Scaling laws (Kaplan et al., Hoffmann et al.) show that ML performance follows predictable power-law relationships with data size, model size, and compute. Researchers waste days training on full datasets when a 30-minute scaling study would show the expected gain is 0.3%. `/turing:scale` makes this estimation trivial.

**Implementation:**
1. Create `commands/scale.md` — `/turing:scale [--axis data|compute|params] [--points 4]`
2. Add `templates/scripts/scaling_estimator.py`:
   - Takes the current best experiment config
   - Generates scaled-down versions along the chosen axis:
     - **data:** train on 10%, 25%, 50%, 75% of the dataset
     - **compute:** train for 10%, 25%, 50%, 75% of max epochs
     - **params:** scale model size (reduce layers/width/depth/estimators)
   - Runs each scaled experiment (with seed study from Phase 10.1 for error bars)
   - Fits a power law: `performance = a * scale^b + c`
   - Extrapolates to full scale and beyond:
     ```
     Scaling analysis (data axis):
     | Data % | Accuracy (mean±std) |
     |--------|---------------------|
     | 10%    | 0.821 ± 0.012       |
     | 25%    | 0.847 ± 0.008       |
     | 50%    | 0.862 ± 0.006       |
     | 75%    | 0.869 ± 0.005       |
     
     Power law fit: acc = 0.723 * n^0.089 + 0.142 (R²=0.997)
     
     Predictions:
       100% data → 0.874 ± 0.004 (expected gain from 75%: +0.005)
       200% data → 0.882 ± 0.003 (hypothetical, requires more data)
     
     Verdict: Diminishing returns — 100% data gains only +0.005 over 75%.
     Consider investing in feature engineering instead of more data.
     ```
   - Writes to `experiments/scaling/scale-YYYY-MM-DD.md`
3. Integration with `/turing:budget` (Phase 18.2) — scaling predictions inform budget allocation
4. Add `--plot` flag: ASCII plot of the scaling curve with prediction bands
5. Add tests for power law fitting, extrapolation, confidence intervals

**Depends on:** Phase 10.1 (seed runner for error bars at each scale point)

**Acceptance:** `/turing:scale --axis data` runs 4 scaled experiments, fits a power law, and correctly predicts full-scale accuracy within 2% margin.

### 18.2 Compute Budget Manager — `/turing:budget`

**What:** Set a total compute budget (hours, experiment count, or estimated cost), and the system allocates across exploration vs. exploitation. Automatically shifts to exploit mode when budget runs low. Prevents runaway compute spend.

**Why:** Autonomous experiment loops are dangerous without a budget. `/turing:train` with no constraints will run indefinitely, exploring dead ends. `/turing:budget` gives the researcher a spend ceiling and lets the system optimize within it — exploring broadly early, exploiting the best direction late.

**Implementation:**
1. Create `commands/budget.md` — `/turing:budget [set|status|reset]`
2. Add `templates/scripts/budget_manager.py`:
   - **set:** `/turing:budget set --experiments 50 --hours 8` — set budget constraints
   - **status:** show remaining budget, burn rate, and projected exhaustion:
     ```
     Budget status:
       Experiments: 23/50 used (46%), 27 remaining
       Time: 3.2/8.0 hours used (40%), 4.8h remaining
       Burn rate: 7.2 experiments/hour
       Projected: budget exhausts in ~3.75 hours
       
       Allocation:
         Explore: 15 experiments (65%) — 8 remaining
         Exploit: 8 experiments (35%) — 19 remaining
       
       Auto-mode shift: switching to exploit at 80% budget (exp 40)
     ```
   - **Budget allocation policy:**
     - 0-50% budget: explore mode (try diverse hypotheses)
     - 50-80% budget: mixed (explore promising, exploit best)
     - 80-100% budget: exploit only (refine the winner)
   - Integrates with research mode (Phase 6.5) — auto-switches mode based on budget phase
   - Hard stop: at 100% budget, `/turing:train` refuses to start new experiments
   - Budget stored in `experiment_state.yaml` under `budget` key
3. Integration with `/turing:queue` (Phase 15.1) — queue respects budget limits
4. Integration with `/turing:scale` (Phase 18.1) — scaling predictions inform whether remaining budget is worth spending
5. Cost estimation: if `/turing:profile` data exists, estimate wall-clock cost per experiment
6. Add tests

**Depends on:** Phase 15.1 (queue for budget-aware scheduling), Phase 6.5 (research mode for auto-switching)

**Acceptance:** With a 50-experiment budget, the system auto-shifts from explore to exploit at experiment 40 and refuses to run experiment 51.

### 18.3 Model Compression — `/turing:distill`

**What:** Take a large accurate model (teacher) and train a smaller model (student) to match its predictions. Measures the accuracy/size/latency tradeoff. Bridges the gap between "best research model" and "model that fits in production constraints."

**Why:** The best model from `/turing:train` is often too large or slow for production. Distillation is the standard technique to compress it, but it requires writing a custom training loop with soft labels, temperature scaling, and student architecture selection. `/turing:distill` automates this.

**Implementation:**
1. Create `commands/distill.md` — `/turing:distill <teacher-exp-id> [--compression 4x] [--method soft-labels|feature-matching]`
2. Add `templates/scripts/model_distiller.py`:
   - Loads the teacher model from the specified experiment
   - Auto-selects student architecture based on compression target:
     - **Tree models:** fewer estimators, shallower depth
     - **Neural networks:** fewer layers, narrower hidden dims, quantization-aware training
     - **scikit-learn:** simpler model family (e.g., teacher=RandomForest, student=DecisionTree)
   - Distillation methods:
     - **Soft labels:** train student on teacher's probability outputs (temperature-scaled)
     - **Feature matching:** align intermediate representations (neural nets only)
     - **Dataset distillation:** train student on teacher-labeled synthetic data
   - Evaluates the student with full metrics + speed comparison:
     ```
     Distillation results (4x compression):
     | Model    | Accuracy | Size (MB) | Latency (ms) |
     |----------|----------|-----------|---------------|
     | Teacher  | 0.884    | 48.2      | 12.3          |
     | Student  | 0.877    | 11.8      | 3.1           |
     | Δ        | -0.007   | -75%      | -75%          |
     
     Verdict: 0.7% accuracy loss for 4x compression. Acceptable for production.
     ```
   - Logs as a new experiment with `family: "distillation"` and parent link to teacher
3. Integration with `/turing:export` (Phase 13.1) — export the distilled model directly
4. Integration with `/turing:frontier` (Phase 11.3) — student appears on the Pareto frontier
5. Add `--target-latency` flag: auto-select compression ratio to meet a latency target
6. Add tests

**Depends on:** Phase 13.1 (export for size/latency comparison)

**Acceptance:** `/turing:distill exp-042 --compression 4x` produces a student model with <1% accuracy loss and 4x smaller size.

---

## Phase 19: Meta-Intelligence (v3.0.0)

*The v3.0 milestone. Turing becomes project-aware — learning across projects, not just within one. The system accumulates institutional ML knowledge.*

### 19.1 Cross-Project Knowledge Transfer — `/turing:transfer`

**What:** Scan prior Turing projects for similar task characteristics and surface what worked. "Last time you had a tabular classification with class imbalance, SMOTE + LightGBM beat everything else by 3%." Builds institutional memory across projects.

**Why:** ML researchers repeat the same discoveries across projects: "random forests work well on small tabular data," "batch normalization helps deep networks converge," "learning rate 3e-4 is a good default." This knowledge lives in the researcher's head and is lost when they leave. `/turing:transfer` makes it systematic and persistent.

**Implementation:**
1. Create `commands/transfer.md` — `/turing:transfer [--from project-path] [--auto]`
2. Add `templates/scripts/knowledge_transfer.py`:
   - Scans all Turing projects on the machine (searches for `config.yaml` + `experiments/log.jsonl` patterns)
   - For each project, extracts a project signature:
     - Task type (classification/regression/ranking)
     - Dataset characteristics (size, dimensionality, class balance, feature types)
     - Best model family and key hyperparameters
     - What worked (kept experiments) and what didn't (discarded)
   - Compares current project's signature against prior projects by similarity
   - Generates transfer recommendations:
     ```
     Similar prior projects found:
     
     1. ~/projects/fraud-detection/ (similarity: 0.87)
        Task: binary classification, tabular, imbalanced (5:1)
        Winner: LightGBM + SMOTE, accuracy=0.923
        Key insight: oversampling before CV caused leakage — use SMOTE inside CV folds
        Hypothesis: "Try LightGBM with scale_pos_weight instead of SMOTE"
     
     2. ~/projects/churn-prediction/ (similarity: 0.72)
        Task: binary classification, tabular, moderate imbalance (3:1)
        Winner: XGBoost + feature selection, accuracy=0.891
        Key insight: removing correlated features improved generalization by 2%
        Hypothesis: "Run feature selection before training"
     ```
   - Auto-queues hypotheses from transfer recommendations with `source: "transfer"`
3. Project index: maintains a lightweight index at `~/.turing/project_index.yaml` (cross-project, not per-project)
4. Privacy-aware: only indexes projects on the local machine, never uploads
5. Integration with `/turing:init` — suggest starting hypotheses from similar projects during scaffolding
6. Add tests

**Depends on:** Phase 9.1 (semantic index for similarity matching)

**Acceptance:** `/turing:transfer` finds a similar prior project and suggests a hypothesis that the researcher wouldn't have tried otherwise. The hypothesis proves useful.

### 19.2 Pre-Submission Methodology Audit — `/turing:audit`

**What:** Check for common ML paper methodology mistakes before submission: data leakage, wrong CV strategy, missing baselines, unreported hyperparameter tuning cost, cherry-picked seeds, train/test overlap. A reviewer checklist you run *before* submitting.

**Why:** The top reasons for ML paper desk rejections are methodological, not novelty: leakage, unfair comparisons, missing ablations, unreproducible results. These are all checkable from experiment logs. `/turing:audit` is a pre-flight check that catches these before a reviewer does.

**Implementation:**
1. Create `commands/audit.md` — `/turing:audit [--strict] [--checklist venue-name]`
2. Add `templates/scripts/methodology_audit.py`:
   - Reads the full experiment history and project configuration
   - Checks against a methodology checklist:
     ```yaml
     checks:
       data_leakage:
         description: "Test data not used during training or feature engineering"
         check: "Verify prepare.py splits before any feature computation"
         severity: critical
       
       cv_strategy:
         description: "CV strategy appropriate for data type"
         check: "Temporal data uses time-series split, grouped data uses group k-fold"
         severity: critical
       
       seed_sensitivity:
         description: "Results reported with error bars from multiple seeds"
         check: "Seed study exists for best experiment (Phase 10.1)"
         severity: high
       
       ablation_completeness:
         description: "All major components ablated"
         check: "Ablation study exists (Phase 11.2) covering all non-trivial components"
         severity: high
       
       baseline_comparison:
         description: "Compared against reasonable baselines"
         check: "At least one simple baseline (majority class, mean prediction) in experiment log"
         severity: high
       
       hyperparameter_budget:
         description: "Total hyperparameter tuning budget reported"
         check: "Experiment count and compute hours documented"
         severity: medium
       
       reproducibility:
         description: "Best result successfully reproduced"
         check: "Reproduction report exists (Phase 10.2)"
         severity: high
       
       train_test_overlap:
         description: "No overlap between train and test samples"
         check: "Hash-based deduplication check on prepare.py output"
         severity: critical
     ```
   - Produces an audit report:
     ```
     Methodology Audit Report
     ========================
     ✓ PASS  Data leakage: prepare.py splits before feature computation
     ✓ PASS  Seed sensitivity: 5-seed study, CV=0.82%
     ✗ FAIL  Baseline comparison: no simple baseline found in experiment log
     ⚠ WARN  Ablation: 3 of 5 components ablated, missing: augmentation, postprocessing
     ✓ PASS  Reproducibility: exp-042 reproduced within tolerance
     ⚠ WARN  Hyperparameter budget: 47 experiments run, not documented in paper sections
     
     Score: 4/7 pass, 2 warnings, 1 failure
     Action required: Add a simple baseline experiment before submission
     ```
   - Venue-specific checklists: `--checklist neurips` adds NeurIPS-specific checks (reproducibility checklist, broader impact statement)
3. Integration with `/turing:paper` (Phase 14.2) — audit failures generate TODO items in paper sections
4. Auto-fix suggestions: for each failure, suggest the `/turing:` command that would fix it
5. Add tests

**Depends on:** Phases 10.1 (seed), 11.2 (ablation), 10.2 (reproduce), 14.2 (paper)

**Acceptance:** `/turing:audit` catches a missing baseline comparison and a partial ablation study. The researcher fixes both before submission, avoiding a desk rejection.

---

## Phase 20: Pre-Training Intelligence (v3.1.0)

*Catch problems before you waste a single GPU cycle.*

### 20.1 Pre-Training Sanity Checks — `/turing:sanity`

**What:** Run a battery of fast sanity checks before committing to a full training run: Can the model overfit a single batch? Do gradients flow? Is the loss at initialization what theory predicts? Does a forward pass produce valid outputs?

**Why:** The most frustrating ML failure mode: train for 2 hours, get garbage results, realize there was a bug in data loading that was detectable in 30 seconds. `/turing:sanity` catches wiring bugs, shape mismatches, and configuration errors before they cost real compute.

**Implementation:**
1. Create `commands/sanity.md` — `/turing:sanity [--quick] [--verbose]`
2. Add `templates/scripts/sanity_checks.py`:
   - **Initial loss check:** compute loss on first batch, compare to theoretical expectation (e.g., `−log(1/num_classes)` for cross-entropy). Flag if >2x expected.
   - **Single-batch overfit:** train on one batch for 50 steps. If loss doesn't approach zero, the model can't even memorize — something is broken.
   - **Gradient flow:** check that gradients are non-zero and non-exploding for every parameter. Flag dead layers (zero gradient) and unstable layers (gradient > 100x mean).
   - **Output validation:** forward pass produces valid (non-NaN, non-constant) outputs with reasonable range.
   - **Data pipeline check:** first batch loads correctly, shapes match model expectations, no NaN/Inf in inputs.
   - **Config consistency:** learning rate and batch size are in reasonable ranges for model size.
   - Report:
     ```
     Sanity Check Report (14 seconds):
     ✓ PASS  Data pipeline: batch loads, shapes correct (X: [32,128], y: [32])
     ✓ PASS  Initial loss: 2.31 (expected: 2.30 for 10-class CE)
     ✓ PASS  Gradient flow: all 47 parameters have non-zero gradients
     ✗ FAIL  Single-batch overfit: loss stuck at 1.82 after 50 steps
              → Model cannot memorize 1 batch. Check: architecture, learning rate, loss function
     ⚠ WARN  Output range: predictions in [-12.4, 15.7], consider adding output clamping
     
     Verdict: 1 FAIL — do not proceed to full training
     ```
3. Integration with `/turing:train` — optionally auto-run sanity before first experiment
4. Add `--quick` flag: skip single-batch overfit (fastest, 5 seconds)
5. Add tests

**Acceptance:** `/turing:sanity` catches a broken data loader or misconfigured loss function in under 30 seconds, before the researcher commits to a full training run.

### 20.2 Automatic Baseline Generation — `/turing:baseline`

**What:** Auto-generate trivial baselines: majority class predictor, mean predictor, random predictor, linear model, k-NN. Every experiment needs a "is this better than dumb?" reference point.

**Why:** Reviewers always ask "how does this compare to a simple baseline?" and researchers always forget to include one. Worse, sometimes the fancy model barely beats a linear classifier — knowing this early changes the research direction entirely. `/turing:baseline` takes 60 seconds and saves weeks of misguided optimization.

**Implementation:**
1. Create `commands/baseline.md` — `/turing:baseline [--methods all|simple|linear]`
2. Add `templates/scripts/generate_baselines.py`:
   - Auto-detects task type from `config.yaml` (classification vs regression)
   - For classification:
     - **Random:** uniform random predictions
     - **Majority:** always predict the most common class
     - **Stratified:** predict class proportional to training distribution
     - **Linear:** `LogisticRegression(max_iter=1000)` with default params
     - **k-NN:** `KNeighborsClassifier(n_neighbors=5)` with default params
   - For regression:
     - **Mean:** always predict training set mean
     - **Median:** always predict training set median
     - **Linear:** `Ridge(alpha=1.0)` with default params
     - **k-NN:** `KNeighborsRegressor(n_neighbors=5)` with default params
   - Evaluates all baselines with the same `evaluate.py` protocol
   - Runs with seed study (Phase 10.1) for stochastic baselines
   - Produces comparison table:
     ```
     Baselines for binary_classification (accuracy):
     | Method          | Accuracy | Notes                    |
     |-----------------|----------|--------------------------|
     | Random          | 0.502    | Floor — below this = bug |
     | Majority class  | 0.627    | Naive floor              |
     | Logistic Reg.   | 0.814    | Linear ceiling           |
     | k-NN (k=5)      | 0.793    | Non-parametric reference  |
     | Current best    | 0.872    | +0.058 over linear       |
     
     Your model beats the linear baseline by 5.8%.
     ```
   - Logs each baseline as an experiment with `family: "baseline"`
3. Integration with `/turing:audit` (Phase 19.2) — satisfies the "baseline comparison" audit check
4. Integration with `/turing:paper` (Phase 14.2) — baseline rows auto-included in results tables
5. Add tests

**Depends on:** Phase 10.1 (seed runner), Phase 19.2 (audit integration)

**Acceptance:** `/turing:baseline` produces 4-5 trivial baselines in under 60 seconds. The researcher immediately knows if their model is meaningfully better than simple approaches.

### 20.3 Targeted Leakage Detection — `/turing:leak`

**What:** Actively probe for data leakage by training on single features, checking temporal splits for future information, detecting target encoding leakage, and flagging features that perform suspiciously well in isolation.

**Why:** Leakage is the #1 cause of "too good to be true" results and the #1 cause of ML paper retractions. It's also the hardest bug to catch because the model trains fine and metrics look great — until deployment. `/turing:leak` probes for specific leakage patterns that aggregate statistics can't detect.

**Implementation:**
1. Create `commands/leak.md` — `/turing:leak [--deep] [--features "feature_1,feature_2"]`
2. Add `templates/scripts/leakage_detector.py`:
   - **Single-feature test:** train a simple model on each feature individually. Flag any feature where single-feature accuracy > 80% of full-model accuracy (suspiciously predictive).
     ```
     Leakage scan (single-feature analysis):
     ⚠ FLAG  feature_12 alone achieves accuracy=0.91 (full model: 0.87)
              → This feature is MORE predictive alone than the full model.
              → Likely leakage. Investigate: is this derived from the target?
     ✓ OK    feature_3 alone: accuracy=0.63 (expected for informative feature)
     ✓ OK    feature_7 alone: accuracy=0.51 (near-random, weak feature)
     ```
   - **Temporal leakage:** if timestamps exist, check whether any feature contains future information relative to the prediction target.
   - **Target encoding leakage:** detect if categorical encoding was fit on the full dataset (including test) rather than train-only.
   - **Train/test overlap:** hash-based deduplication to find identical or near-identical samples across splits.
   - **Feature-target correlation:** flag features with Pearson/Spearman correlation > 0.95 with the target.
   - Writes to `experiments/leakage/leak-YYYY-MM-DD.md`
3. Add `--deep` flag: runs full single-feature analysis (slow but thorough)
4. Integration with `/turing:audit` (Phase 19.2) — satisfies the "data leakage" audit check
5. Add tests

**Depends on:** Phase 19.2 (audit integration)

**Acceptance:** `/turing:leak` detects a leaked feature that achieves 91% accuracy alone when the full model achieves 87%. The researcher removes it before publishing.

---

## Phase 21: Model Debugging (v3.2.0)

*Understand what the model is actually doing, not just what numbers come out.*

### 21.1 Internal Model Diagnostics — `/turing:xray`

**What:** Inspect model internals: gradient flow per layer, activation statistics, dead neurons, weight distributions, decision path analysis. Answers "what is the model doing internally?" rather than "what are its predictions?"

**Why:** When a model underperforms, the fix depends on *why*. Dead neurons → reinitialize. Vanishing gradients → skip connections or residual learning. Saturated activations → different normalization. Without `/turing:xray`, the researcher guesses. With it, the diagnosis is direct.

**Implementation:**
1. Create `commands/xray.md` — `/turing:xray [exp-id] [--layer "encoder.layer.2"]`
2. Add `templates/scripts/model_xray.py`:
   - Auto-detects model type and runs appropriate diagnostics:
   - **Neural networks:**
     - Layer-wise gradient magnitudes (mean, max, min per parameter group)
     - Activation statistics (mean, std, % zeros for each layer)
     - Dead neuron detection: neurons with zero activation across the full validation set
     - Weight distribution: mean, std, % near-zero per layer (pruning candidates)
     - Gradient-to-weight ratio: learning rate effectiveness per layer
     ```
     X-Ray: exp-042 (3-layer MLP)
     | Layer      | Grad Mean | Grad Max | Act Mean | Dead % | Weight Std |
     |------------|-----------|----------|----------|--------|------------|
     | linear_1   | 3.2e-03   | 1.1e-01  | 0.42     | 0%     | 0.31       |
     | linear_2   | 8.1e-05   | 2.3e-03  | 0.08     | 23%    | 0.28       |  ← ISSUE
     | linear_3   | 1.4e-04   | 5.6e-03  | 0.31     | 2%     | 0.15       |
     
     Issues detected:
     ⚠ linear_2: 23% dead neurons — consider reducing layer width or adding batch norm
     ⚠ linear_2: gradient 40x weaker than linear_1 — possible vanishing gradient
     ```
   - **Tree models (XGBoost/LightGBM):**
     - Tree depth utilization: are trees using their full allowed depth?
     - Leaf purity: how pure are the leaf nodes?
     - Feature split frequency: which features dominate the splits?
     - Decision path analysis: for misclassified samples, trace the decision path
   - **scikit-learn:**
     - Coefficient magnitudes (linear models)
     - Feature importance confidence intervals (ensemble models)
   - Writes to `experiments/xrays/exp-NNN-xray.md`
3. Integration with `/turing:diagnose` (Phase 11.1) — xray findings inform diagnosis
4. Add `--compare exp-a exp-b` flag: side-by-side xray of two models
5. Add tests

**Acceptance:** `/turing:xray` identifies 23% dead neurons in layer 2 and suggests batch normalization. The fix improves accuracy by 1.5%.

### 21.2 Hyperparameter Sensitivity Analysis — `/turing:sensitivity`

**What:** Vary each hyperparameter individually while holding others fixed, measure the metric response curve, and rank hyperparameters by sensitivity. Answers "which hyperparameters actually matter and which are noise?"

**Why:** Researchers waste hours tuning hyperparameters that have no effect. Learning rate sensitivity is 10x higher than max_depth sensitivity for this model — stop grid-searching max_depth and focus on learning rate. `/turing:sensitivity` produces a definitive ranking so tuning effort is allocated to where it matters.

**Implementation:**
1. Create `commands/sensitivity.md` — `/turing:sensitivity [exp-id] [--params "learning_rate,max_depth,n_estimators"]`
2. Add `templates/scripts/sensitivity_analysis.py`:
   - Takes the best experiment's config
   - For each hyperparameter, generates a sweep: 5 values spanning a reasonable range around the current value (e.g., 0.5x, 0.75x, 1x, 1.5x, 2x)
   - Runs each configuration with seed study (Phase 10.1) for error bars
   - Computes sensitivity = metric range / parameter range (normalized)
   - Produces sensitivity ranking:
     ```
     Hyperparameter Sensitivity Analysis (exp-042):
     | Parameter      | Current | Range Tested     | Metric Range | Sensitivity |
     |----------------|---------|-----------------|--------------|-------------|
     | learning_rate  | 0.1     | [0.01, 0.5]     | 0.831–0.872  | HIGH (0.041)|
     | n_estimators   | 500     | [100, 1000]      | 0.858–0.874  | MED (0.016) |
     | max_depth      | 6       | [3, 12]          | 0.866–0.873  | LOW (0.007) |
     | min_child_wt   | 1       | [1, 10]          | 0.869–0.872  | NONE (0.003)|
     
     Recommendation: Focus tuning on learning_rate and n_estimators.
     Stop tuning max_depth and min_child_weight — they don't matter.
     ```
   - Detects non-monotonic relationships (e.g., accuracy peaks at max_depth=8 then drops)
   - Detects interactions: if varying A changes B's sensitivity, flag the interaction
   - Writes to `experiments/sensitivity/exp-NNN-sensitivity.md`
3. Integration with `/turing:paper` (Phase 14.2) — sensitivity table for appendix
4. Integration with `/turing:brief` — sensitivity summary informs tuning recommendations
5. Add tests

**Depends on:** Phase 10.1 (seed runner for error bars)

**Acceptance:** `/turing:sensitivity` correctly identifies that learning_rate has 6x more impact than max_depth, redirecting the researcher's tuning effort.

### 21.3 Probability Calibration — `/turing:calibrate`

**What:** Measure whether model probabilities are well-calibrated (does 80% confidence mean 80% correct?), compute expected calibration error, plot reliability diagrams, and apply post-hoc calibration (Platt scaling, isotonic regression).

**Why:** Any model whose probability outputs drive decisions (medical diagnosis, fraud detection, risk scoring) must be calibrated. Most ML models are overconfident by default. `/turing:calibrate` measures the problem and fixes it with standard post-hoc techniques — often improving downstream decision quality without touching the model itself.

**Implementation:**
1. Create `commands/calibrate.md` — `/turing:calibrate [exp-id] [--method platt|isotonic|auto]`
2. Add `templates/scripts/calibration.py`:
   - Runs the model on validation set, collects predicted probabilities vs actual outcomes
   - **Reliability diagram:** bin predictions into 10 bins, compute accuracy per bin:
     ```
     Reliability Diagram (exp-042):
     Bin       | Predicted | Actual  | Gap
     [0.0-0.1] | 0.05      | 0.03    | -0.02  ✓
     [0.1-0.2] | 0.15      | 0.12    | -0.03  ✓
     ...
     [0.8-0.9] | 0.85      | 0.71    | -0.14  ⚠ overconfident
     [0.9-1.0] | 0.95      | 0.78    | -0.17  ⚠ overconfident
     
     Expected Calibration Error (ECE): 0.068
     Maximum Calibration Error (MCE): 0.170
     Verdict: Model is overconfident in high-probability predictions
     ```
   - **Post-hoc calibration:**
     - **Platt scaling:** fit a logistic regression on model logits → calibrated probabilities
     - **Isotonic regression:** non-parametric calibration (more flexible, needs more data)
     - **Temperature scaling:** single scalar temperature parameter (neural nets)
     - **auto:** tries all methods, picks the one with lowest ECE on a held-out calibration set
   - Reports calibration improvement:
     ```
     Calibration results:
     | Method    | ECE Before | ECE After | Accuracy Change |
     |-----------|------------|-----------|-----------------|
     | Platt     | 0.068      | 0.021     | 0.872 → 0.872   |  ← BEST
     | Isotonic  | 0.068      | 0.024     | 0.872 → 0.871   |
     | Temp (T=1.7) | 0.068   | 0.031     | 0.872 → 0.872   |
     ```
   - Saves calibrated model as a new experiment with `family: "calibration"`
3. Integration with `/turing:export` (Phase 13.1) — export calibrated model with calibration metadata in model card
4. Integration with `/turing:fairness` — calibration checked per demographic group
5. Add tests for ECE computation, calibration methods, reliability diagram

**Depends on:** Phase 13.1 (export for model card integration)

**Acceptance:** `/turing:calibrate` reduces ECE from 0.068 to 0.021 with Platt scaling, making the model's probability outputs trustworthy for downstream decision-making.

---

## Phase 22: Feature & Training Intelligence (v3.3.0)

*Smarter data handling and smarter training strategies.*

### 22.1 Automated Feature Selection — `/turing:feature`

**What:** Run multiple feature selection methods (mutual information, permutation importance, recursive elimination, L1 regularization), compute consensus, and optionally generate interaction/polynomial features from the consensus set.

**Why:** Feature engineering is the highest-ROI activity in applied ML — often more impactful than model selection. But it's tedious and researcher-dependent. `/turing:feature` systematically evaluates feature importance across multiple methods, identifies redundant features, and suggests new interaction features — turning a craft into a process.

**Implementation:**
1. Create `commands/feature.md` — `/turing:feature [--method all|importance|selection|generation] [--top-k 20]`
2. Add `templates/scripts/feature_intelligence.py`:
   - **Importance ranking** (run all, report consensus):
     - Mutual information (model-agnostic)
     - Permutation importance (model-specific, uses current best model)
     - L1 regularization coefficients (Lasso/LogisticRegression)
     - Tree-based importance (if applicable)
   - **Consensus ranking:** features ranked by number of methods that place them in top-K:
     ```
     Feature Importance Consensus (top 10):
     | Feature    | MI Rank | Perm Rank | L1 Rank | Tree Rank | Consensus |
     |------------|---------|-----------|---------|-----------|-----------|
     | feature_3  | 1       | 2         | 1       | 1         | 4/4 ★     |
     | feature_7  | 3       | 1         | 2       | 4         | 4/4 ★     |
     | feature_12 | 2       | 5         | 3       | 2         | 4/4 ★     |
     | feature_1  | 4       | 3         | 8       | 3         | 3/4       |
     ...
     | feature_22 | 18      | 21        | 19      | 17        | 0/4 — DROP |
     
     Recommendation: drop 8 features with 0/4 consensus (13% of features)
     ```
   - **Redundancy detection:** correlation matrix, flag feature pairs with |r| > 0.95
   - **Feature generation:** generate candidate interaction features (product, ratio, sum) from top-K consensus features, evaluate each for lift
   - Logs results and auto-queues a hypothesis: "Re-train with top-15 features only"
3. Integration with `/turing:ablate` (Phase 11.2) — feature ablation as a special case
4. Integration with `/turing:paper` (Phase 14.2) — feature importance table for appendix
5. Add tests

**Acceptance:** `/turing:feature` identifies 8 redundant features. Dropping them improves accuracy by 0.3% and reduces training time by 20%.

### 22.2 Training Curriculum Optimization — `/turing:curriculum`

**What:** Order training data by difficulty (easy-to-hard, hard-to-easy, anti-curriculum, or mixed strategies) and measure whether curriculum learning improves convergence speed or final performance.

**Why:** The order in which a model sees training data matters — curriculum learning can improve convergence speed by 2-3x and sometimes final performance by 1-2%. Particularly effective for imbalanced or noisy datasets. But finding the right curriculum is empirical — `/turing:curriculum` systematically tests strategies.

**Implementation:**
1. Create `commands/curriculum.md` — `/turing:curriculum [exp-id] [--strategies easy-to-hard,hard-to-easy,anti,random]`
2. Add `templates/scripts/curriculum_optimizer.py`:
   - **Difficulty scoring:** for each training sample, estimate difficulty:
     - **Loss-based:** samples with high loss on a pre-trained model are "hard"
     - **Margin-based:** samples close to the decision boundary are "hard"
     - **Noise-based:** samples that different seeds disagree on are "hard" (likely mislabeled)
   - **Curriculum strategies:**
     - **Easy-to-hard (classic curriculum):** sort by ascending difficulty, train in order
     - **Hard-to-easy (anti-curriculum):** sort by descending difficulty
     - **Pacing function:** start with easy 20%, gradually include harder samples over epochs
     - **Self-paced:** dynamically adjust which samples to include based on current loss
     - **Random baseline:** standard random shuffling (control)
   - Runs each strategy with seed study (Phase 10.1):
     ```
     Curriculum results (exp-042):
     | Strategy      | Final Acc | Convergence Epoch | Time to 0.85 |
     |---------------|-----------|-------------------|--------------|
     | Random         | 0.872     | 47                | 32 epochs    |
     | Easy-to-hard   | 0.878     | 38                | 24 epochs    | ← BEST
     | Hard-to-easy   | 0.869     | 52                | 41 epochs    |
     | Self-paced     | 0.876     | 41                | 28 epochs    |
     
     Verdict: Easy-to-hard curriculum converges 25% faster and improves final accuracy by 0.6%.
     ```
   - Identifies "impossible" samples: consistently high loss across all strategies (likely mislabeled)
   - Writes to `experiments/curriculum/exp-NNN-curriculum.md`
3. Integration with `/turing:clean` — flag impossible samples for review
4. Add tests

**Depends on:** Phase 10.1 (seed runner)

**Acceptance:** `/turing:curriculum` identifies that easy-to-hard training converges 25% faster. The researcher saves hours of training time on future experiments.

---

## Phase 23: Model Surgery (v3.4.0)

*Optimize the model you have without retraining from scratch.*

### 23.1 Weight Pruning — `/turing:prune`

**What:** Structured and unstructured weight pruning. Measures accuracy at different sparsity levels (50%, 75%, 90%, 95%), finds the knee point, and produces a pruned model. Complementary to distillation (Phase 18.3) — pruning preserves architecture, distillation changes it.

**Why:** Most neural network weights are redundant — you can remove 50-90% with minimal accuracy loss. Pruning gives faster inference and smaller models without changing the architecture. Combined with quantization (Phase 23.2), it's the fastest path from research model to edge deployment.

**Implementation:**
1. Create `commands/prune.md` — `/turing:prune <exp-id> [--sparsity 0.5,0.75,0.9,0.95] [--method magnitude|structured|lottery]`
2. Add `templates/scripts/model_pruning.py`:
   - **Magnitude pruning (unstructured):** zero out weights below a threshold, sorted by absolute value
   - **Structured pruning:** remove entire neurons/filters/attention heads based on importance scores
   - **Lottery ticket search:** iterative magnitude pruning with weight rewinding to find the "winning ticket" subnetwork
   - For tree models: reduce `n_estimators` progressively, measure impact per tree removed
   - Sparsity sweep:
     ```
     Pruning sweep (exp-042, magnitude pruning):
     | Sparsity | Accuracy | Δ from Dense | Speedup | Size Reduction |
     |----------|----------|--------------|---------|----------------|
     | 0%       | 0.872    | —            | 1.0x    | 0%             |
     | 50%      | 0.871    | -0.001       | 1.3x    | 50%            |
     | 75%      | 0.868    | -0.004       | 1.8x    | 75%            |
     | 90%      | 0.859    | -0.013       | 2.4x    | 90%            | ← knee point
     | 95%      | 0.831    | -0.041       | 2.7x    | 95%            |
     
     Recommended sparsity: 75% (0.4% accuracy loss for 1.8x speedup)
     ```
   - Optionally fine-tune pruned model for recovery (few epochs, reduced learning rate)
   - Logs as new experiment with `family: "pruning"` and parent link
3. Integration with `/turing:frontier` (Phase 11.3) — pruned models appear on Pareto frontier
4. Integration with `/turing:export` (Phase 13.1) — export pruned model in sparse format
5. Add tests

**Depends on:** Phase 11.3 (Pareto frontier), Phase 13.1 (export)

**Acceptance:** `/turing:prune` achieves 1.8x speedup at 75% sparsity with only 0.4% accuracy loss.

### 23.2 Post-Training Quantization — `/turing:quantize`

**What:** Quantize model weights from FP32 to INT8/FP16, measure accuracy loss per precision level. Apply quantization-aware training if post-training quantization degrades accuracy too much.

**Why:** Quantization is the lowest-effort production optimization: 2-4x speedup and 2-4x memory reduction with typically <0.5% accuracy loss. Every model heading to production should be quantized, but researchers rarely bother during experimentation. `/turing:quantize` makes it a one-command operation.

**Implementation:**
1. Create `commands/quantize.md` — `/turing:quantize <exp-id> [--precision int8|fp16|dynamic] [--aware]`
2. Add `templates/scripts/model_quantization.py`:
   - **Post-training quantization (PTQ):**
     - **Dynamic quantization:** quantize weights statically, activations dynamically (simplest, works for most models)
     - **Static quantization:** calibrate activation ranges on a representative dataset, then quantize (better accuracy, needs calibration data)
     - **FP16:** half-precision floating point (GPU inference)
   - **Quantization-aware training (QAT):** if PTQ accuracy loss > threshold, insert fake quantization nodes into the model and fine-tune for a few epochs
   - Framework-specific:
     - **PyTorch:** `torch.quantization` API
     - **scikit-learn/XGBoost:** weight precision reduction, feature importance-based rounding
   - Accuracy and latency comparison:
     ```
     Quantization results (exp-042):
     | Precision | Accuracy | Δ       | Latency (ms) | Size (MB) | 
     |-----------|----------|---------|--------------|-----------|
     | FP32      | 0.872    | —       | 12.3         | 48.2      |
     | FP16      | 0.872    | -0.000  | 7.1          | 24.1      |
     | INT8 (dyn)| 0.870    | -0.002  | 4.8          | 12.3      | ← BEST
     | INT8 (sta)| 0.871    | -0.001  | 4.6          | 12.1      |
     
     Recommended: INT8 dynamic (0.2% accuracy loss for 2.6x speedup)
     ```
   - Logs as new experiment with `family: "quantization"`
3. Integration with `/turing:prune` — combined pruning + quantization pipeline
4. Integration with `/turing:export` (Phase 13.1) — export quantized model
5. Add tests

**Depends on:** Phase 13.1 (export for deployment)

**Acceptance:** `/turing:quantize` achieves 2.6x speedup with INT8 dynamic quantization at 0.2% accuracy loss.

### 23.3 Model Merging — `/turing:merge`

**What:** Average or merge weights from multiple fine-tuned checkpoints into a single model (model soups, TIES merging, DARE). Often beats any individual model with zero additional training cost.

**Why:** Different from ensembling (Phase 17.1) — ensembles combine *predictions* at inference time (2-5x latency cost), merging combines *weights* into a single model (zero latency cost). Model soups (Wortsman et al., 2022) showed that averaging weights from models fine-tuned with different hyperparameters consistently outperforms individual models. It's free accuracy with no deployment complexity.

**Implementation:**
1. Create `commands/merge.md` — `/turing:merge <exp-ids...> [--method uniform|ties|dare|greedy]`
2. Add `templates/scripts/model_merger.py`:
   - **Uniform soup:** simple average of all model weights. Works when models share architecture and were fine-tuned from the same initialization.
   - **Greedy soup:** iteratively add models to the soup only if they improve the merged result. Filters out models that hurt the average.
   - **TIES merging:** Trim redundant parameters, Elect sign consensus, disjoint Merge. Better than uniform for models with conflicting parameter updates.
   - **DARE:** randomly Drop parameters And REscale survivors. Reduces interference between merged models.
   - For tree models: merge by averaging prediction probabilities at the leaf level, or combine tree sets with weight adjustment
   - Compatibility check: verify all models share the same architecture (weight shapes match)
   - Report:
     ```
     Model merge (3 models, uniform soup):
     | Model    | Individual Acc | Contribution |
     |----------|---------------|--------------|
     | exp-042  | 0.872         | included     |
     | exp-053  | 0.878         | included     |
     | exp-067  | 0.869         | included     |
     
     | Method       | Merged Acc | Δ vs Best Single |
     |--------------|-----------|------------------|
     | Uniform soup | 0.881     | +0.003           |
     | Greedy soup  | 0.883     | +0.005 ← BEST   |
     | TIES         | 0.880     | +0.002           |
     
     Greedy soup excluded 0 models. All 3 contribute positively.
     ```
   - Logs as new experiment with `family: "merge"` and parent links to all source models
3. Integration with `/turing:export` (Phase 13.1) — merged model is a single artifact, no ensemble overhead
4. Integration with `/turing:frontier` (Phase 11.3) — merged model has same latency as individual models but better accuracy
5. Add tests

**Depends on:** Phase 12.2 (checkpoint manager for loading), Phase 13.1 (export)

**Acceptance:** `/turing:merge` produces a single model that outperforms the best individual model, with identical inference latency.

### 23.4 Architecture Modification — `/turing:surgery`

**What:** Programmatic architecture changes: add/remove layers, widen/narrow, swap activation functions, inject skip connections, change normalization — then fine-tune briefly. Automates the "what if I tweaked the architecture?" experiments.

**Why:** Architecture modifications are the most error-prone manual edits in ML research. Adding a layer requires matching dimensions, updating optimizer parameter groups, and adjusting learning rate schedules. `/turing:surgery` handles the plumbing and produces a runnable modified `train.py`, so the researcher specifies *what* to change and the system handles *how*.

**Implementation:**
1. Create `commands/surgery.md` — `/turing:surgery <exp-id> [--op add-layer|remove-layer|widen|swap-activation|add-skip|add-norm]`
2. Add `templates/scripts/architecture_surgery.py`:
   - **Operations** (each produces a modified config and/or `train.py`):
     - `add-layer`: insert a layer at a specified position with auto-matched dimensions
     - `remove-layer`: remove a layer and reconnect surrounding layers
     - `widen <factor>`: multiply hidden dimensions by factor (e.g., 2x wider)
     - `narrow <factor>`: reduce hidden dimensions (e.g., 0.5x narrower)
     - `swap-activation <from> <to>`: replace ReLU→GELU, Sigmoid→SiLU, etc.
     - `add-skip`: inject residual connections between specified layers
     - `add-norm <type>`: insert BatchNorm/LayerNorm/GroupNorm at specified positions
   - For tree models:
     - `deepen`: increase max_depth
     - `widen`: increase n_estimators
     - `swap-objective`: change loss function (logloss→focal, mse→huber)
   - Auto warm-start: loads weights from source experiment where dimensions match, initializes new parameters (Phase 17.3 integration)
   - Parameter count comparison: report new vs old parameter count
   - Report:
     ```
     Surgery: add-layer (exp-042)
     Operation: Insert Linear(256→256) + ReLU after layer 2
     Parameters: 1.2M → 1.3M (+8.3%)
     Warm-started: 4/5 layers from exp-042, 1 layer initialized fresh
     Ready to train: experiments/surgery/exp-042-add-layer/train.py
     ```
   - Logs the modified experiment as a child of the source
3. Integration with `/turing:warm` (Phase 17.3) — surgery auto-warm-starts from source weights
4. Integration with `/turing:sensitivity` (Phase 21.2) — test which architectural changes matter
5. Add tests

**Depends on:** Phase 17.3 (warm-start), Phase 21.2 (sensitivity for testing changes)

**Acceptance:** `/turing:surgery exp-042 --op widen 2` doubles the hidden dimensions, warm-starts from existing weights, and produces a runnable experiment in under 10 seconds.

---

## Phase 24: Experiment Archaeology (v3.5.0)

*Manage a research project that spans months, not just sessions.*

### 24.1 Long-Term Trend Analysis — `/turing:trend`

**What:** Analyze the full experiment history for strategic patterns: are improvements slowing down? Which experiment families are exhausted? Is the search space being efficiently explored? Produces a "state of the research" that's more strategic than `/turing:brief`.

**Why:** After 100+ experiments, `/turing:brief` shows recent activity. `/turing:trend` shows the arc: "You spent 40 experiments on architecture search with diminishing returns. The last 10 feature engineering experiments had a steeper improvement curve. Shift focus." This strategic view is invisible in individual experiment logs.

**Implementation:**
1. Create `commands/trend.md` — `/turing:trend [--window 30d] [--metric accuracy]`
2. Add `templates/scripts/trend_analysis.py`:
   - **Improvement velocity:** metric improvement per experiment over time. Is it accelerating, steady, or decelerating?
   - **Family ROI:** for each experiment family, compute experiments-per-unit-improvement. Which families are high-ROI vs exhausted?
   - **Search space coverage:** cluster experiments by config similarity. Are experiments spreading across the search space or clustering?
   - **Diminishing returns detection:** fit a logarithmic curve to improvement-over-experiments. Predict how many more experiments for the next 0.5% gain.
   - **Phase transition detection:** identify moments where a new approach caused a step-change vs incremental improvement
   - Report:
     ```
     Research Trend Analysis (127 experiments over 3 weeks):
     
     Improvement Velocity:
       Week 1: +0.082 (high — initial exploration)
       Week 2: +0.031 (moderate — refinement)
       Week 3: +0.008 (low — diminishing returns) ← YOU ARE HERE
     
     Family ROI:
       feature-engineering: 0.004/experiment (HIGH — still productive)
       architecture-search: 0.001/experiment (LOW — exhausted after 40 experiments)
       hyperparameter-tuning: 0.002/experiment (MEDIUM — some room left)
     
     Prediction: next 0.5% gain will take ~25 experiments at current rate.
     
     Recommendation: Stop architecture search. Double down on feature engineering.
     Consider new families: ensemble construction, data augmentation.
     ```
   - Writes to `experiments/trends/trend-YYYY-MM-DD.md`
3. Integration with `/turing:budget` (Phase 18.2) — trend predictions inform budget remaining
4. Add tests

**Acceptance:** `/turing:trend` identifies that architecture search is exhausted after 40 experiments and recommends shifting to feature engineering.

### 24.2 Session Context Restoration — `/turing:flashback`

**What:** Restore the researcher's mental state after days away from a project. Reads the last brief, recent decision packets, pending hypotheses, and recent experiment outcomes to produce a "where was I?" summary.

**Why:** Context switching is the biggest productivity killer in ML research. Coming back to a project after a week, the researcher has forgotten: what was the current best? What was I about to try? What failed? `/turing:flashback` takes 10 seconds to reconstruct what would otherwise take 30 minutes of log-reading.

**Implementation:**
1. Create `commands/flashback.md` — `/turing:flashback [--depth 7d]`
2. Add `templates/scripts/session_flashback.py`:
   - Reads recent artifacts (configurable lookback window):
     - Last `/turing:brief` report
     - Recent decision packets (Phase 6.2)
     - Pending hypotheses in queue
     - Last 5 experiments with outcomes
     - Last research mode (Phase 6.5)
     - Any `/turing:annotate` notes (Phase 24.4)
   - Produces a compact summary:
     ```
     Flashback (last active: 2026-03-25, 7 days ago):
     
     Current best: exp-089 — LightGBM, accuracy=0.883 ± 0.005
     Research mode: exploit
     Budget: 73/100 experiments used (73%)
     
     Last session:
       ✓ exp-087: Added polynomial features → 0.879 (kept, marginal)
       ✓ exp-088: Feature selection (top 15) → 0.881 (kept)
       ✓ exp-089: LightGBM + selected features → 0.883 (NEW BEST)
       ✗ exp-090: Neural net attempt → 0.861 (discarded)
     
     Pending hypotheses:
       1. [HIGH] "Try CatBoost with native categorical handling" (human-injected)
       2. [MED]  "Ensemble top 3 models" (from decision packet)
     
     Your note (2026-03-25): "Neural net failed because dataset is too small.
     Don't try deep learning again unless we get more data."
     
     Suggested next action: Execute hypothesis #1 (CatBoost)
     ```
3. Integration with all experiment tracking phases — reads their artifacts
4. Add `--oneline` flag for ultra-brief summary: "Best: exp-089 (0.883). Pending: CatBoost, ensemble. 27 experiments left."
5. Add tests

**Acceptance:** After a week away, `/turing:flashback` reconstructs the project state in under 10 seconds. The researcher starts working immediately instead of re-reading logs.

### 24.3 Experiment Lifecycle Cleanup — `/turing:archive`

**What:** Compress old experiment artifacts, prune dominated checkpoints, summarize archived experiments into a compact index. Keeps the project directory manageable after 200+ experiments.

**Why:** Experiment directories grow without bound: 200 experiments × checkpoints × logs × profiles × diagnoses = gigabytes of files. Disk fills up, directory listings are unreadable, and loading the full log into context becomes expensive. `/turing:archive` compresses the past while preserving queryable history.

**Implementation:**
1. Create `commands/archive.md` — `/turing:archive [--older-than 30d] [--keep-best 10] [--dry-run]`
2. Add `templates/scripts/experiment_archive.py`:
   - **Identify archivable experiments:** older than threshold, not Pareto-optimal, not the current best, not referenced by pending hypotheses
   - **Compress:** tar+gzip experiment artifacts (logs, profiles, diagnoses) into `experiments/archive/exp-NNN.tar.gz`
   - **Summarize:** create a compact summary entry for each archived experiment in `experiments/archive/index.yaml`:
     ```yaml
     - id: exp-023
       description: "XGBoost baseline with default params"
       metric: {accuracy: 0.834}
       status: discarded
       family: architecture-search
       archived: 2026-04-01
     ```
   - **Prune checkpoints:** use Phase 12.2's Pareto pruning on archived experiments (more aggressive — keep zero checkpoints for discarded experiments)
   - **Report:**
     ```
     Archive summary:
       Archived: 143 experiments (of 200)
       Preserved: 57 experiments (Pareto-optimal, recent, best, referenced)
       Space reclaimed: 8.7 GB → 1.2 GB (saved 7.5 GB)
       Summaries written to experiments/archive/index.yaml
     ```
   - Archived experiments remain queryable via the summary index and semantic search (Phase 9.1)
3. Add `--dry-run` flag: show what would be archived without doing it
4. Integration with `/turing:trend` — trend analysis works on both active and archived experiments
5. Add tests

**Depends on:** Phase 12.2 (checkpoint Pareto pruning)

**Acceptance:** `/turing:archive` reclaims 7.5 GB from a 200-experiment project while keeping all experiments queryable via the summary index.

### 24.4 Retrospective Experiment Annotations — `/turing:annotate`

**What:** Add human notes to any experiment after the fact. "This only worked because the data was pre-sorted" or "Don't try this again, the improvement was a data bug." Structured post-hoc knowledge that experiment logs can't capture.

**Why:** Experiment logs capture what happened. Annotations capture *why it mattered* and *what to learn from it*. Six months later, the researcher (or a collaborator) needs context that no automated metric can provide: "we tried this approach because a reviewer suggested it, but it turned out the reviewer was wrong about the dataset."

**Implementation:**
1. Create `commands/annotate.md` — `/turing:annotate <exp-id> "note text"` or `/turing:annotate --list`
2. Add `templates/scripts/experiment_annotations.py`:
   - **add:** `/turing:annotate exp-042 "This result is fragile — only works with the specific preprocessing in commit abc123"`
   - **tag:** `/turing:annotate exp-042 --tag "reviewer-requested" --tag "fragile"`
   - **list:** show all annotations for an experiment or across all experiments
   - **search:** find experiments by annotation content or tag
   - Annotations stored in `experiments/annotations.yaml`:
     ```yaml
     exp-042:
       - text: "This result is fragile — only works with specific preprocessing"
         author: human
         date: 2026-04-01
         tags: [fragile, preprocessing-dependent]
       - text: "Reviewer 2 specifically asked for this comparison"
         author: human
         date: 2026-04-02
         tags: [reviewer-requested]
     ```
   - Annotations surface in `/turing:brief`, `/turing:flashback`, `/turing:diff`
3. Integration with `/turing:paper` (Phase 14.2) — annotations with tag "paper-note" appear as footnotes
4. Integration with `/turing:transfer` (Phase 19.1) — annotations transfer as lessons to similar projects
5. Add tests

**Acceptance:** `/turing:annotate exp-042 "fragile result"` attaches a note that appears in all subsequent briefs and comparisons involving exp-042.

### 24.5 Natural Language Experiment Search — `/turing:search`

**What:** Query experiment history with natural language: "experiments that used dropout and got accuracy above 0.86" or "all failed LightGBM runs from last week." Combines semantic search (Phase 9.1) with structured filters over config fields and metrics.

**Why:** After 200+ experiments, finding specific runs is painful. `/turing:tag` requires pre-tagging. The semantic index (Phase 9.1) finds similar experiments but doesn't support structured constraints. `/turing:search` merges both: natural language for intent, structured filters for precision.

**Implementation:**
1. Create `commands/search.md` — `/turing:search <query> [--filter "metric>0.85"] [--limit 10]`
2. Add `templates/scripts/experiment_search.py`:
   - Parses the query into two components:
     - **Semantic:** embed the text portion and query the FAISS index (Phase 9.1)
     - **Structured:** extract filter predicates (metric comparisons, date ranges, status, family, tags)
   - Combines results: semantic similarity score × filter pass/fail
   - Output:
     ```
     Search: "LightGBM experiments with high accuracy"
     Filters: accuracy > 0.85, model_type = lightgbm
     
     Results (7 matches):
     | Exp ID  | Description                    | Accuracy | Family           | Status |
     |---------|--------------------------------|----------|------------------|--------|
     | exp-089 | LightGBM + dart + feat select  | 0.883    | model-comparison | kept   |
     | exp-056 | LightGBM baseline              | 0.865    | model-comparison | kept   |
     | exp-071 | LightGBM + deeper trees        | 0.861    | architecture     | kept   |
     ...
     ```
   - Supports date filters: `--filter "date>2026-03-20"`, status: `--filter "status=discarded"`, family: `--filter "family=ensemble"`
   - Searches annotations (Phase 24.4) as well as experiment descriptions
3. Integration with `/turing:flashback` — flashback uses search internally for context reconstruction
4. Add tests

**Depends on:** Phase 9.1 (semantic index), Phase 24.4 (annotations)

**Acceptance:** `/turing:search "neural net experiments that failed" --filter "date>2026-03-15"` returns all matching experiments with relevance ranking.

### 24.6 Experiment Template Library — `/turing:template`

**What:** Save an experiment configuration as a reusable template. "This XGBoost config with this preprocessing is my go-to for tabular classification." Templates persist across projects and feed into `/turing:transfer` (Phase 19.1).

**Why:** Researchers develop personal "recipes" — starting configurations they reach for based on task type. These recipes live in the researcher's head and are reconstructed from memory each time. `/turing:template` makes them explicit, versioned, and shareable.

**Implementation:**
1. Create `commands/template.md` — `/turing:template [save|list|apply|share]`
2. Add `templates/scripts/experiment_templates.py`:
   - **save:** `/turing:template save exp-042 --name "tabular-xgboost-v2" --description "XGBoost with feature selection, good for tabular classification with <50 features"`
     - Extracts: model config, preprocessing pipeline, feature engineering steps, evaluation protocol
     - Strips project-specific details (dataset paths, column names)
     - Saves to `~/.turing/templates/tabular-xgboost-v2.yaml` (global, cross-project)
   - **list:** show all saved templates with descriptions and source project:
     ```
     Templates (5 saved):
     | Name                | Description                              | Source Project   | Accuracy |
     |---------------------|------------------------------------------|------------------|----------|
     | tabular-xgboost-v2  | XGBoost + feat selection, <50 features   | fraud-detection  | 0.923    |
     | lightgbm-imbalanced | LightGBM for imbalanced classification   | churn-prediction | 0.891    |
     | nn-small-tabular    | 3-layer MLP for small tabular datasets   | credit-scoring   | 0.847    |
     ```
   - **apply:** `/turing:template apply tabular-xgboost-v2` — generates a `train.py` and `config.yaml` from the template, adapted to the current project's dataset
   - **share:** export template as a standalone YAML file for sharing with collaborators
3. Integration with `/turing:transfer` (Phase 19.1) — templates are the mechanism for transferring knowledge
4. Integration with `/turing:init` — suggest templates during project scaffolding based on task description
5. Templates stored at `~/.turing/templates/` (cross-project persistence)
6. Add tests

**Depends on:** Phase 19.1 (transfer for cross-project use)

**Acceptance:** `/turing:template save` captures a winning config. `/turing:template apply` in a new project produces a runnable starting point that beats a default configuration.

### 24.7 Experiment Replay — `/turing:replay`

**What:** Re-run a historical experiment with the current infrastructure — current code, current data, current libraries. Different from `/turing:reproduce` (which verifies the *same* result) — replay tests whether an *old approach* would do better *now*.

**Why:** ML projects evolve: data pipelines improve, bugs get fixed, preprocessing gets refined. An approach that failed 3 weeks ago might succeed today because the underlying data quality improved. `/turing:replay` answers "should I revisit old ideas?" without the researcher manually reconstructing configs.

**Implementation:**
1. Create `commands/replay.md` — `/turing:replay <exp-id> [--with-current-data] [--with-current-preprocessing]`
2. Add `templates/scripts/experiment_replay.py`:
   - Reads the experiment's config from `log.jsonl`
   - **Default replay:** applies the old config to the *current* `train.py` and data:
     - Uses current preprocessing (from current `prepare.py`)
     - Uses the old model config and hyperparameters
     - Runs with current library versions
   - **Selective replay:**
     - `--with-original-code`: checkout `train.py` from the experiment's git commit (like reproduce but with current data)
     - `--with-current-data`: use current data with old code (test if data improvements help old approaches)
     - `--with-current-preprocessing`: use current `prepare.py` with old model config
   - Comparison report:
     ```
     Replay: exp-023 (XGBoost baseline, originally accuracy=0.834)
     
     | Condition            | Accuracy | Δ from Original |
     |----------------------|----------|-----------------|
     | Original (3 weeks ago) | 0.834  | —               |
     | Current code + data  | 0.856    | +0.022          |
     | Current data only    | 0.851    | +0.017          |
     | Current code only    | 0.839    | +0.005          |
     
     Verdict: Data improvements account for 77% of the gain.
     This approach is worth revisiting with current infrastructure.
     ```
   - Optionally auto-queues a hypothesis if replay shows significant improvement: "Revisit exp-023 approach with current infrastructure"
   - Logs as new experiment with parent link and `family: "replay"`
3. Integration with `/turing:trend` (Phase 24.1) — identify old experiments worth replaying based on infrastructure changes
4. Add tests

**Depends on:** Experiment logging (Phase 1), git-based experiment tracking

**Acceptance:** `/turing:replay exp-023` shows that an old approach gains +2.2% from infrastructure improvements, prompting the researcher to revisit it.

---

## Phase 25: Research Communication (v4.0.0)

*The v4.0 milestone. Turing goes from research tool to research-to-communication pipeline. Every result becomes a shareable artifact — citations tracked, presentations generated, progress communicated.*

### 25.1 Citation & Attribution Manager — `/turing:cite`

**What:** Track which papers, codebases, datasets, and methods influenced each experiment. Generate bibliography, ensure proper attribution, and catch missing citations before submission.

**Why:** Citation management in ML is a mess: the researcher uses a method from a 2019 paper, bases preprocessing on a GitHub repo, and evaluates on a dataset with its own citation requirements. Forgetting any of these is embarrassing at best and an integrity issue at worst. `/turing:cite` makes attribution automatic and auditable.

**Implementation:**
1. Create `commands/cite.md` — `/turing:cite [add|list|check|bib]`
2. Add `templates/scripts/citation_manager.py`:
   - **add:** `/turing:cite add exp-042 --paper "Chen2016XGBoost" --url "https://arxiv.org/abs/1603.02754"` — associate a citation with an experiment
   - **add from lit:** automatically capture citations from `/turing:lit` searches (Phase 14.1)
   - **add from suggest:** capture citations from `/turing:suggest` model suggestions (Phase 7.1)
   - **list:** show all citations grouped by experiment, method, dataset, or codebase:
     ```
     Project citations (23 sources):
     
     Methods:
       [Chen2016] XGBoost — used in exp-031, exp-042, exp-053
       [Ke2017] LightGBM — used in exp-056, exp-089
     
     Datasets:
       [UCI] Heart Disease Dataset — used in all experiments
     
     Techniques:
       [Chawla2002] SMOTE — used in exp-044, exp-045
       [Platt1999] Platt Scaling — used in exp-078 (calibration)
     
     Missing citations:
       ⚠ exp-089 uses LightGBM but no citation for the `dart` boosting method
       ⚠ exp-078 uses isotonic regression calibration but no Zadrozny2002 citation
     ```
   - **check:** audit for missing citations — methods used without attribution
   - **bib:** generate BibTeX file from all project citations:
     ```bibtex
     @inproceedings{chen2016xgboost,
       title={XGBoost: A Scalable Tree Boosting System},
       author={Chen, Tianqi and Guestrin, Carlos},
       booktitle={KDD},
       year={2016}
     }
     ```
   - Citations stored in `experiments/citations.yaml` with DOI/arXiv IDs for deduplication
3. Integration with `/turing:paper` (Phase 14.2) — auto-generate bibliography and inline citations
4. Integration with `/turing:audit` (Phase 19.2) — add "proper attribution" to the audit checklist
5. Add tests

**Depends on:** Phase 14.1 (lit), Phase 7.1 (suggest), Phase 14.2 (paper)

**Acceptance:** `/turing:cite check` catches 2 missing method citations before paper submission. `/turing:cite bib` produces a complete BibTeX file.

### 25.2 Presentation Figure Generation — `/turing:present`

**What:** Generate presentation-ready figures from experiment data: training curves, comparison bar charts, ablation tables, Pareto plots, sensitivity heatmaps. Formatted for talks (large fonts, clean aesthetics), not papers.

**Why:** Researchers spend hours making figures for lab meetings, conference talks, and stakeholder presentations. The data is already in the experiment log — the transformation to visual form is purely mechanical. `/turing:present` generates publication-quality figures in seconds.

**Implementation:**
1. Create `commands/present.md` — `/turing:present [--figures training,comparison,ablation,pareto,sensitivity] [--format svg|png] [--style dark|light|poster]`
2. Add `templates/scripts/generate_figures.py`:
   - **Training curve:** metric over experiments/epochs with error bands (from seed studies):
     - Clean axes, large labels, legend outside plot
   - **Comparison bar chart:** model families side-by-side with error bars:
     - Sorted by metric, best highlighted, baseline marked
   - **Ablation table figure:** formatted ablation results as a visual table:
     - Delta bars, color-coded (green=helps, red=hurts, gray=negligible)
   - **Pareto plot:** scatter of accuracy vs latency/size with Pareto frontier line:
     - Points labeled with experiment IDs, dominated points faded
   - **Sensitivity heatmap:** hyperparameter sensitivity as a colored grid:
     - Rows = hyperparameters, columns = values, color = metric
   - Style presets:
     - `light`: white background, clean for papers and slides
     - `dark`: dark background, good for live demos
     - `poster`: extra large fonts, simplified axes
   - Outputs to `paper/figures/` directory
   - Uses matplotlib with a custom Turing style sheet
3. Integration with `/turing:paper` — auto-include figures in paper sections
4. Add `--deck` flag: generate a reveal.js/Mermaid slide deck from all available figures with auto-generated captions
5. Add tests

**Acceptance:** `/turing:present` produces 5 publication-quality figures from experiment data in under 30 seconds. The researcher pastes them directly into their slide deck.

### 25.3 Model Changelog Generation — `/turing:changelog`

**What:** Auto-generate a human-readable changelog of model improvements from experiment history. "v3: switched from XGBoost to LightGBM (+1.2%), added polynomial features (+0.5%)." For communicating progress to non-technical stakeholders.

**Why:** ML researchers track progress in experiment logs. Stakeholders (PMs, executives, clients) want a simple narrative: "what improved, by how much, and why?" Translating between these formats is tedious and error-prone. `/turing:changelog` automates the translation.

**Implementation:**
1. Create `commands/changelog.md` — `/turing:changelog [--since exp-id|date] [--audience technical|stakeholder]`
2. Add `templates/scripts/generate_changelog.py`:
   - Reads experiment history, identifies "keep" decisions that improved the primary metric
   - Groups improvements by family/category:
     ```
     Model Changelog (since exp-001 baseline):
     
     ## v4 — Current (exp-089, accuracy: 0.883)
     - Switched to LightGBM with dart boosting (+0.011)
     - Applied feature selection: top 15 features (+0.002)
     - Added Platt calibration for probability outputs (ECE: 0.068 → 0.021)
     
     ## v3 (exp-078, accuracy: 0.872)
     - Added polynomial features for top 5 features (+0.005)
     - Increased n_estimators from 300 to 500 (+0.003)
     
     ## v2 (exp-042, accuracy: 0.864)
     - Switched from random forest to XGBoost (+0.021)
     - Tuned learning rate from 0.3 to 0.1 (+0.009)
     
     ## v1 — Baseline (exp-001, accuracy: 0.834)
     - Logistic regression baseline
     
     Total improvement: +0.049 (5.9%) over 89 experiments
     ```
   - **Audience adaptation:**
     - `technical`: includes hyperparameters, experiment IDs, metrics
     - `stakeholder`: plain English, percentage improvements, no jargon
   - Writes to `paper/CHANGELOG.md`
3. Integration with `/turing:brief` — changelog summary appended to briefings
4. Integration with `/turing:annotate` — changelog entries can be manually overridden with human explanations
5. Add tests for changelog generation, version detection, audience formatting

**Acceptance:** `/turing:changelog --audience stakeholder` produces a 1-page summary that a PM can read in 2 minutes and understand the current model's evolution.

---

## Updated Full Implementation Order

| # | Feature | Phase | Version | Priority | Status | Depends On |
|---|---------|-------|---------|----------|--------|------------|
| 1–26 | Phases 1–9 | 1–9 | v1.0–v1.2 | — | **DONE** | — |
| 27 | Multi-seed runner `/turing:seed` | 10.1 | v1.3.0 | **Critical** | **DONE** | Phase 2.1 (statistical compare) |
| 28 | Reproducibility verification `/turing:reproduce` | 10.2 | v1.3.0 | **High** | **DONE** | Experiment logging (Phase 1) |
| 29 | Error analysis `/turing:diagnose` | 11.1 | v1.4.0 | **Critical** | **DONE** | Phase 3.1 (metric decomposition) |
| 30 | Ablation studies `/turing:ablate` | 11.2 | v1.4.0 | **High** | **DONE** | Phase 10.1 (seed runner) |
| 31 | Pareto frontier `/turing:frontier` | 11.3 | v1.4.0 | **Medium** | **DONE** | Experiment logging |
| 32 | Computational profiling `/turing:profile` | 12.1 | v1.5.0 | **High** | **DONE** | — (standalone) |
| 33 | Smart checkpoint manager `/turing:checkpoint` | 12.2 | v1.5.0 | **Medium** | **DONE** | Phase 11.3 (Pareto logic) |
| 34 | Model export `/turing:export` | 13.1 | **v2.0.0** | **High** | **DONE** | Phase 10.1 (seed study for model card) |
| 35 | Literature integration `/turing:lit` | 14.1 | v2.1.0 | **Medium** | **DONE** | Phase 7.1 (scholarly API infra) |
| 36 | Paper section drafting `/turing:paper` | 14.2 | v2.1.0 | **Medium** | **DONE** | Phases 10.1, 11.2, 14.1, 6.3 |
| 37 | Experiment scheduler `/turing:queue` | 15.1 | v2.2.0 | **Critical** | Planned | — (standalone) |
| 38 | Smart failure recovery `/turing:retry` | 15.2 | v2.2.0 | **High** | Planned | Phase 11.1 (diagnose) |
| 39 | Experiment branching `/turing:fork` | 15.3 | v2.2.0 | **High** | Planned | Phase 1.3 (dependency graph) |
| 40 | Deep experiment comparison `/turing:diff` | 16.1 | v2.3.0 | **High** | Planned | Phase 11.3 (frontier metrics) |
| 41 | Live training monitor `/turing:watch` | 16.2 | v2.3.0 | **High** | Planned | Phase 12.1 (profiling infra) |
| 42 | Performance regression gate `/turing:regress` | 16.3 | v2.3.0 | **Medium** | Planned | Phase 10.2 (reproduce) |
| 43 | Automated ensemble construction `/turing:ensemble` | 17.1 | v2.4.0 | **High** | Planned | Phase 11.3 (Pareto for model selection) |
| 44 | Pipeline composition `/turing:stitch` | 17.2 | v2.4.0 | **High** | Planned | Phase 11.2 (ablation for stage testing) |
| 45 | Warm-start from prior model `/turing:warm` | 17.3 | v2.4.0 | **Medium** | Planned | Phase 12.2 (checkpoint manager) |
| 46 | Scaling law estimator `/turing:scale` | 18.1 | v2.5.0 | **High** | Planned | Phase 10.1 (seed for statistical fit) |
| 47 | Compute budget manager `/turing:budget` | 18.2 | v2.5.0 | **High** | Planned | Phase 15.1 (queue integration) |
| 48 | Model compression `/turing:distill` | 18.3 | v2.5.0 | **Medium** | Planned | Phase 13.1 (export for size comparison) |
| 49 | Cross-project knowledge transfer `/turing:transfer` | 19.1 | **v3.0.0** | **High** | Planned | Phase 9.1 (semantic index) |
| 50 | Pre-submission methodology audit `/turing:audit` | 19.2 | **v3.0.0** | **High** | Planned | Phases 10.1, 11.2, 14.2 |
| 51 | Pre-training sanity checks `/turing:sanity` | 20.1 | v3.1.0 | **High** | **DONE** | — (standalone) |
| 52 | Automatic baseline generation `/turing:baseline` | 20.2 | v3.1.0 | **High** | **DONE** | Phase 10.1 (seed runner) |
| 53 | Targeted leakage detection `/turing:leak` | 20.3 | v3.1.0 | **Critical** | **DONE** | Phase 19.2 (audit integration) |
| 54 | Internal model diagnostics `/turing:xray` | 21.1 | v3.2.0 | **High** | **DONE** | Phase 11.1 (diagnose) |
| 55 | Hyperparameter sensitivity `/turing:sensitivity` | 21.2 | v3.2.0 | **High** | **DONE** | Phase 10.1 (seed runner) |
| 56 | Probability calibration `/turing:calibrate` | 21.3 | v3.2.0 | **Medium** | **DONE** | Phase 13.1 (export) |
| 57 | Automated feature selection `/turing:feature` | 22.1 | v3.3.0 | **High** | **DONE** | Phase 11.2 (ablation) |
| 58 | Training curriculum optimization `/turing:curriculum` | 22.2 | v3.3.0 | **Medium** | **DONE** | Phase 10.1 (seed runner) |
| 59 | Weight pruning `/turing:prune` | 23.1 | v3.4.0 | **High** | **DONE** | Phase 11.3 (Pareto), Phase 13.1 (export) |
| 60 | Post-training quantization `/turing:quantize` | 23.2 | v3.4.0 | **High** | **DONE** | Phase 13.1 (export) |
| 61 | Model merging `/turing:merge` | 23.3 | v3.4.0 | **High** | **DONE** | Phase 12.2 (checkpoint), Phase 13.1 (export) |
| 62 | Architecture modification `/turing:surgery` | 23.4 | v3.4.0 | **Medium** | **DONE** | Phase 17.3 (warm), Phase 21.2 (sensitivity) |
| 63 | Long-term trend analysis `/turing:trend` | 24.1 | v3.5.0 | **High** | **DONE** | Phase 18.2 (budget) |
| 64 | Session context restoration `/turing:flashback` | 24.2 | v3.5.0 | **Critical** | **DONE** | Phases 6.2, 6.5, 24.7 |
| 65 | Experiment lifecycle cleanup `/turing:archive` | 24.3 | v3.5.0 | **Medium** | **DONE** | Phase 12.2 (checkpoint pruning) |
| 66 | Retrospective annotations `/turing:annotate` | 24.4 | v3.5.0 | **Medium** | **DONE** | — (standalone) |
| 67 | Natural language experiment search `/turing:search` | 24.5 | v3.5.0 | **High** | **DONE** | Phase 9.1 (semantic index), Phase 24.4 |
| 68 | Experiment template library `/turing:template` | 24.6 | v3.5.0 | **Medium** | **DONE** | Phase 19.1 (transfer) |
| 69 | Experiment replay `/turing:replay` | 24.7 | v3.5.0 | **Medium** | **DONE** | Experiment logging (Phase 1) |
| 70 | Citation & attribution manager `/turing:cite` | 25.1 | **v4.0.0** | **High** | Planned | Phases 14.1, 7.1, 14.2 |
| 71 | Presentation figure generation `/turing:present` | 25.2 | **v4.0.0** | **High** | Planned | Phases 11.2, 11.3, 21.2 |
| 72 | Model changelog generation `/turing:changelog` | 25.3 | **v4.0.0** | **Medium** | Planned | Phase 24.4 (annotations) |

### Dependency Graph

```
Phases 1–14 (v1.0–v2.1) ─── ALL DONE ───────────────────────────────────

Phase 15 (Orchestration)              Phase 16 (Deep Analysis)
  15.1 queue (standalone)               16.1 diff ← 11.3 frontier
  15.2 retry ← 11.1 diagnose           16.2 watch ← 12.1 profile
  15.3 fork ← 1.3 dep graph            16.3 regress ← 10.2 reproduce
           │                                     │
           ▼                                     ▼
Phase 17 (Model Composition)          Phase 18 (Scaling & Efficiency)
  17.1 ensemble ← 11.3 Pareto          18.1 scale ← 10.1 seed
  17.2 stitch ← 11.2 ablation          18.2 budget ← 15.1 queue
  17.3 warm ← 12.2 checkpoint          18.3 distill ← 13.1 export
                                                 │
                                                 ▼
                                      Phase 19 (Meta-Intelligence) ← v3.0
                                        19.1 transfer ← 9.1 semantic index
                                        19.2 audit ← 10.1, 11.2, 14.2
                                                 │
           ┌─────────────────────────────────────┘
           ▼
Phase 20 (Pre-Training Intel)         Phase 21 (Model Debugging)
  20.1 sanity (standalone)              21.1 xray ← 11.1 diagnose
  20.2 baseline ← 10.1 seed            21.2 sensitivity ← 10.1 seed
  20.3 leak ← 19.2 audit               21.3 calibrate ← 13.1 export
           │                                     │
           ▼                                     ▼
Phase 22 (Feature & Training)         Phase 23 (Model Surgery)
  22.1 feature ← 11.2 ablation         23.1 prune ← 11.3 Pareto, 13.1 export
  22.2 curriculum ← 10.1 seed          23.2 quantize ← 13.1 export
                                        23.3 merge ← 12.2 checkpoint, 13.1 export
                                        23.4 surgery ← 17.3 warm, 21.2 sensitivity
                                                 │
           ┌─────────────────────────────────────┘
           ▼
Phase 24 (Experiment Archaeology)
  24.1 trend ← 18.2 budget
  24.2 flashback ← 6.2, 6.5, 24.7
  24.3 archive ← 12.2 checkpoint
  24.4 annotate (standalone)
  24.5 search ← 9.1 semantic index, 24.4
  24.6 template ← 19.1 transfer
  24.7 replay ← experiment logging
           │
           ▼
Phase 25 (Research Communication) ← v4.0
  25.1 cite ← 14.1, 7.1, 14.2
  25.2 present ← 11.2, 11.3, 21.2
  25.3 changelog ← 24.4 annotate
```

### Version Release Criteria

| Version | Phase | Release Gate |
|---------|-------|-------------|
| v1.3.0 | 10 (Statistical Rigor) | **RELEASED** |
| v1.4.0 | 11 (Experiment Intelligence) | **RELEASED** |
| v1.5.0 | 12 (Performance) | **RELEASED** |
| **v2.0.0** | **13 (Deployment Bridge)** | **RELEASED** |
| v2.1.0 | 14 (Research Workflow) | **RELEASED** |
| v2.2.0 | 15 (Orchestration) | `/turing:queue` runs 10 experiments overnight with priority ordering; `/turing:retry` auto-recovers from OOM by reducing batch size |
| v2.3.0 | 16 (Deep Analysis) | `/turing:diff` identifies the divergence point between two experiments; `/turing:watch` catches a loss spike mid-training |
| v2.4.0 | 17 (Model Composition) | `/turing:ensemble` beats the best single model on 3 tasks; `/turing:stitch` swaps a pipeline stage without rewriting train.py |
| v2.5.0 | 18 (Scaling & Efficiency) | `/turing:scale` predicts full-dataset accuracy within 2% from 10% data runs; `/turing:budget` halts exploration when budget exhausted |
| **v3.0.0** | **19 (Meta-Intelligence)** | **`/turing:transfer` surfaces a winning hypothesis from a prior project; `/turing:audit` catches a real methodological error — Turing becomes project-aware, not just experiment-aware** |
| v3.1.0 | 20 (Pre-Training Intelligence) | `/turing:sanity` catches a broken data loader in <30s; `/turing:leak` detects a leaked feature before training |
| v3.2.0 | 21 (Model Debugging) | `/turing:xray` identifies dead neurons; `/turing:sensitivity` redirects tuning effort to high-impact hyperparameters |
| v3.3.0 | 22 (Feature & Training Intelligence) | `/turing:feature` identifies redundant features that hurt generalization; `/turing:curriculum` demonstrates faster convergence |
| v3.4.0 | 23 (Model Surgery) | `/turing:prune` achieves 1.8x speedup at <1% accuracy loss; `/turing:merge` beats best individual model with zero latency overhead; `/turing:surgery` produces runnable modified architecture in <10s |
| v3.5.0 | 24 (Experiment Archaeology) | `/turing:flashback` restores context in <10s after days away; `/turing:search` finds relevant experiments via natural language; `/turing:template` transfers a winning recipe to a new project |
| **v4.0.0** | **25 (Research Communication)** | **`/turing:cite` catches missing attributions; `/turing:present` generates publication-quality figures; `/turing:changelog` produces stakeholder-readable progress narrative — Turing becomes a research-to-communication pipeline** |
