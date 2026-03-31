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
| 9 | Hidden file tier | 5.1 | **Critical** | Planned | — |
| 10 | Behavioral probes | 5.2 | **Critical** | Planned | — |
| 11 | Stability validation | 5.3 | **High** | Planned | — |
| 12 | Tool restriction | 5.4 | **High** | Partial | — |
| 13 | Diff-based history | 5.5 | **Medium** | Planned | — |
| 14 | Novelty guard | 6.1 | **Critical** | Planned | — |
| 15 | Decision packets | 6.2 | **High** | Planned | — |
| 16 | Experiment families | 6.3 | **High** | Planned | — |
| 17 | Failure clustering | 6.4 | **Medium** | Planned | — |
| 18 | Research mode selection | 6.5 | **Medium** | Planned | — |

Phases 1-4 complete (165 tests). Phase 5 (anti-cheating) and Phase 6 (MemoryLab research-ops) are next.

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
