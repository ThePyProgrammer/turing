# turing

*The research assistant that can't fool itself.*

---

An autonomous ML research harness for Claude Code. Turing implements the autoresearch pattern — an AI agent that iteratively trains, evaluates, and improves machine learning models through a structured experiment loop with convergence detection, immutable evaluation infrastructure, and safety guardrails.

The name references Alan Turing — the person who first asked whether machines could think, then built the framework for answering the question. Turing the plugin does what Turing the person formalized: it defines a computational process, executes it mechanically, and determines whether the result constitutes an improvement.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [snoglobe/helios](https://github.com/snoglobe/helios).

## Three Commands

That's all you need.

```
/turing:init                          Set up a new ML project
/turing:train                         Run the experiment loop
/turing:brief                         What happened? What's next?
```

Initialize. Train. Read the briefing. Inject your taste. Repeat.

```
/turing:try switch to LightGBM        Steer the agent
/turing:train                          It follows your lead
/turing:brief --deep                   Get literature-backed suggestions
```

Everything else — experiment logging, convergence detection, hypothesis tracking, statistical validation, anti-cheating guardrails — happens automatically. You think about *what* to try. Turing handles *how* to try it.

## Table of Contents

- [When Code Is Free, Research Is All That Matters](#when-code-is-free-research-is-all-that-matters)
- [The Human-AI Interface](#the-human-ai-interface)
- [The Problem Turing Solves](#the-problem-turing-solves)
- [Philosophical Foundations](#philosophical-foundations)
- [How Turing Works](#how-turing-works)
- [Commands](#commands)
- [The Hypothesis Database](#the-hypothesis-database)
- [The Agent Architecture](#the-agent-architecture)
- [The Anti-Cheating Stack](#the-anti-cheating-stack)
- [Convergence Detection](#convergence-detection)
- [Installation](#installation)
- [Architecture of Turing Itself](#architecture-of-turing-itself)
- [Intellectual Heritage](#intellectual-heritage)

## When Code Is Free, Research Is All That Matters

> *"You're in a room with a quadrillion biased coins, and you want to maximize the number of heads in the shortest amount of time. Almost all coins are 'duds.' The novice coin-flipper might start flipping one-by-one, but heads come few and far between. The learned coin-flipper weaves through the quadrillion-coin room with a preternatural air; they flip many coins at once. What comes across as luck is really the refinement of taste: years of feeling faint differences in the weight of the metal, the subtle offsets of a mis-mint."* — [Amy Tam](https://x.com/amytam01/status/2031072399731675269)

This is the most precise metaphor for ML research in the age of autonomous agents: a quadrillion-coin room where the researcher's value lies not in the mechanical act of flipping but in *choosing which coins to flip at all*.

Tam's insight cuts to the heart of what Turing exists to do. The agentic coding tools consuming software engineering alive right now — Cursor, Claude Code, Codex — work precisely because engineering has a built-in feedback signal: a test to pass, a spec to meet, a benchmark to clear. You can RL on [SWE-bench](https://www.swebench.com/) because the ground truth exists. **Research has no equivalent.** It is not clear what it means to RL on a research question, because it is not clear what definition of "ground truth" one should optimize for. The coin room has a quadrillion coins but no label telling you which ones are biased toward heads.

And yet Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) ran 126 experiments overnight on a single GPU: agents modifying LLM training code, running a five-minute training loop, checking if the result improved, and repeating. [Tobias Lütke reported](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/) that after letting it run overnight, it executed 37 experiments and delivered a 19% performance gain. That is a lot more coins flipped than the average human in the same time.

This creates a new kind of division of labor:

```
HUMAN RESEARCHER                    AUTONOMOUS AGENT
─────────────────                   ─────────────────
Research taste                      Coin flipping
Which coins to flip                 How fast to flip them
Problem selection                   Hypothesis execution
Judgment under ambiguity            Measurement under control
Knowing when the room has changed   Running the room as-is
```

The researcher's job becomes the selection function: *which 20 of the quadrillion coins are worth flipping in the first place?* And the agent's job — Turing's job — is to flip those coins with the discipline, speed, and memory that humans cannot sustain. Every experiment logged. Every variant preserved. Every comparison valid. No amnesia. No fatigue. No accidental contamination of the measurement.

*When anyone can build for free, the differentiator is knowing what's worth building and whether it's buildable at all.* Turing handles the building. You bring the knowing.

## The Human-AI Interface

Turing is not a black box you point at data and hope for the best. It is a conversation between your taste and the agent's discipline.

### The Taste-Leverage Loop

```
         ┌─────────────────────┐
         │  YOU (taste)         │
         │                     │
         │  /turing:brief      │◄──── "What have we learned?"
         │  /turing:try ...    │────► "Try this next."
         └────────┬────────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │  TURING (discipline) │
         │                     │
         │  Hypothesize        │◄──── Reads your injection + history
         │  Train              │────► Runs the experiment
         │  Evaluate           │────► Immutable measurement
         │  Decide             │────► Keep or discard
         │  Record             │────► Updates hypothesis database
         └────────┬────────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │  BRIEFING            │
         │                     │
         │  Campaign summary   │
         │  Best model         │
         │  What's exhausted   │
         │  What's promising   │
         │  Recommendations    │
         └─────────────────────┘
                  │
                  ▼
              You again.
```

The loop is bidirectional. You inject hypotheses. The agent executes them. The briefing tells you what happened. You inject new hypotheses informed by the results. The agent never forgets what it tried. You never lose context between sessions.

### What This Looks Like in Practice

**Morning 1:** You have a dataset and a prediction task.

```
/turing:init
# Answer: project name, metric, data location
# Turing scaffolds everything
```

**Morning 1, 10 minutes later:**

```
/turing:train
# Agent runs 5-10 experiments autonomously
# XGBoost baseline → hyperparameter sweep → convergence
```

**Morning 1, 30 minutes later:**

```
/turing:brief
# Campaign: 8 experiments, 5 kept, accuracy 0.82 → 0.87
# Best: XGBoost, max_depth=6, n_estimators=200
# Exhausted: hyperparameter tuning on XGBoost
# Recommendation: try LightGBM or feature engineering
```

**Your taste kicks in:**

```
/turing:try switch to LightGBM with dart boosting — XGBoost plateaued
/turing:try add polynomial interaction features for the numeric columns
/turing:train
```

**Afternoon:**

```
/turing:brief --deep
# Standard briefing + literature-grounded suggestions
# Papers suggest: target encoding for high-cardinality categoricals
# → Auto-queued as hyp-012
```

**You leave. Come back tomorrow.**

```
/turing:brief
# Everything is there. Nothing was forgotten.
# The hypothesis database has the complete trail.
```

That's the interface. Six words to inject an idea. One command to get a briefing. The agent handles everything in between.

## The Problem Turing Solves

> "An experiment is a question which science poses to Nature, and a measurement is the recording of Nature's answer." — Max Planck

The central activity of machine learning research is the experiment loop: change something, train, evaluate, decide, repeat. This loop is simultaneously the most important and the most tedious part of ML work. Researchers spend their days doing what is essentially a manual search over a high-dimensional space of model architectures, hyperparameters, feature transformations, and data preprocessing strategies.

The tragedy is not that this is slow — it is that the process is structurally unsound. When a human researcher modifies both the training code *and* the evaluation code in the same session, the experiment is no longer a controlled experiment. When experiment results are tracked in notebook cells rather than structured logs, reproducibility is aspirational. When a promising direction is abandoned because the researcher forgot what they tried three hours ago, the search is not even a search — it is a random walk with amnesia.

Turing does not replace the researcher's judgment. It replaces the researcher's *discipline* — or more precisely, it makes discipline the default rather than an act of willpower. The experiment loop is formalized. The evaluation harness is immutable. Every experiment is logged. Every code variant is preserved. Convergence is detected automatically. The researcher's role shifts from "person who types hyperparameters and reads loss curves" to "person who decides what hypotheses are worth testing" — from coin-flipper to coin-selector.

## Philosophical Foundations

### On Separating Hypothesis from Measurement

> "The first principle is that you must not fool yourself — and you are the easiest person to fool." — Richard Feynman

Turing is built on a specific epistemological claim: **the entity that generates hypotheses must not be the entity that evaluates them**. This is not a software engineering pattern — it is the methodological foundation of modern science, and it predates software by centuries.

In experimental physics, the [double-blind protocol](https://en.wikipedia.org/wiki/Blinded_experiment) ensures that the experimenter's expectations cannot influence the measurement. In ML, the equivalent risk is more insidious: an agent that can modify both `train.py` and `evaluate.py` can — deliberately or through optimization pressure — find metrics that look good but don't reflect genuine model improvement.

This is [Goodhart's Law](https://en.wikipedia.org/wiki/Goodhart%27s_law) made architectural: *"When a measure becomes a target, it ceases to be a good measure."* The only defense is to make the measure structurally immutable.

Turing enforces this with a three-tier access model:

```
┌──────────────────────────────────────────────────────┐
│                  HYPOTHESIS SPACE                     │
│              (agent can modify)                       │
│    train.py          config.yaml                     │
├──────────────────────────────────────────────────────┤
│                MEASUREMENT APPARATUS                  │
│         prepare.py (READ-ONLY)                       │
│         evaluate.py (HIDDEN — agent cannot even see) │
└──────────────────────────────────────────────────────┘
```

The evaluation harness is not just immutable — it is *invisible*. The agent cannot read `evaluate.py`, cannot discover its implementation, cannot reverse-engineer fixed seeds or scoring formulas. It knows only the metric name, the direction (higher or lower is better), and the result. This is the difference between "please don't change the test" and "you literally cannot see the test."

### On Research Taste and Autonomous Execution

> *"Research taste is about how well you choose your coins: how well you choose which problems are worth working on at all."* — Amy Tam

There is a paradox at the heart of autonomous ML research: the parts of research that are hardest to automate are precisely the parts that matter most. Problem selection, hypothesis formation, knowing when a line of inquiry has become a dead end — these require what Tam calls *taste*, the accumulated judgment that comes from years of feeling faint differences in which problems are tractable, which results are meaningful, and which metrics actually capture what you care about.

Autoresearch does not solve this. Turing does not solve this. No one has solved this. But what autoresearch *does* solve is the complementary problem: given a well-selected hypothesis space, execute the search within it with superhuman discipline and throughput. The human provides the taste. The agent provides the tirelessness.

This is why Turing's interface is built around two verbs: **try** and **brief**. `/turing:try` is how taste reaches the agent. `/turing:brief` is how results reach the human. Everything else is infrastructure.

### On Experiment Tracking as Institutional Memory

> "Those who cannot remember the past are condemned to repeat it." — George Santayana

An LLM agent without persistent memory is a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) — its next action depends only on its current state, not on the path that led there. This is catastrophically inefficient for optimization: the agent will re-try failed approaches, abandon promising directions, and fail to recognize when it has converged. It will keep flipping coins it has already flipped.

Turing addresses this with a structured memory stack:

| System | Format | Purpose |
|--------|--------|---------|
| **Hypothesis database** | `hypotheses.yaml` + `hypotheses/hyp-NNN.yaml` | Complete ledger of every idea — human and agent — with full detail |
| **Experiment log** | `experiments/log.jsonl` | Append-only record of every experiment run |
| **Novelty guard** | `scripts/novelty_guard.py` | Blocks duplicate and near-duplicate hypotheses before execution |
| **Agent memory** | `.claude/agent-memory/ml-researcher/MEMORY.md` | Working notes across sessions |
| **Git history** | Experiment branches | Every code variant preserved |

The hypothesis database is the single source of truth. Every idea gets registered before execution. Every outcome gets written back. The novelty guard reads the history and prevents the agent from re-trying things it has already failed at — even across `/loop` sessions where the agent's context is lost.

## How Turing Works

### The Experiment Loop

Every iteration follows the same protocol:

```
1. OBSERVE     Read metrics, check hypothesis queue, review failed diffs
2. HYPOTHESIZE Check queue (human ideas first) or generate + register own
3. PREPARE     Edit train.py or config.yaml
4. COMMIT      Git branch per experiment
5. EXECUTE     python train.py > run.log 2>&1
6. MEASURE     Parse metrics (agent can't see how they're computed)
7. DECIDE      Keep improvements, revert regressions
8. RECORD      Log experiment, update hypothesis, synthesize decision
9. CONVERGE?   Stop after N non-improvements, or repeat
```

### The Hypothesis Lifecycle

Every experiment — human-injected or agent-generated — flows through the hypothesis database:

```
  /turing:try "idea"                     Agent generates idea
        │                                       │
        ▼                                       ▼
  ┌──────────────────────────────────────────────────┐
  │  hypotheses.yaml (index)                          │
  │  hypotheses/hyp-001.yaml (detail)                 │
  │                                                   │
  │  architecture:                                    │
  │    model_type: lightgbm                           │
  │  hyperparameters:                                 │
  │    n_estimators: 200                              │
  │    learning_rate: 0.05                            │
  │  expected_outcome:                                │
  │    rationale: "dart boosting may escape plateau"  │
  │  family: architecture-search                      │
  │  tags: [lightgbm, dart]                           │
  └────────────────────┬──────────────────────────────┘
                       │
                 novelty guard
                 (block duplicates)
                       │
                       ▼
                  experiment
                       │
                       ▼
  ┌──────────────────────────────────────────────────┐
  │  result:                                          │
  │    experiment_id: exp-007                          │
  │    metrics: {accuracy: 0.89}                      │
  │    verdict: promising                             │
  │    notes: "3% improvement, follow up with..."     │
  └──────────────────────────────────────────────────┘
```

The index (`hypotheses.yaml`) is the lightweight queue. The detail files (`hypotheses/hyp-NNN.yaml`) hold the full structured record: architecture, hyperparameters, features, expected outcome, actual result, lineage, family tags. Both are updated atomically.

## Commands

### Core Loop

| Command | What it does |
|---------|-------------|
| `/turing:init [--plan]` | Scaffold a new ML project. `--plan` generates a literature-grounded research plan. Supports multiple projects in subdirectories. |
| `/turing:train [ml/project] [N]` | Run the experiment loop. Auto-detects project from cwd or explicit path. |
| `/turing:sweep` | Systematic hyperparameter sweep via cartesian product |
| `/turing:status` | Quick experiment status — best model, convergence state |
| `/turing:compare <a> <b>` | Side-by-side experiment comparison with causal analysis |

### Taste-Leverage Interface

| Command | What it does |
|---------|-------------|
| `/turing:try <hypothesis>` | Inject a hypothesis — free text or `archetype:model_comparison` |
| `/turing:brief [--deep]` | Research briefing — campaign summary, failure patterns, literature-grounded suggestions |
| `/turing:suggest` | Literature-grounded model architecture suggestions with citations |
| `/turing:suggest --strategy treequest` | Tree-search hypothesis exploration (alias for `/turing:explore`) |
| `/turing:explore` | AB-MCTS tree search over critique-scored hypothesis space |
| `/turing:design <hyp-id>` | Generate structured experiment design from a hypothesis |
| `/turing:mode <explore\|exploit\|replicate>` | Set research strategy — drives novelty guard policy |

### Reporting & Validation

| Command | What it does |
|---------|-------------|
| `/turing:validate [--auto]` | Check metric stability — auto-configure multi-run if noisy |
| `/turing:seed [N] [--quick]` | Multi-seed study — mean/std/CI, flag seed-sensitive results |
| `/turing:reproduce <exp-id>` | Reproducibility verification — re-run and check tolerance |
| `/turing:diagnose [exp-id]` | Error analysis — failure modes, confused pairs, feature-range bias |
| `/turing:ablate [--components]` | Ablation study — remove components, measure impact, flag dead weight |
| `/turing:frontier [--metrics]` | Pareto frontier — multi-objective tradeoff visualization |
| `/turing:profile [exp-id]` | Computational profiling — timing, memory, throughput, bottleneck detection |
| `/turing:checkpoint <action>` | Smart checkpoint management — list, prune (Pareto), average, resume, stats |
| `/turing:lit <query>` | Literature search — papers, SOTA baselines, related work |
| `/turing:paper [--sections] [--format]` | Draft paper sections from experiment logs (setup, results, ablation, hyperparams) |
| `/turing:queue <action>` | Batch experiment scheduler — add, list, run, pause, clear |
| `/turing:retry <exp-id>` | Smart failure recovery — auto-diagnose crash, apply fix, re-run |
| `/turing:fork <exp-id>` | Experiment branching — run parallel tracks, report winner |
| `/turing:export [--format]` | Export model to production format with equivalence check + latency benchmark |
| `/turing:card` | Generate a model card — performance, limitations, intended use, artifact contract |
| `/turing:logbook` | Generate HTML experiment logbook |
| `/turing:report` | Generate research report |
| `/turing:poster` | Generate research poster |
| `/turing:preflight` | Pre-release validation checks |
| `/turing:diff <a> <b>` | Deep experiment comparison — config diffs, metric significance, per-class regressions, curve divergence |
| `/turing:watch [--analyze]` | Live training monitor — loss spikes, NaN detection, overfitting, plateau alerts |
| `/turing:regress [--tolerance]` | Performance regression gate — verify metrics haven't degraded after changes |
| `/turing:ensemble [--top-k]` | Automated ensemble — voting, stacking, blending from top-K models |
| `/turing:stitch <action>` | Pipeline composition — show, swap, cache, and run stages independently |
| `/turing:warm <exp-id>` | Warm-start from prior model — load checkpoint, freeze layers, adjust LR |
| `/turing:scale [--axis]` | Scaling law estimator — power-law fit, full-scale predictions, diminishing returns verdict |
| `/turing:budget <action>` | Compute budget manager — set limits, track allocation, auto-shift explore/exploit |
| `/turing:distill <exp-id>` | Model compression — distill teacher into smaller student with accuracy/size tradeoff |
| `/turing:transfer [--from]` | Cross-project knowledge transfer — find similar projects, surface what worked |
| `/turing:audit [--strict]` | Pre-submission methodology audit — data leakage, baselines, seeds, ablations, reproducibility |
| `/turing:sanity [--quick]` | Pre-training sanity checks — initial loss, single-batch overfit, gradient flow, output validation |
| `/turing:baseline [--methods]` | Automatic baseline generation — random, majority/mean, linear, k-NN |
| `/turing:leak [--deep]` | Targeted leakage detection — single-feature tests, correlation, train/test overlap |
| `/turing:xray [exp-id]` | Internal model diagnostics — gradient flow, dead neurons, weight distributions, tree analysis |
| `/turing:sensitivity [exp-id]` | Hyperparameter sensitivity — rank parameters by impact, detect non-monotonic responses |
| `/turing:calibrate [exp-id]` | Probability calibration — ECE/MCE, reliability diagrams, Platt/isotonic/temperature scaling |
| `/turing:feature [--method]` | Automated feature selection — multi-method consensus ranking, redundancy, interactions |
| `/turing:curriculum [exp-id]` | Training curriculum optimization — difficulty scoring, strategy comparison, mislabeled sample detection |
| `/turing:prune <exp-id>` | Weight pruning — magnitude/structured/lottery, sparsity sweep, knee point detection |
| `/turing:quantize <exp-id>` | Post-training quantization — FP16/INT8, accuracy-latency comparison |
| `/turing:merge <exp-ids...>` | Model merging — uniform/greedy soup, TIES, DARE, zero latency cost |
| `/turing:surgery <exp-id>` | Architecture modification — add/remove layer, widen/narrow, swap activation |
| `/turing:trend` | Long-term trend analysis — improvement velocity, family ROI, diminishing returns |
| `/turing:flashback` | Session context restoration — "where was I?" after days away |
| `/turing:archive` | Experiment lifecycle cleanup — compress old artifacts, summary index |
| `/turing:annotate <exp-id>` | Retrospective annotations — human notes and tags on experiments |
| `/turing:search <query>` | Natural language experiment search — text + structured filters |
| `/turing:template <action>` | Experiment template library — save/list/apply reusable configs |
| `/turing:replay <exp-id>` | Experiment replay — re-run old approach with current infrastructure |
| `/turing:cite <action>` | Citation & attribution manager — track papers, audit missing citations, generate BibTeX |
| `/turing:present [--figures]` | Presentation figures — training curves, comparisons, ablation, Pareto, sensitivity |
| `/turing:changelog [--audience]` | Model changelog — version-grouped improvements for technical or stakeholder audiences |
| `/turing:onboard [--audience]` | Project onboarding — walkthrough for new collaborators |
| `/turing:share <exp-ids...>` | Experiment packaging — portable archive with manifest |
| `/turing:review [--venue]` | Peer review simulation — weaknesses, fix commands, score |

And for fully hands-off operation:

```
/loop 5m /turing:train
```

The agent trains, evaluates, keeps improvements, discards regressions, detects convergence, and stops. You come back to a briefing.

## The Agent Architecture

Two agents with a strict capability boundary:

| Agent | Tools | Role | Turns |
|-------|-------|------|-------|
| **@ml-researcher** | Read, Write, Edit, Bash (whitelisted), Grep, Glob | Modifies `train.py` and `config.yaml`. Runs experiments. | 200 |
| **@ml-evaluator** | Read, Bash (whitelisted), Grep, Glob | Reads results. Analyzes trends. Cannot modify code. | 50 |

The evaluator's read-only constraint is not a limitation — it is a feature. An analyst who cannot act on their observations makes more trustworthy observations.

## The Anti-Cheating Stack

Research on autonomous ML agents has documented a recurring problem: [agents learn to game their own metrics](https://suzuke.github.io/blog/posts/ai-cheating-experiments/). Given a number to push up and a code editor, the agent finds the shortest path to a high number — even if that path subverts the entire purpose of the experiment. This is not theoretical. It has been observed in practice.

Turing implements six defense layers, informed by the [autocrucible](https://github.com/suzuke/autocrucible) project and documented failure modes from [karpathy/autoresearch#322](https://github.com/karpathy/autoresearch/discussions/322):

```
┌─────────────────────────────────────────────────┐
│  LAYER 1: Architectural Separation               │
│  Hypothesis space vs measurement apparatus        │
├─────────────────────────────────────────────────┤
│  LAYER 2: Hidden File Tier                        │
│  evaluate.py invisible to agent                   │
├─────────────────────────────────────────────────┤
│  LAYER 3: Behavioral Probes                       │
│  Training time, model size, prediction diversity   │
├─────────────────────────────────────────────────┤
│  LAYER 4: Statistical Validation                  │
│  Multi-run evaluation, CV check, median           │
├─────────────────────────────────────────────────┤
│  LAYER 5: Tool Restriction                        │
│  Whitelisted Bash commands only                   │
├─────────────────────────────────────────────────┤
│  LAYER 6: Diff-Based History                      │
│  Show actual changes, not agent descriptions      │
└─────────────────────────────────────────────────┘
```

The core insight from the research: **every prompt-based rule got worked around; every code-based rule held.** Turing's guardrails are structural, not conversational.

## Convergence Detection

When to stop flipping coins in this corner of the room:

```yaml
convergence:
  patience: 3                    # Consecutive non-improvements before stopping
  improvement_threshold: 0.005   # 0.5% relative improvement required
```

After N experiments with no meaningful improvement, the agent stops and reports what it found. The human then decides: is this good enough, or should we point the agent at a different region?

For noisy metrics, `/turing:validate` runs the pipeline multiple times and measures variance. If the coefficient of variation exceeds 5%, it auto-configures multi-run evaluation so the agent can't be rewarded for lucky single runs.

## Statistical Rigor

> *"Stop publishing lucky seeds. Start publishing distributions."*

Before claiming a result, run a seed study:

```
/turing:seed              # 5 seeds on best experiment
/turing:seed --quick      # 3 seeds for fast check
/turing:seed 10           # 10 seeds for thorough study
```

This runs the same experiment across multiple random seeds and reports mean +/- std with 95% confidence intervals. If the coefficient of variation exceeds 5%, the result is flagged as **seed-sensitive** — meaning you should report the distribution, not a single number.

To verify an experiment can be reproduced:

```
/turing:reproduce exp-042                  # Default: 3 runs, 2% tolerance
/turing:reproduce exp-042 --strict         # Exact match required
/turing:reproduce exp-042 --tolerance 0.05 # Custom tolerance
```

This re-runs the experiment from the logged config and checks that metrics fall within tolerance. It also detects environment drift — if library versions have changed since the original run, you'll know before a reviewer tells you.

Seed study results automatically appear in `/turing:brief` and `/turing:card`.

## Tree-Search Hypothesis Exploration

> *"The learned coin-flipper weaves through the quadrillion-coin room with a preternatural air."*

Sometimes the best experiment to try next isn't obvious from the literature or the agent's memory. `/turing:explore` uses [TreeQuest](https://github.com/SakanaAI/treequest)'s AB-MCTS (Adaptive Branching Monte Carlo Tree Search) to search the space of experiment *ideas* as a tree, scored by the critique engine (novelty x feasibility x impact).

```
/turing:explore                         # Run MCTS over hypothesis space
/turing:explore --strategy greedy       # Greedy fallback (no TreeQuest needed)
/turing:explore --iterations 50 --top 8 # Deeper search, more results
/turing:suggest --strategy treequest    # Same thing via suggest
```

How it works:

```
         Seeds                    MCTS expands best-scoring branches
           │
    ┌──────┼──────┐               Each node is a hypothesis scored by:
    ▼      ▼      ▼                 - Novelty (vs experiment history)
  LightGBM Reg  Features            - Feasibility (hardware, deps)
    │       │      │                - Expected impact (type success rate)
    ▼       ▼      ▼
  +dart   +L1   +poly             Top-K results queued as hypotheses
    │              │              for the next /turing:train run
    ▼              ▼
  +subsamp      +target-enc
```

Unlike `/turing:suggest` (which searches the web for papers), `/turing:explore` searches the space of *refinement chains* — combinations and sequences of modifications that score well together. It discovers non-obvious experiment strategies that independent suggestions cannot find.

Falls back to greedy best-first search when TreeQuest is not installed.

## Cost-Performance Frontier

> *"This model is 2% better but takes 10x longer to train. Is that worth it?"*

The briefing now surfaces [Pareto-optimal](https://en.wikipedia.org/wiki/Pareto_efficiency) experiments — the efficient set where no other experiment is both faster AND has a better metric. The cost report tells you the tradeoff in plain language:

```
Best metric: exp-012 (accuracy=0.893, 2400s)
Best efficiency: exp-003 (accuracy=0.871, 3s)
The 2.5% improvement costs 800x more compute.
```

Run `python scripts/cost_frontier.py` directly, or read the "Cost-Performance Analysis" section in `/turing:brief`.

## Model Cards

When it's time to ship, `/turing:card` generates a standardized model card documenting:
- Model type, framework, training time
- Performance metrics (all configured metrics)
- Training data source and split ratios
- Limitations (including overfit detection)
- Intended use and ethical considerations (user fills these in)
- Artifact contract version for production consumers

Inspired by [Google's Model Cards](https://arxiv.org/abs/1810.03993) and [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards).

## Installation

```bash
# Via npm (recommended)
npm install -g claude-turing
claude-turing install --global
claude-turing verify

# Via local path
claude plugin add /path/to/turing
```

### Quick Start

```bash
/turing:init                    # Scaffold project (answer 3 prompts)
/turing:train                   # Run experiment loop
/turing:brief                   # Read what happened
/turing:try "idea"              # Inject your taste
```

### Multiple Projects

```bash
/turing:init                    # Scaffold ml/sentiment
/turing:init                    # Scaffold ml/churn
/turing:train ml/sentiment      # Train in specific project
/turing:brief ml/churn          # Brief for specific project
cd ml/sentiment && /turing:train  # Auto-detects from cwd
```

Each project gets independent config, data, experiments, models, and agent memory.

## Architecture of Turing Itself

66 commands, 2 agents, 10 config files, 85 template scripts, model registry, artifact contract, cost-performance frontier, model cards, tree-search exploration, statistical rigor, experiment intelligence, performance profiling, smart checkpoints, production model export, literature integration, paper section drafting, experiment orchestration (queue + retry + fork), deep analysis (diff + watch + regress), model composition (ensemble + stitch + warm), scaling & efficiency (scale + budget + distill), meta-intelligence (transfer + audit), pre-training intelligence (sanity + baseline + leak), model debugging (xray + sensitivity + calibrate), feature & training intelligence (feature + curriculum), model surgery (prune + quantize + merge + surgery), experiment archaeology (trend + flashback + archive + annotate + search + template + replay), research communication (cite + present + changelog), collaboration (onboard + share + review), 16 ADRs. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full codemap.

```
turing/
├── commands/              62 skill files (core + taste-leverage + reporting + exploration + statistical rigor + experiment intelligence + performance + deployment + research workflow + orchestration + deep analysis + model composition + scaling & efficiency + meta-intelligence + pre-training intelligence + model debugging + feature & training intelligence + model surgery + experiment archaeology + research communication)
├── agents/                2 agents (researcher: read/write, evaluator: read-only)
├── config/                8 files (lifecycle, taxonomy, archetypes, novelty aliases)
├── templates/             Scaffolded into user projects by /turing:init
│   ├── prepare.py         Data loading (HIDDEN from agent)
│   ├── evaluate.py        Evaluation harness (HIDDEN from agent)
│   ├── train.py           Training code (AGENT-EDITABLE)
│   ├── model_contract.md  Artifact schema for production consumers
│   ├── model_registry.yaml  Available model architectures + hyperparams
│   └── scripts/           26 Python scripts (core loop + analysis + infra + tree search)
├── tests/                 338 tests (unit + integration + anti-pattern + manifest)
├── src/                   5 JS installer files (npm deployment)
├── bin/                   CLI entry points
└── docs/                  ARCHITECTURE.md + 16 ADRs
```

## Intellectual Heritage

- **[When Code Is Free](https://x.com/amytam01/status/2031072399731675269)** (Tam, 2026) — when execution cost approaches zero, the differentiator becomes research taste
- **[Autoresearch](https://github.com/karpathy/autoresearch)** (Karpathy, 2026) — ML experiment loops are mechanical enough to automate, with the constraint that evaluation must be immutable
- **[AutoCrucible](https://github.com/suzuke/autocrucible)** (suzuke, 2026) — autoresearch with guardrails: hidden evaluation, behavioral probes, tool restriction, stability validation
- **[Goodhart's Law](https://en.wikipedia.org/wiki/Goodhart%27s_law)** — "When a measure becomes a target, it ceases to be a good measure." The architectural justification for immutable, hidden evaluation
- **[Double-Blind Protocols](https://en.wikipedia.org/wiki/Blinded_experiment)** — the entity that evaluates must not be the entity that modifies
- **[Falsificationism](https://en.wikipedia.org/wiki/Falsifiability)** (Popper, 1934) — hypotheses gain credibility by surviving falsification, not by accumulating confirmations
- **[Principle of Least Privilege](https://en.wikipedia.org/wiki/Principle_of_least_privilege)** (Saltzer & Schroeder, 1975) — each agent has exactly the capabilities needed for its role
- **[Early Stopping](https://en.wikipedia.org/wiki/Early_stopping)** (Prechelt, 1998) — convergence detection as discrete early stopping
- **[Multi-Armed Bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit)** — the explore-exploit tradeoff
- **[TreeQuest](https://github.com/SakanaAI/treequest)** (Sakana AI, 2025) — AB-MCTS for inference-time scaling; repurposed here for hypothesis-space exploration
- **[Version Control as Lab Notebook](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004668)** (Ram, 2013) — git as a scientific record-keeping system
- **[Reproducibility Crisis](https://en.wikipedia.org/wiki/Replication_crisis)** — if the measurement can change between experiments, results are not reproducible

## License

MIT

---

*"In God we trust. All others must bring data."* — W. Edwards Deming

*"When code is free, research is all that matters."* — Amy Tam

*Turing flips the coins. You choose which ones.*
