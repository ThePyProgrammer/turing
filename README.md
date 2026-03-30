# helios

*"The purpose of computing is insight, not numbers."* — Richard Hamming

---

An autonomous ML research harness for Claude Code. Helios implements the autoresearch pattern — an AI agent that iteratively trains, evaluates, and improves machine learning models through a structured experiment loop with convergence detection, immutable evaluation infrastructure, and safety guardrails.

The name references the Greek sun god, but the real reference is to what Helios does: it watches everything, measures everything, and brings clarity to the landscape below.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [snoglobe/helios](https://github.com/snoglobe/helios).

## Table of Contents

- [The Problem Helios Solves](#the-problem-helios-solves)
- [Philosophical Foundations](#philosophical-foundations)
- [How Helios Works](#how-helios-works)
- [Commands](#commands)
- [The Agent Architecture](#the-agent-architecture)
- [The Immutable Evaluation Invariant](#the-immutable-evaluation-invariant)
- [Convergence Detection](#convergence-detection)
- [The Config DSL](#the-config-dsl)
- [Installation](#installation)
- [Architecture of Helios Itself](#architecture-of-helios-itself)
- [Intellectual Heritage](#intellectual-heritage)

## The Problem Helios Solves

> "An experiment is a question which science poses to Nature, and a measurement is the recording of Nature's answer." — Max Planck

The central activity of machine learning research is the experiment loop: change something, train, evaluate, decide, repeat. This loop is simultaneously the most important and the most tedious part of ML work. Researchers spend their days doing what is essentially a manual search over a high-dimensional space of model architectures, hyperparameters, feature transformations, and data preprocessing strategies.

The tragedy is not that this is slow — it is that the process is structurally unsound. When a human researcher modifies both the training code *and* the evaluation code in the same session, the experiment is no longer a controlled experiment. When experiment results are tracked in notebook cells rather than structured logs, reproducibility is aspirational. When a promising direction is abandoned because the researcher forgot what they tried three hours ago, the search is not even a search — it is a random walk with amnesia.

Helios does not replace the researcher's judgment. It replaces the researcher's *discipline* — or more precisely, it makes discipline the default rather than an act of willpower. The experiment loop is formalized. The evaluation harness is immutable. Every experiment is logged. Every code variant is preserved. Convergence is detected automatically. The researcher's role shifts from "person who types hyperparameters and reads loss curves" to "person who decides what hypotheses are worth testing."

This is the same insight that drove Karpathy's autoresearch: the experiment loop is mechanical enough to automate, but the automation must be *safe* — it must not be able to game its own metrics.

## Philosophical Foundations

### On Separating Hypothesis from Measurement

> "The first principle is that you must not fool yourself — and you are the easiest person to fool." — Richard Feynman

Helios is built on a specific epistemological claim: **the entity that generates hypotheses must not be the entity that evaluates them**. This is not a software engineering pattern — it is the methodological foundation of modern science, and it predates software by centuries.

In experimental physics, the [double-blind protocol](https://en.wikipedia.org/wiki/Blinded_experiment) ensures that the experimenter's expectations cannot influence the measurement. In ML, the equivalent risk is more insidious: an agent that can modify both `train.py` and `evaluate.py` can — deliberately or through optimization pressure — find metrics that look good but don't reflect genuine model improvement. It can overfit the evaluation function. It can inadvertently change what "accuracy" means between experiments, making comparison meaningless.

Helios enforces this separation architecturally:

```
┌──────────────────────────────────────────────────────┐
│                  HYPOTHESIS SPACE                     │
│              (agent can modify)                       │
│                                                      │
│    train.py          config.yaml                     │
│    ┌─────────┐       ┌──────────┐                    │
│    │ Model   │       │ Hyper-   │                    │
│    │ code    │       │ params   │                    │
│    └────┬────┘       └────┬─────┘                    │
│         │                 │                          │
├─────────┼─────────────────┼──────────────────────────┤
│         │    BOUNDARY      │                          │
│         ▼    (enforced)    ▼                          │
├──────────────────────────────────────────────────────┤
│                MEASUREMENT APPARATUS                  │
│               (agent cannot modify)                   │
│                                                      │
│    prepare.py        evaluate.py                     │
│    ┌─────────┐       ┌──────────┐                    │
│    │ Data    │       │ Metrics  │                    │
│    │ splits  │       │ compute  │                    │
│    └─────────┘       └──────────┘                    │
└──────────────────────────────────────────────────────┘
```

This is the load-bearing architectural invariant. Everything else in Helios — the git discipline, the convergence detection, the experiment logging — is scaffolding around this single idea.

### On the Scientific Method as a Control Loop

The autoresearch experiment loop is not an engineering heuristic. It is a formalization of the [hypothetico-deductive method](https://en.wikipedia.org/wiki/Hypothetico-deductive_model):

1. **Observe** — read recent experiment results and persistent memory
2. **Hypothesize** — propose a specific, falsifiable change with expected outcome
3. **Deduce** — predict what metrics should change and by how much
4. **Experiment** — modify the training code and execute
5. **Measure** — evaluate against the immutable harness
6. **Decide** — keep if the hypothesis was supported, revert if not
7. **Record** — log everything, including failures

Each iteration of this loop is an experiment in the scientific sense: a controlled manipulation of one variable (the hypothesis) with all other variables held constant (the evaluation infrastructure). The agent doesn't just iterate — it does science.

The key insight from [Karl Popper's philosophy of science](https://en.wikipedia.org/wiki/Karl_Popper#Philosophy_of_science) is that hypotheses gain credibility not by accumulating confirmations but by surviving attempts at falsification. In the Helios context, a model configuration gains credibility not because it produces good numbers, but because it produces *better* numbers than the previous best — measured by the same immutable yardstick.

### On Experiment Tracking as Institutional Memory

> "Those who cannot remember the past are condemned to repeat it." — George Santayana

ML experiment tracking is typically an afterthought — notebook cells with outputs, a spreadsheet someone maintains for a week, or a Weights & Biases dashboard that nobody checks after the paper is submitted. Helios treats experiment history as a first-class artifact for a specific reason: **the agent needs to remember what it tried.**

An LLM agent without persistent memory is a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) — its next action depends only on its current state, not on the path that led there. This is catastrophically inefficient for optimization: the agent will re-try failed approaches, abandon promising directions, and fail to recognize when it has converged.

Helios addresses this with three complementary memory systems:

| System | Format | Purpose | Persistence |
|--------|--------|---------|-------------|
| **Experiment log** | `experiments/log.jsonl` | Structured record of every experiment (kept and discarded) | Append-only file |
| **Agent memory** | `.claude/agent-memory/ml-researcher/MEMORY.md` | Researcher's working notes: best results, failed approaches, promising directions | Updated per experiment |
| **Git history** | Experiment branches | Complete code variants for every experiment attempted | Permanent |

The JSONL log is the ground truth. The agent memory is the working set. The git history is the complete audit trail. Together, they ensure that no information is lost and the agent can always reconstruct its reasoning.

### On Convergence as a First-Class Concept

Most ML workflows don't detect convergence — the researcher just stops when they get bored, run out of compute budget, or decide the results are "good enough." This is the optimization equivalent of declaring victory by exhaustion.

Helios implements formal convergence detection because autonomous agents cannot get bored:

```
convergence:
  patience: 3                    # Consecutive non-improvements before stopping
  improvement_threshold: 0.005   # 0.5% relative improvement required
```

The convergence criterion is simple: if the last N experiments (where N = patience) each failed to improve the primary metric by at least the improvement threshold, the search space near the current optimum is exhausted. This is a discrete approximation of the [epsilon-greedy stopping rule](https://en.wikipedia.org/wiki/Multi-armed_bandit#Empirical_motivation) from bandit theory — stop exploring when the expected marginal value of exploration falls below the cost.

The Stop hook (`scripts/stop-hook.sh`) implements this check and returns exit code 2 to signal convergence to the Claude Code loop system, enabling fully hands-off operation:

```
/loop 5m /helios:train
```

## How Helios Works

### The Experiment Lifecycle

Every experiment passes through a formal lifecycle encoded as a [finite state machine](https://en.wikipedia.org/wiki/Finite-state_machine) in `config/lifecycle.toml`:

```
                ┌───────────┐
                │  Proposed  │ ← hypothesis formed, code committed
                └─────┬─────┘
                      │
                ┌─────┴─────┐
                │  Running   │ ← training in progress
                └─────┬─────┘
                      │
               ┌──────┴──────┐
               │  Evaluating  │ ← metrics being computed
               └──────┬──────┘
                      │
           ┌──────────┼──────────┐
           ▼                     ▼
      ┌─────────┐          ┌───────────┐
      │  Kept   │          │ Discarded │
      │ (merge) │          │ (revert)  │
      └────┬────┘          └─────┬─────┘
           │                     │
           └──────────┬──────────┘
                      │
              next iteration...
                      │
              (or)    ▼
              ┌──────────────┐
              │  Converged   │ ← patience exhausted
              └──────────────┘
```

Transitions are validated against the state machine. You cannot evaluate a proposed experiment (you must run it first). You cannot keep an experiment that hasn't been evaluated. These rules are enforced by structure, not by prose.

### The Project Structure

After `/helios:init` scaffolds a project:

```
your-project/
  {{ML_DIR}}/
    ┌─ MEASUREMENT APPARATUS (READ-ONLY) ──────────────┐
    │  prepare.py           Data loading, splitting     │
    │  evaluate.py          Evaluation harness, metrics  │
    └───────────────────────────────────────────────────┘

    ┌─ HYPOTHESIS SPACE (AGENT-EDITABLE) ──────────────┐
    │  train.py             Training code               │
    │  config.yaml          Hyperparameters             │
    └───────────────────────────────────────────────────┘

    ┌─ INFRASTRUCTURE ─────────────────────────────────┐
    │  program.md           Agent instructions          │
    │  sweep_config.yaml    Sweep parameter ranges      │
    │  features/            Feature engineering          │
    │  scripts/             Logging, metrics, sweeps     │
    │  experiments/         Structured experiment logs    │
    │  models/              Model artifacts              │
    │  tests/               Test fixtures               │
    └───────────────────────────────────────────────────┘

  .claude/
    agent-memory/
      ml-researcher/
        MEMORY.md           Persistent agent memory
    settings.local.json     Hooks for auto-logging
```

## Commands

### Setup

| Command | Purpose |
|---------|---------|
| `/helios:init` | Scaffold a new ML project with the full autoresearch harness — creates the separation between measurement apparatus and hypothesis space |

### Experiment Loop

| Command | Agent | Purpose |
|---------|-------|---------|
| `/helios:train [N]` | @ml-researcher | Run the autonomous experiment loop (optional max iterations) |
| `/helios:sweep` | @ml-researcher | Generate and run systematic hyperparameter sweep via cartesian product |

### Analysis

| Command | Agent | Purpose |
|---------|-------|---------|
| `/helios:status` | @ml-evaluator | Show experiment status, best model, convergence state |
| `/helios:compare <a> <b>` | @ml-evaluator | Side-by-side experiment comparison with causal analysis |

### Router

| Command | Purpose |
|---------|---------|
| `/helios` | Thin dispatcher — detects ML intent and routes to the appropriate sub-command |

## The Agent Architecture

Helios decomposes the experiment loop into two agents with complementary capabilities and a strict capability boundary:

```
┌───────────────────────────────────────────────────────┐
│                    HELIOS ROUTER                       │
│             (thin dispatcher, ~40 lines)               │
└──────────┬────────────────────────────┬───────────────┘
           │                            │
           ▼                            ▼
    ┌──────────────┐            ┌──────────────┐
    │ ml-researcher │            │ ml-evaluator  │
    │              │            │              │
    │ Read, Write, │            │ Read, Bash,  │
    │ Edit, Bash,  │            │ Grep, Glob   │
    │ Grep, Glob   │            │              │
    │              │            │ NO Write     │
    │ Can modify   │            │ NO Edit      │
    │ train.py     │            │              │
    │              │            │ Read-only    │
    │ 200 turns    │            │ analysis     │
    │              │            │              │
    │ Has memory   │            │ 50 turns     │
    └──────┬───────┘            └──────┬───────┘
           │                            │
           ▼                            ▼
    ┌──────────────────────────────────────────┐
    │         SHARED INFRASTRUCTURE             │
    │                                          │
    │  config/defaults.yaml  ← fallback values │
    │  config/lifecycle.toml ← state machine   │
    │  config/taxonomy.toml  ← classifications │
    └──────────────────────────────────────────┘
```

The researcher has Write and Edit tools — it can modify `train.py` and `config.yaml`. The evaluator has only Read and Bash — it can run analysis scripts and read results but cannot change any code. This is a deliberate application of the [principle of least privilege](https://en.wikipedia.org/wiki/Principle_of_least_privilege): each agent has exactly the capabilities it needs and no more.

The evaluator's read-only constraint is not a limitation — it is a feature. In [quantum mechanics](https://en.wikipedia.org/wiki/Observer_effect_(physics)), observation changes the system. In ML experimentation, an evaluator that can change code might unconsciously bias its analysis toward changes it could make. By removing the ability to act, the evaluator's observations become more trustworthy.

## The Immutable Evaluation Invariant

> "I often say that when you can measure what you are speaking about, and express it in numbers, you know something about it; but when you cannot measure it, when you cannot express it in numbers, your knowledge is of a meagre and unsatisfactory kind." — Lord Kelvin

The most important design decision in Helios is that `prepare.py` and `evaluate.py` are READ-ONLY. The agent cannot modify them. This is the mechanism by which experiment results remain comparable across iterations.

Consider the failure mode: an agent discovers that switching from XGBoost to a neural network improves accuracy from 0.82 to 0.87. But it also changed `evaluate.py` to use micro-averaged F1 instead of macro-averaged F1 because the neural network performs better on the majority class. Was the improvement real? You cannot know. The measurement changed between experiments. The comparison is meaningless.

Helios prevents this structurally. The evaluation function is fixed at project initialization. Every experiment is measured by the same yardstick. Comparisons are always valid. The agent can propose any hypothesis it wants about the training process — model architecture, hyperparameters, feature engineering, regularization — but it cannot change the definition of success.

This mirrors the structure of well-designed ML benchmarks: GLUE, ImageNet, and SQuAD define fixed evaluation protocols precisely because changing the evaluation mid-benchmark would invalidate all results. Helios applies the same principle at the project level.

## Convergence Detection

Convergence detection answers the question: "should I keep searching or has the current approach been exhausted?"

Helios uses a patience-based criterion analogous to [early stopping](https://en.wikipedia.org/wiki/Early_stopping) in gradient descent:

1. After each experiment, compute the relative improvement over the prior best
2. If the improvement is below the threshold, increment the non-improvement counter
3. If the counter reaches `patience`, declare convergence and stop

```python
relative_improvement = (current - prior_best) / prior_best
if relative_improvement < improvement_threshold:
    non_improvements += 1
if non_improvements >= patience:
    STOP  # exit code 2
```

This is a conservative heuristic, not an optimal stopping rule. It can be fooled by plateaus that precede breakthroughs (like the loss plateaus observed in training large language models). But for the typical use case — iterative improvement of tabular ML models — it provides a reasonable balance between exploration and termination.

The `/loop` integration enables fully autonomous operation:

```
/loop 5m /helios:train
```

The Stop hook (`scripts/stop-hook.sh`) runs after each training iteration and returns exit code 2 when convergence is detected, signaling the loop system to halt. The agent reads MEMORY.md at each iteration start, maintaining continuity across loop cycles.

## The Config DSL

Helios encodes domain knowledge as structured data rather than prose instructions. This is an application of the principle that **data outlives code** — when the experiment lifecycle rules or classification categories need to change, you edit a TOML file, not an agent prompt.

| Config | What it encodes | Why it matters |
|--------|----------------|----------------|
| `lifecycle.toml` | Experiment state machine (proposed -> running -> evaluating -> kept/discarded) | Agents validate transitions against data, not English |
| `taxonomy.toml` | Experiment types, failure modes, model families, severity levels | Classification is consistent across agents and sessions |
| `defaults.yaml` | Fallback hyperparameters, split ratios, convergence settings | Conservative starting points when project config is missing |

This is a lightweight [domain-specific language](https://en.wikipedia.org/wiki/Domain-specific_language) — not a general-purpose programming language, but a structured vocabulary for expressing ML experiment lifecycle concepts.

## Installation

```bash
# Via Claude Code plugin (recommended)
claude plugin add /path/to/helios

# Via npm
npm install -g claude-helios
claude-helios install --global

# Verify installation
claude-helios verify
```

Or from source:

```bash
cd ~/ThePyProgrammer/helios
npm install
npm link
claude-helios install --global
```

The installer deploys 6 commands, 2 agents, and 3 config files to `~/.claude/`, and inserts a managed section into CLAUDE.md with the command reference.

### Quick Start

```bash
# 1. Initialize a new ML project
/helios:init

# 2. Follow the prompts (project name, metric, data location)

# 3. Add your training data

# 4. Start the autonomous experiment loop
/helios:train

# 5. For fully hands-off training:
/loop 5m /helios:train
```

## Architecture of Helios Itself

```
helios/
├── commands/              6 skill files
│   ├── helios.md          Thin router (intent detection + dispatch)
│   ├── init.md            Project scaffolding
│   ├── train.md           Autonomous experiment loop
│   ├── status.md          Experiment status display
│   ├── compare.md         Side-by-side run comparison
│   ├── sweep.md           Hyperparameter sweep
│   └── rules/
│       └── loop-protocol.md   Safety constraints for the experiment loop
├── agents/                2 agent definitions
│   ├── ml-researcher.md   Autonomous researcher (Read/Write/Edit/Bash)
│   └── ml-evaluator.md    Read-only analyst (Read/Bash only)
├── config/                Domain-specific language
│   ├── defaults.yaml      Fallback hyperparameters and settings
│   ├── lifecycle.toml     Experiment state machine
│   └── taxonomy.toml      Classification system
├── templates/             Project scaffolding templates
│   ├── prepare.py         Data loading/splitting (READ-ONLY template)
│   ├── evaluate.py        Evaluation harness (READ-ONLY template)
│   ├── train.py           Training code (AGENT-EDITABLE template)
│   ├── config.yaml        Default experiment configuration
│   ├── sweep_config.yaml  Hyperparameter sweep ranges
│   ├── program.md         Agent protocol instructions
│   ├── README.md          Per-project README template
│   ├── MEMORY.md          Agent memory template
│   ├── requirements.txt   Python dependencies
│   ├── pyproject.toml     Project/pytest config
│   ├── features/          Feature engineering templates
│   ├── scripts/           Utility scripts (logging, metrics, sweeps)
│   └── tests/             Test fixture templates
├── src/                   Installation machinery
│   ├── install.js         Deploy to ~/.claude/
│   ├── verify.js          Check installation completeness
│   └── postinstall.js     npm postinstall hook
├── bin/                   CLI entry points
│   ├── cli.sh             Unified CLI (install/verify/init)
│   └── helios-init.sh     Direct project scaffolding
└── .claude-plugin/        Plugin registration
    └── plugin.json        Plugin metadata
```

Helios practices what it preaches: the plugin itself maintains the separation between infrastructure (commands, agents, config, src) and templates (the project scaffolding that gets copied into user projects). The commands are thin dispatchers. The agents have single responsibilities. Domain knowledge lives in config, not in prompts.

## Intellectual Heritage

Helios draws on several traditions, spanning experimental methodology, optimization theory, and software architecture:

- **[The Scientific Method](https://en.wikipedia.org/wiki/Scientific_method)** — the experiment loop is a formalization of hypothesize-experiment-measure-decide, the core pattern of empirical inquiry since [Francis Bacon](https://en.wikipedia.org/wiki/Novum_Organum) (1620)

- **[Autoresearch](https://github.com/karpathy/autoresearch)** (Karpathy) — the specific insight that ML experiment loops are mechanical enough to automate, with the critical constraint that evaluation must be immutable

- **[Double-Blind Protocols](https://en.wikipedia.org/wiki/Blinded_experiment)** — the methodological principle that the measurer must not know the hypothesis, generalized here to: the entity that evaluates must not be the entity that modifies

- **[Falsificationism](https://en.wikipedia.org/wiki/Falsifiability)** (Popper, 1934) — hypotheses gain credibility by surviving falsification attempts, not by accumulating confirmations. Each experiment is a falsification test: "does this change actually improve the metric?"

- **[Finite State Machines](https://en.wikipedia.org/wiki/Finite-state_machine)** — the formal model underlying experiment lifecycle management. Transitions are validated against data, preventing invalid state sequences

- **[Principle of Least Privilege](https://en.wikipedia.org/wiki/Principle_of_least_privilege)** (Saltzer & Schroeder, 1975) — each agent has exactly the capabilities needed for its role. The evaluator cannot write; the researcher cannot modify evaluation

- **[Early Stopping](https://en.wikipedia.org/wiki/Early_stopping)** — the convergence detection criterion is a discrete analogue of early stopping in gradient descent, adapted from continuous optimization to the discrete experiment loop

- **[Multi-Armed Bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit)** — the explore-exploit tradeoff underlying the decision to continue experimenting vs. declare convergence

- **[Domain-Specific Languages](https://en.wikipedia.org/wiki/Domain-specific_language)** (Fowler, 2010) — encoding domain concepts (experiment lifecycle, taxonomy) as structured data rather than general-purpose code

- **[Version Control as Lab Notebook](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004668)** (Ram, 2013) — git as a scientific record-keeping system, with experiment branches preserving every code variant

- **[Reproducibility Crisis](https://en.wikipedia.org/wiki/Replication_crisis)** — the broader scientific motivation for immutable evaluation: if the measurement can change between experiments, results are not reproducible

- **[ARCHITECTURE.md](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html)** (matklad, 2021) — the bird's-eye codemap approach to documentation, applied here to the plugin's own structure

## License

MIT

---

*"In God we trust. All others must bring data."* — W. Edwards Deming

*Helios brings the data.*
