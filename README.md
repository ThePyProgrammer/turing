# turing

*"The purpose of computing is insight, not numbers."* — Richard Hamming

---

An autonomous ML research harness for Claude Code. Turing implements the autoresearch pattern — an AI agent that iteratively trains, evaluates, and improves machine learning models through a structured experiment loop with convergence detection, immutable evaluation infrastructure, and safety guardrails.

The name references Alan Turing — the person who first asked whether machines could think, then built the framework for answering the question. Turing the plugin does what Turing the person formalized: it defines a computational process, executes it mechanically, and determines whether the result constitutes an improvement.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [snoglobe/helios](https://github.com/snoglobe/helios).

## Table of Contents

- [When Code Is Free, Research Is All That Matters](#when-code-is-free-research-is-all-that-matters)
- [The Problem Turing Solves](#the-problem-turing-solves)
- [Philosophical Foundations](#philosophical-foundations)
- [How Turing Works](#how-turing-works)
- [Commands](#commands)
- [The Agent Architecture](#the-agent-architecture)
- [The Immutable Evaluation Invariant](#the-immutable-evaluation-invariant)
- [Convergence Detection](#convergence-detection)
- [The Config DSL](#the-config-dsl)
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

Tam writes: *"Research taste is about how well you choose your coins: how well you choose which problems are worth working on at all."* Turing is the infrastructure that makes taste *leverageable*. A researcher with great taste but no execution bandwidth is a connoisseur who never eats. A researcher with great execution but poor taste is a high-throughput random number generator. The combination — human taste amplified by autonomous execution — is what the autoresearch pattern unlocks.

But there is a deeper problem, one that Tam identifies and that Turing is architecturally designed to address: *"Success makes taste legible, but it also makes it rigid. The researcher who was right about everything from 2018 to 2024 may be pattern-matching by 2026 and not know it yet."* Autonomous agents can break this rigidity. They don't have the ego investment in past hypotheses that makes human researchers unconsciously avoid experiments that might disprove their previous successes. An agent will try the experiment that the researcher's intuition says won't work — and sometimes the intuition is wrong. The [fastest-improving labs](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/) are the ones that have figured this out: taste selects the search space, but the search within that space must be exhaustive and unsentimental.

This is why Turing enforces immutable evaluation (the agent cannot change the definition of "heads"), structured logging (no coins are flipped without recording the outcome), and convergence detection (the agent knows when a region of the coin room has been exhausted). The infrastructure exists so that the human can think about which room to enter, not whether the flipping machinery is working correctly.

*When anyone can build for free, the differentiator is knowing what's worth building and whether it's buildable at all.* Turing handles the building. You bring the knowing.

## The Problem Turing Solves

> "An experiment is a question which science poses to Nature, and a measurement is the recording of Nature's answer." — Max Planck

The central activity of machine learning research is the experiment loop: change something, train, evaluate, decide, repeat. This loop is simultaneously the most important and the most tedious part of ML work. Researchers spend their days doing what is essentially a manual search over a high-dimensional space of model architectures, hyperparameters, feature transformations, and data preprocessing strategies.

The tragedy is not that this is slow — it is that the process is structurally unsound. When a human researcher modifies both the training code *and* the evaluation code in the same session, the experiment is no longer a controlled experiment. When experiment results are tracked in notebook cells rather than structured logs, reproducibility is aspirational. When a promising direction is abandoned because the researcher forgot what they tried three hours ago, the search is not even a search — it is a random walk with amnesia.

Turing does not replace the researcher's judgment. It replaces the researcher's *discipline* — or more precisely, it makes discipline the default rather than an act of willpower. The experiment loop is formalized. The evaluation harness is immutable. Every experiment is logged. Every code variant is preserved. Convergence is detected automatically. The researcher's role shifts from "person who types hyperparameters and reads loss curves" to "person who decides what hypotheses are worth testing" — from coin-flipper to coin-selector.

This is the same insight that drove Karpathy's autoresearch: the experiment loop is mechanical enough to automate, but the automation must be *safe* — it must not be able to game its own metrics. The agent flips the coins; the immutable evaluation tells you honestly whether they came up heads.

## Philosophical Foundations

### On Separating Hypothesis from Measurement

> "The first principle is that you must not fool yourself — and you are the easiest person to fool." — Richard Feynman

Turing is built on a specific epistemological claim: **the entity that generates hypotheses must not be the entity that evaluates them**. This is not a software engineering pattern — it is the methodological foundation of modern science, and it predates software by centuries.

In experimental physics, the [double-blind protocol](https://en.wikipedia.org/wiki/Blinded_experiment) ensures that the experimenter's expectations cannot influence the measurement. In ML, the equivalent risk is more insidious: an agent that can modify both `train.py` and `evaluate.py` can — deliberately or through optimization pressure — find metrics that look good but don't reflect genuine model improvement. It can overfit the evaluation function. It can inadvertently change what "accuracy" means between experiments, making comparison meaningless.

This is [Goodhart's Law](https://en.wikipedia.org/wiki/Goodhart%27s_law) made architectural: *"When a measure becomes a target, it ceases to be a good measure."* The only defense is to make the measure structurally immutable — not by policy, not by convention, but by removing the agent's ability to change it.

Turing enforces this separation architecturally:

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

This is the load-bearing architectural invariant. Everything else in Turing — the git discipline, the convergence detection, the experiment logging — is scaffolding around this single idea. The agent can flip any coin it wants, but it cannot change the definition of heads.

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

The key insight from [Karl Popper's philosophy of science](https://en.wikipedia.org/wiki/Karl_Popper#Philosophy_of_science) is that hypotheses gain credibility not by accumulating confirmations but by surviving attempts at falsification. In the Turing context, a model configuration gains credibility not because it produces good numbers, but because it produces *better* numbers than the previous best — measured by the same immutable yardstick.

### On Research Taste and Autonomous Execution

> *"Research taste is about how well you choose your coins: how well you choose which problems are worth working on at all."* — Amy Tam

There is a paradox at the heart of autonomous ML research: the parts of research that are hardest to automate are precisely the parts that matter most. Problem selection, hypothesis formation, knowing when a line of inquiry has become a dead end — these require what Tam calls *taste*, the accumulated judgment that comes from years of feeling faint differences in which problems are tractable, which results are meaningful, and which metrics actually capture what you care about.

Autoresearch does not solve this. Turing does not solve this. No one has solved this. But what autoresearch *does* solve is the complementary problem: given a well-selected hypothesis space, execute the search within it with superhuman discipline and throughput. The human provides the taste. The agent provides the tirelessness.

This is not a new division of labor — it is the oldest division of labor in science. The principal investigator decides which experiments to run. The lab technician runs them. The insight of the autoresearch pattern is that the "lab technician" for ML can be an LLM agent, and the "lab" can be a git repository with an immutable evaluation harness.

The danger — and Tam identifies this precisely — is confusing *speed* for *taste*. An agent that runs 126 experiments overnight has flipped many coins. But if all 126 experiments varied the learning rate between 0.001 and 0.1, the agent explored one dimension of a thousand-dimensional space very thoroughly and everything else not at all. The human must select the *dimensions* worth exploring. The agent explores them.

### On Experiment Tracking as Institutional Memory

> "Those who cannot remember the past are condemned to repeat it." — George Santayana

ML experiment tracking is typically an afterthought — notebook cells with outputs, a spreadsheet someone maintains for a week, or a Weights & Biases dashboard that nobody checks after the paper is submitted. Turing treats experiment history as a first-class artifact for a specific reason: **the agent needs to remember what it tried.**

An LLM agent without persistent memory is a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) — its next action depends only on its current state, not on the path that led there. This is catastrophically inefficient for optimization: the agent will re-try failed approaches, abandon promising directions, and fail to recognize when it has converged. It will keep flipping coins it has already flipped.

Turing addresses this with three complementary memory systems:

| System | Format | Purpose | Persistence |
|--------|--------|---------|-------------|
| **Experiment log** | `experiments/log.jsonl` | Structured record of every experiment (kept and discarded) | Append-only file |
| **Agent memory** | `.claude/agent-memory/ml-researcher/MEMORY.md` | Researcher's working notes: best results, failed approaches, promising directions | Updated per experiment |
| **Git history** | Experiment branches | Complete code variants for every experiment attempted | Permanent |

The JSONL log is the ground truth. The agent memory is the working set. The git history is the complete audit trail. Together, they ensure that no coin is flipped without recording the outcome, no promising direction is forgotten, and the agent can always reconstruct the path that led to the current best.

### On Convergence as a First-Class Concept

Most ML workflows don't detect convergence — the researcher just stops when they get bored, run out of compute budget, or decide the results are "good enough." This is the optimization equivalent of declaring victory by exhaustion.

Turing implements formal convergence detection because autonomous agents cannot get bored — and this is both their strength and their danger. An agent that never stops exploring wastes compute searching an exhausted region of the coin room. An agent that stops too early leaves heads on the table.

```
convergence:
  patience: 3                    # Consecutive non-improvements before stopping
  improvement_threshold: 0.005   # 0.5% relative improvement required
```

The convergence criterion is simple: if the last N experiments (where N = patience) each failed to improve the primary metric by at least the improvement threshold, the search space near the current optimum is exhausted. This is a discrete approximation of the [epsilon-greedy stopping rule](https://en.wikipedia.org/wiki/Multi-armed_bandit#Empirical_motivation) from bandit theory — stop exploring when the expected marginal value of exploration falls below the cost.

When convergence is detected, the agent stops and reports what it found. The human then applies taste: is this result good enough, or should we point the agent at a different region of the search space? The agent knows when a room is exhausted. The human decides which room to enter next.

The Stop hook (`scripts/check_convergence.py`) implements this check and returns exit code 2 to signal convergence to the Claude Code loop system, enabling fully hands-off operation:

```
/loop 5m /turing:train
```

## How Turing Works

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

After `/turing:init` scaffolds a project:

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
| `/turing:init` | Scaffold a new ML project with the full autoresearch harness — creates the separation between measurement apparatus and hypothesis space |

### Experiment Loop

| Command | Agent | Purpose |
|---------|-------|---------|
| `/turing:train [N]` | @ml-researcher | Run the autonomous experiment loop (optional max iterations) |
| `/turing:sweep` | @ml-researcher | Generate and run systematic hyperparameter sweep via cartesian product |

### Analysis

| Command | Agent | Purpose |
|---------|-------|---------|
| `/turing:status` | @ml-evaluator | Show experiment status, best model, convergence state |
| `/turing:compare <a> <b>` | @ml-evaluator | Side-by-side experiment comparison with causal analysis |

### Router

| Command | Purpose |
|---------|---------|
| `/turing` | Thin dispatcher — detects ML intent and routes to the appropriate sub-command |

## The Agent Architecture

Turing decomposes the experiment loop into two agents with complementary capabilities and a strict capability boundary:

```
┌───────────────────────────────────────────────────────┐
│                    TURING ROUTER                       │
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

The most important design decision in Turing is that `prepare.py` and `evaluate.py` are READ-ONLY. The agent cannot modify them. This is the mechanism by which experiment results remain comparable across iterations.

Consider the failure mode: an agent discovers that switching from XGBoost to a neural network improves accuracy from 0.82 to 0.87. But it also changed `evaluate.py` to use micro-averaged F1 instead of macro-averaged F1 because the neural network performs better on the majority class. Was the improvement real? You cannot know. The measurement changed between experiments. The comparison is meaningless. The coin landed on a different surface each time — you are not comparing flips, you are comparing surfaces.

Turing prevents this structurally. The evaluation function is fixed at project initialization. Every experiment is measured by the same yardstick. Comparisons are always valid. The agent can propose any hypothesis it wants about the training process — model architecture, hyperparameters, feature engineering, regularization — but it cannot change the definition of success.

This mirrors the structure of well-designed ML benchmarks: GLUE, ImageNet, and SQuAD define fixed evaluation protocols precisely because changing the evaluation mid-benchmark would invalidate all results. Turing applies the same principle at the project level.

## Convergence Detection

Convergence detection answers the question: "should I keep searching or has the current approach been exhausted?" In the coin-room metaphor: "have I flipped every coin in this corner of the room?"

Turing uses a patience-based criterion analogous to [early stopping](https://en.wikipedia.org/wiki/Early_stopping) in gradient descent:

1. After each experiment, compute the relative improvement over the prior best
2. If the improvement is below the threshold, increment the non-improvement counter
3. If the counter reaches `patience`, declare convergence and stop

```python
relative_improvement = (current - prior_best) / prior_best
if relative_improvement < improvement_threshold:
    non_improvements += 1
if non_improvements >= patience:
    STOP  # exit code 2 → human selects next search region
```

This is a conservative heuristic, not an optimal stopping rule. It can be fooled by plateaus that precede breakthroughs (like the loss plateaus observed in training large language models). But for the typical use case — iterative improvement of tabular ML models — it provides a reasonable balance between exploration and termination.

The `/loop` integration enables fully autonomous operation:

```
/loop 5m /turing:train
```

The Stop hook (`scripts/check_convergence.py`) runs after each training iteration and returns exit code 2 when convergence is detected, signaling the loop system to halt. The agent reads MEMORY.md at each iteration start, maintaining continuity across loop cycles.

## The Config DSL

Turing encodes domain knowledge as structured data rather than prose instructions. This is an application of the principle that **data outlives code** — when the experiment lifecycle rules or classification categories need to change, you edit a TOML file, not an agent prompt.

| Config | What it encodes | Why it matters |
|--------|----------------|----------------|
| `lifecycle.toml` | Experiment state machine (proposed -> running -> evaluating -> kept/discarded) | Agents validate transitions against data, not English |
| `taxonomy.toml` | Experiment types, failure modes, model families, severity levels | Classification is consistent across agents and sessions |
| `defaults.yaml` | Fallback hyperparameters, split ratios, convergence settings | Conservative starting points when project config is missing |

This is a lightweight [domain-specific language](https://en.wikipedia.org/wiki/Domain-specific_language) — not a general-purpose programming language, but a structured vocabulary for expressing ML experiment lifecycle concepts.

## Installation

```bash
# Via Claude Code plugin (recommended)
claude plugin add /path/to/turing

# Via npm
npm install -g claude-turing
claude-turing install --global

# Verify installation
claude-turing verify
```

Or from source:

```bash
cd ~/ThePyProgrammer/turing
npm install
npm link
claude-turing install --global
```

The installer deploys 6 commands, 2 agents, and 3 config files to `~/.claude/`, and inserts a managed section into CLAUDE.md with the command reference.

### Quick Start

```bash
# 1. Initialize a new ML project
/turing:init

# 2. Follow the prompts (project name, metric, data location)

# 3. Add your training data

# 4. Start the autonomous experiment loop
/turing:train

# 5. For fully hands-off training:
/loop 5m /turing:train
```

## Architecture of Turing Itself

```
turing/
├── commands/              6 skill files
│   ├── turing.md          Thin router (intent detection + dispatch)
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
│   ├── scripts/           Utility scripts (logging, metrics, sweeps, convergence)
│   └── tests/             Test fixture templates
├── tests/                 Plugin test suite (68 tests)
│   ├── test_evaluate.py   Measurement apparatus verification
│   ├── test_convergence.py Convergence detection verification
│   ├── test_log_experiment.py Append-only logging verification
│   ├── test_sweep.py      Sweep queue verification
│   ├── test_placeholder_verification.py  Scaffolding verification
│   └── test_anti_patterns.py  ADR invariant enforcement
├── src/                   Installation machinery
│   ├── install.js         Deploy to ~/.claude/
│   ├── verify.js          Check installation completeness
│   └── postinstall.js     npm postinstall hook
├── bin/                   CLI entry points
│   ├── cli.sh             Unified CLI (install/verify/init)
│   └── turing-init.sh     Direct project scaffolding
├── docs/                  Architecture documentation
│   ├── ARCHITECTURE.md    Bird's-eye codemap (matklad style)
│   └── adr/               16 Architecture Decision Records
└── .claude-plugin/        Plugin registration
    └── plugin.json        Plugin metadata
```

Turing practices what it preaches: the plugin itself maintains the separation between infrastructure (commands, agents, config, src) and templates (the project scaffolding that gets copied into user projects). The commands are thin dispatchers. The agents have single responsibilities. Domain knowledge lives in config, not in prompts. And the 68-test suite enforces that the ADR invariants hold in code, not just in documentation.

## Intellectual Heritage

Turing draws on several traditions, spanning experimental methodology, optimization theory, philosophy of science, and the emerging discourse on what matters when code becomes free:

- **[When Code Is Free](https://x.com/amytam01/status/2031072399731675269)** (Tam, 2026) — the thesis that when execution cost approaches zero, the differentiator becomes research taste: knowing which problems are worth solving. Turing is the infrastructure that makes taste leverageable by providing disciplined, tireless execution

- **[The Scientific Method](https://en.wikipedia.org/wiki/Scientific_method)** — the experiment loop is a formalization of hypothesize-experiment-measure-decide, the core pattern of empirical inquiry since [Francis Bacon](https://en.wikipedia.org/wiki/Novum_Organum) (1620)

- **[Autoresearch](https://github.com/karpathy/autoresearch)** (Karpathy, 2026) — the specific insight that ML experiment loops are mechanical enough to automate, with the critical constraint that evaluation must be immutable. 126 experiments overnight on a single GPU.

- **[Double-Blind Protocols](https://en.wikipedia.org/wiki/Blinded_experiment)** — the methodological principle that the measurer must not know the hypothesis, generalized here to: the entity that evaluates must not be the entity that modifies

- **[Goodhart's Law](https://en.wikipedia.org/wiki/Goodhart%27s_law)** — "When a measure becomes a target, it ceases to be a good measure." The architectural justification for immutable evaluation: if the agent could change the metric, the metric would stop measuring what you care about

- **[Falsificationism](https://en.wikipedia.org/wiki/Falsifiability)** (Popper, 1934) — hypotheses gain credibility by surviving falsification attempts, not by accumulating confirmations. Each experiment is a falsification test: "does this change actually improve the metric?"

- **[Finite State Machines](https://en.wikipedia.org/wiki/Finite-state_machine)** — the formal model underlying experiment lifecycle management. Transitions are validated against data, preventing invalid state sequences

- **[Principle of Least Privilege](https://en.wikipedia.org/wiki/Principle_of_least_privilege)** (Saltzer & Schroeder, 1975) — each agent has exactly the capabilities needed for its role. The evaluator cannot write; the researcher cannot modify evaluation

- **[Early Stopping](https://en.wikipedia.org/wiki/Early_stopping)** (Prechelt, 1998) — the convergence detection criterion is a discrete analogue of early stopping in gradient descent, adapted from continuous optimization to the discrete experiment loop

- **[Multi-Armed Bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit)** — the explore-exploit tradeoff underlying the decision to continue experimenting vs. declare convergence

- **[Domain-Specific Languages](https://en.wikipedia.org/wiki/Domain-specific_language)** (Fowler, 2010) — encoding domain concepts (experiment lifecycle, taxonomy) as structured data rather than general-purpose code

- **[Version Control as Lab Notebook](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004668)** (Ram, 2013) — git as a scientific record-keeping system, with experiment branches preserving every code variant

- **[Reproducibility Crisis](https://en.wikipedia.org/wiki/Replication_crisis)** — the broader scientific motivation for immutable evaluation: if the measurement can change between experiments, results are not reproducible

- **[ARCHITECTURE.md](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html)** (matklad, 2021) — the bird's-eye codemap approach to documentation, applied here to the plugin's own structure

- **[The Cost of Staying](https://amytam01.substack.com/p/the-cost-of-staying)** (Tam, 2026) — on the shift from execution capability to judgment as the scarce resource, and why the window for repositioning around taste-leveraged research is narrowing

## License

MIT

---

*"In God we trust. All others must bring data."* — W. Edwards Deming

*"When code is free, research is all that matters."* — Amy Tam

*Turing flips the coins. You choose which ones.*
