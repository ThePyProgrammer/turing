# turing

*The research assistant that can't fool itself.*

---

> Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) proved that ML experiment loops are mechanical enough to automate.
> But the agent that runs experiments must not be the agent that evaluates them.
> Turing is an autonomous ML research harness that enforces this separation structurally, not conversationally.

---

<p align="center">
  <sub>Built with :heart: by <a href="https://github.com/prannayag">@prannayag</a></sub>
</p>

**Turing is a Claude Code plugin that runs autonomous ML experiment loops with immutable evaluation, anti-cheating guardrails, and structured hypothesis tracking.**

- **Separation** -- the agent modifies `train.py`; it cannot see or touch `evaluate.py`
- **Memory** -- every hypothesis registered, every experiment logged, every variant preserved
- **Convergence** -- automatic detection of diminishing returns; the agent stops when it should
- **Taste interface** -- you inject ideas with `/turing:try`, read results with `/turing:brief`

## Install

```bash
# Via npm
npm install -g claude-turing
claude-turing install --global
claude-turing verify

# Via local path
claude plugin add /path/to/turing
```

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

Everything else (experiment logging, convergence detection, hypothesis tracking, statistical validation, anti-cheating guardrails) happens automatically. You think about *what* to try. Turing handles *how* to try it.

And for fully hands-off operation:

```
/loop 5m /turing:train
```

## How It Works

Turing structures ML research around the **taste-leverage loop**: you bring research taste (which hypotheses are worth testing), the agent brings discipline (running them without fatigue, amnesia, or measurement contamination).

**The experiment loop.** Every iteration: observe metrics, hypothesize (human ideas first), edit `train.py`, commit to a git branch, train, measure (agent can't see how), keep or revert, log, check convergence.

**Hypothesis tracking.** Every idea, human-injected or agent-generated, flows through `hypotheses.yaml`. A novelty guard blocks duplicates. Detail files record architecture, hyperparameters, expected outcome, actual result, and lineage. Nothing is forgotten.

**Anti-cheating stack.** Six layers: architectural separation (hypothesis vs measurement), hidden evaluation (`evaluate.py` invisible to agent), behavioral probes, statistical validation, tool restriction (whitelisted Bash only), and diff-based history. Every prompt-based rule got worked around in prior research; every code-based rule held. Turing's guardrails are structural.

**Two agents.** `@ml-researcher` (Read/Write/Edit/Bash) modifies code and runs experiments. `@ml-evaluator` (Read/Bash only) analyzes results. The evaluator's read-only constraint makes its observations more trustworthy.

**Convergence detection.** After N consecutive non-improvements (default 3, configurable), the agent stops and reports. For noisy metrics, `/turing:validate` auto-configures multi-run evaluation.

## Command Reference

### Core Loop

| Command | What it does |
|---------|-------------|
| `/turing:init [--plan]` | Scaffold a new ML project. `--plan` for literature-grounded research plan. |
| `/turing:train [path] [N]` | Run the experiment loop. Auto-detects project from cwd. |
| `/turing:status` | Quick status: best model, convergence state |
| `/turing:compare <a> <b>` | Side-by-side experiment comparison |
| `/turing:sweep` | Systematic hyperparameter sweep |

### Taste-Leverage Interface

| Command | What it does |
|---------|-------------|
| `/turing:try <hypothesis>` | Inject a hypothesis (free text or archetype) |
| `/turing:brief [--deep]` | Research briefing with literature-grounded suggestions |
| `/turing:suggest` | Literature-grounded model architecture suggestions |
| `/turing:explore` | AB-MCTS tree search over hypothesis space |
| `/turing:design <hyp-id>` | Generate structured experiment design |
| `/turing:mode <mode>` | Set research strategy (explore/exploit/replicate) |

### Validation & Statistical Rigor

| Command | What it does |
|---------|-------------|
| `/turing:validate [--auto]` | Metric stability check, auto-configure multi-run |
| `/turing:seed [N]` | Multi-seed study: mean/std/CI, flag seed-sensitive results |
| `/turing:reproduce <exp-id>` | Reproducibility verification with tolerance checking |
| `/turing:sanity` | Pre-training sanity checks |
| `/turing:baseline` | Automatic baseline generation |
| `/turing:leak` | Targeted data leakage detection |
| `/turing:audit` | Pre-submission methodology audit |

See [COMMANDS.md](docs/commands/index.md) for the full reference covering all 74 commands.

## Architecture

74 commands, 2 agents, 10 config files, 93 template scripts, model registry, artifact contract, cost-performance frontier, model cards, tree-search exploration, statistical rigor, experiment intelligence, performance profiling, smart checkpoints, production model export, literature integration, paper section drafting, experiment orchestration, deep analysis, model composition, scaling & efficiency, meta-intelligence, pre-training intelligence, model debugging, feature & training intelligence, model surgery, experiment archaeology, research communication, what-if analysis, model lifecycle, operational intelligence, 16 ADRs.

```
turing/
├── commands/              70 skill files
├── agents/                2 agents (researcher: read/write, evaluator: read-only)
├── config/                8 files (lifecycle, taxonomy, archetypes, novelty aliases)
├── templates/             Scaffolded into user projects by /turing:init
│   ├── prepare.py         Data loading (HIDDEN from agent)
│   ├── evaluate.py        Evaluation harness (HIDDEN from agent)
│   ├── train.py           Training code (AGENT-EDITABLE)
│   ├── model_contract.md  Artifact schema for production consumers
│   ├── model_registry.yaml  Available model architectures + hyperparams
│   └── scripts/           26 Python scripts
├── tests/                 338 tests
├── src/                   5 JS installer files (npm deployment)
├── bin/                   CLI entry points
└── docs/                  ARCHITECTURE.md + 16 ADRs + philosophy
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full codemap.

## Credits

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [snoglobe/helios](https://github.com/snoglobe/helios). Anti-cheating stack informed by [autocrucible](https://github.com/suzuke/autocrucible). The name references Alan Turing, who first asked whether machines could think, then built the framework for answering the question.

## Documentation

-> [The Taste-Leverage Thesis](docs/philosophy/index.md) -- why Turing exists, the philosophical foundations, and what it means when code is free

-> [Technical Documentation](docs/ARCHITECTURE.md) -- architectural narrative, system design rationale

-> [docs/](docs/) -- commands, philosophy, architecture, getting started, and more

## Links

-> [License](LICENSE) -- MIT

---

*"In God we trust. All others must bring data."* - W. Edwards Deming

*Turing flips the coins. You choose which ones.*
