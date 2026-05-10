# turing

*The research assistant that can't fool itself.*

<p align="center">
  <img src="https://img.shields.io/badge/version-4.8.1-ffb74d?style=flat-square&labelColor=1a1a2e" alt="Version" />
  <img src="https://img.shields.io/badge/license-MIT-ff4d4d?style=flat-square&labelColor=1a1a2e" alt="License" /> 
  <img src="https://img.shields.io/badge/Claude_Code-plugin-ff4d4d?style=flat-square&labelColor=1a1a2e" alt="Claude Code" /> 
  <img src="https://img.shields.io/badge/Node.js-20%2B-ff4d4d?style=flat-square&labelColor=1a1a2e" alt="Node.js" />
</p>

A Claude Code plugin that runs autonomous ML experiment loops, named after the man who first asked whether machines could think. Two agents enforce a strict separation: one writes code, one scores it, and neither can see the other's work. Immutable evaluation, anti-cheating guardrails, and structured hypothesis tracking make sure the results stay honest. [When code is free, research is all that matters](https://x.com/amytam01/status/2031072399731675269). You bring the research taste; Turing handles the rest.

- **Separation:** the agent modifies `train.py`; it cannot see or touch `evaluate.py`
- **Memory:** every hypothesis registered, every experiment logged, every variant preserved
- **Convergence:** automatic detection of diminishing returns; the agent stops when it should
- **Taste:** you inject ideas with `/turing:try`, read results with `/turing:brief`

> [!NOTE]
> Turing is in active development. Some features are rough around the edges. [Issues and feedback welcome.](https://github.com/ThePyProgrammer/turing/issues)

## Install

```bash
npm install -g claude-turing && claude-turing install --global && claude-turing verify
```

## The Taste-Leverage Loop

You have taste: the accumulated judgment about which problems are tractable, which metrics matter, and which directions are dead ends. Turing has leverage: the discipline to run experiments without fatigue, track every result without amnesia, and measure without contamination.

The interface is two verbs:

```
/turing:try switch to LightGBM        Your taste → the agent
/turing:brief --deep                   The agent's results → you
```

Everything in between (experiment logging, convergence detection, hypothesis tracking, statistical validation, anti-cheating guardrails) is infrastructure connecting those two endpoints. You think about *what* to try. Turing handles *how* to try it.

### What a Session Looks Like

```
/turing:init                          Scaffold a new ML project
/turing:train                         Agent runs 5-10 experiments autonomously
/turing:brief                         Campaign summary: what improved, what's exhausted
/turing:try "add polynomial features" Inject your next idea
/turing:train                         Agent follows your lead
```

For fully hands-off operation:

```
/loop 5m /turing:train
```

The agent trains, evaluates, keeps improvements, discards regressions, detects convergence, and stops. You come back to a briefing.

## How It Works

**The experiment loop.** Every iteration: observe metrics, hypothesize (human ideas first), edit `train.py`, commit to a git branch, train, measure (agent can't see how), keep or revert, log, check convergence.

**Hypothesis tracking.** Every idea flows through `hypotheses.yaml` with a novelty guard that blocks duplicates. Detail files record architecture, hyperparameters, expected outcome, actual result, and lineage. Nothing is forgotten between sessions.

**Anti-cheating stack.** Six structural layers, not prompt-based rules. The agent cannot see `evaluate.py`, cannot discover scoring formulas, cannot reverse-engineer fixed seeds. It knows the metric name, the direction, and the result. That's it. Research on autonomous ML agents shows that [every prompt-based rule got worked around; every code-based rule held](https://github.com/karpathy/autoresearch/discussions/322).

**Two agents, strict boundary.** `@ml-researcher` (Read/Write/Edit/Bash) modifies code and runs experiments. `@ml-evaluator` (Read/Bash only) analyzes results. An analyst who cannot act on their observations makes more trustworthy observations.

**Convergence detection.** After N consecutive non-improvements (default 3, configurable), the agent stops. For noisy metrics, `/turing:validate` auto-configures multi-run evaluation so the agent can't be rewarded for lucky single runs.

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

See [the command reference](docs/commands/index.md) for all 74 commands.

## Credits

Turing would not exist without these projects, ideas, and intellectual traditions:

**Projects**

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch): proved the experiment loop is mechanical enough to automate. Turing's core loop is a direct descendant.
- [snoglobe/helios](https://github.com/snoglobe/helios): early inspiration for structured ML experiment harnesses.
- [suzuke/autocrucible](https://github.com/suzuke/autocrucible): autoresearch with guardrails. Turing's six-layer anti-cheating stack is directly informed by autocrucible's documented failure modes.
- [SakanaAI/treequest](https://github.com/SakanaAI/treequest): AB-MCTS for inference-time scaling, repurposed in `/turing:explore` for hypothesis-space tree search.
- [Google's Model Cards](https://arxiv.org/abs/1810.03993): inspiration for `/turing:card` and structured model documentation.

**Ideas**

- ["When Code Is Free, Research Is All That Matters"](https://x.com/amytam01/status/2031072399731675269) (Tam, 2026): when execution cost approaches zero, research taste is the differentiator. The entire taste-leverage interface is built around this insight.
- "The first principle is that you must not fool yourself, and you are the easiest person to fool." (Feynman) The separation of hypothesis from measurement is Turing's answer to Feynman's first principle.
- [*The Tacit Dimension*](https://en.wikipedia.org/wiki/The_Tacit_Dimension) (Polanyi, 1966): "We can know more than we can tell." Research taste is tacit knowledge that resists formalization, which is why the human stays in the loop.
- [The context of discovery vs. the context of justification](https://en.wikipedia.org/wiki/Context_of_justification) (Reichenbach, 1938; Popper, 1959): hypothesis generation is creative and non-logical; only testing admits of formal treatment. Turing is a justification machine. You provide the discovery.
- [*The Structure of Scientific Revolutions*](https://en.wikipedia.org/wiki/The_Structure_of_Scientific_Revolutions) (Kuhn, 1962): the risk of efficiently optimizing within a degenerating paradigm. Convergence detection is Turing's partial answer; knowing when to leave the corner is still yours.
- [Goodhart's Law](https://en.wikipedia.org/wiki/Goodhart%27s_law) (1975) and [Campbell's Law](https://en.wikipedia.org/wiki/Campbell%27s_law) (1979): when a measure becomes a target, it ceases to be a good measure. The entire anti-cheating stack exists because these laws activate the moment an agent evaluates itself.
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) (Amodei et al., 2016) and [DeepMind's specification gaming catalogue](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/): documented that reward hacking is not a theoretical risk but an observed behavior of capable optimizers.
- [NIST CAISI](https://www.nist.gov/artificial-intelligence/executive-order-safe-secure-and-trustworthy-artificial-intelligence) (2025): documented systematic cheating by frontier models (downloading solutions, commenting out assertions, crashing servers). Every prompt-based rule got worked around; every code-based rule held.


## Links

- [License](LICENSE) (MIT)

---

*"In God we trust. All others must bring data."* - W. Edwards Deming

*Turing flips the coins. You choose which ones.*
