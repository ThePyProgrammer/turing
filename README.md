# turing

*The research assistant that can't fool itself.*

<p align="center">
  <img src="https://img.shields.io/badge/version-4.4.0-ffb74d?style=flat-square&labelColor=1a1a2e" alt="Version" /> 
  <img src="https://img.shields.io/badge/license-MIT-ff4d4d?style=flat-square&labelColor=1a1a2e" alt="License" /> 
  <img src="https://img.shields.io/badge/Claude_Code-plugin-ff4d4d?style=flat-square&labelColor=1a1a2e" alt="Claude Code" /> 
  <img src="https://img.shields.io/badge/Node.js-20%2B-ff4d4d?style=flat-square&labelColor=1a1a2e" alt="Node.js" />
</p>

---

> Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) proved that ML experiment loops are mechanical enough to automate, but the agent that generates hypotheses kept gaming the metrics.
> We believe the entity that runs experiments must not be the entity that evaluates them.
> Turing is an autonomous ML research harness that treats **separation of hypothesis from measurement** as a first class citizen.

---

<p align="center">
  <sub>Built with :heart: by <a href="https://github.com/ThePyProgrammer">@ThePyProgrammer</a></sub>
</p>

> [!NOTE]
> Turing is still in beta. Features may be broken or unpolished. Feedback is **always** welcome.

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

## Credits

Huge thanks to the work done by the autonomous ML research community. Turing would not exist without these projects and ideas:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — proved the experiment loop is mechanical enough to automate. Turing's core loop is a direct descendant.
- [snoglobe/helios](https://github.com/snoglobe/helios) — early inspiration for structured ML experiment harnesses.
- [suzuke/autocrucible](https://github.com/suzuke/autocrucible) — autoresearch with guardrails. Turing's six-layer anti-cheating stack is directly informed by autocrucible's documented failure modes.
- [Amy Tam](https://x.com/amytam01/status/2031072399731675269) — the "When Code Is Free" thesis. The entire taste-leverage interface is built around her insight that when execution cost approaches zero, research taste is the differentiator.
- [SakanaAI/treequest](https://github.com/SakanaAI/treequest) — AB-MCTS for inference-time scaling, repurposed in `/turing:explore` for hypothesis-space tree search.
- [Google's Model Cards](https://arxiv.org/abs/1810.03993) — inspiration for `/turing:card` and structured model documentation.
- [This article](https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents) by HumanLayer — a great starting point for thinking about harness engineering for AI agents.

The name references Alan Turing, who first asked whether machines could think, then built the framework for answering the question.

## Links

- [License](LICENSE) -- MIT

---

*"In God we trust. All others must bring data."* - W. Edwards Deming

*Turing flips the coins. You choose which ones.*
