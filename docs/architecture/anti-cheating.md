---
title: "The Anti-Cheating Stack"
description: "Six defense layers that prevent autonomous agents from gaming their own metrics. The lesson from autocrucible: every prompt-based rule got worked around; every code-based rule held."
---

# The Anti-Cheating Stack

## The Problem

When you give an autonomous agent a metric to optimize and the tools to modify its own training code, the agent will find ways to improve the metric that do not improve the model. This is not hypothetical -- it is the expected behavior of any sufficiently capable optimizer.

The [autocrucible](https://github.com/jxnl/autocrucible) project documented this pattern: agents that could read their evaluation code learned to exploit fixed seeds, memorize test distributions, and reverse-engineer scoring functions. The metric went up. The model did not get better.

The core insight from that work:

> **Every prompt-based rule got worked around. Every code-based rule held.**

Turing's anti-cheating stack is built on this insight. The defenses are architectural, not instructional.

## The Six Layers

```
┌─────────────────────────────────────────────────────────┐
│  Layer 6: DIFF-BASED HISTORY                            │
│  Git diffs are ground truth. Agent reads actual code    │
│  changes, not its own descriptions of what it changed.  │
├─────────────────────────────────────────────────────────┤
│  Layer 5: TOOL RESTRICTION                              │
│  Bash access whitelisted to specific commands.          │
│  No cat, head, tail, curl, wget, or arbitrary shell.    │
├─────────────────────────────────────────────────────────┤
│  Layer 4: STATISTICAL VALIDATION                        │
│  Multi-run evaluation, CV checks, seed studies.         │
│  Single lucky results cannot survive validation.        │
├─────────────────────────────────────────────────────────┤
│  Layer 3: BEHAVIORAL PROBES                             │
│  Training time, model size, prediction diversity.       │
│  Detects degenerate solutions that game metrics.        │
├─────────────────────────────────────────────────────────┤
│  Layer 2: HIDDEN FILE TIER                              │
│  evaluate.py is invisible to the agent.                 │
│  Cannot read, reference, or access evaluation code.     │
├─────────────────────────────────────────────────────────┤
│  Layer 1: ARCHITECTURAL SEPARATION                      │
│  Hypothesis space vs measurement apparatus.             │
│  Three-tier access model enforced at the tool level.    │
└─────────────────────────────────────────────────────────┘
```

## Layer 1: Architectural Separation

The foundational defense. Files are divided into three tiers with access enforced at the tool level, not by prompt instructions:

| Tier | Files | Agent Access |
|------|-------|-------------|
| **Hidden** | `evaluate.py` | NONE -- invisible |
| **Read-only** | `prepare.py`, `features/featurizers.py` | READ-ONLY |
| **Hypothesis** | `train.py`, `config.yaml` | READ-WRITE |

The agent modifies the hypothesis space. The measurement apparatus is immutable. If the agent cannot change how results are scored, it cannot game the scoring.

!!! info "Why this works"
    Prompt instructions like "do not modify evaluate.py" are suggestions. Tool-level access restrictions are constraints. The agent does not have a Write tool call that can target evaluate.py -- the capability does not exist, so there is nothing to work around.

## Layer 2: Hidden File Tier

`evaluate.py` is not just read-only -- it is invisible. The agent cannot read it, cannot reference it in conversation, and has no tool that would return its contents.

This prevents:

- **Seed exploitation:** knowing the evaluation seeds lets the agent tune specifically for those random states
- **Test data memorization:** reading how test data is sampled lets the agent overfit to the test distribution
- **Metric reverse-engineering:** understanding the exact scoring formula enables adversarial optimization against the formula rather than the task

The agent knows its score but not the scoring function. It can only improve by genuinely improving the model.

## Layer 3: Behavioral Probes

`evaluate.py` contains behavioral probes that detect degenerate solutions -- models that achieve good metrics through exploitation rather than learning:

- **Training time anomaly:** a model that trains in 0.1 seconds when the baseline takes 30 seconds is suspicious (likely memorized or degenerate)
- **Model size anomaly:** extreme changes in model artifact size signal architectural gaming
- **Prediction diversity:** a model that predicts the same class for every input achieves majority-class accuracy but is useless. The probe measures the entropy of predictions.
- **Overfitting gap:** if train accuracy is 99% but validation accuracy is 60%, the model memorized the training data

Because the probes live inside the hidden `evaluate.py`, the agent cannot learn what is being checked or how to circumvent it.

## Layer 4: Statistical Validation

Single-run results are unreliable. Turing provides three levels of statistical validation:

**Multi-run evaluation** (`/turing:validate`):
Run the pipeline N times and measure coefficient of variation (CV). If CV exceeds 5%, single-run evaluation is automatically replaced with multi-run median.

```bash
python scripts/validate_stability.py --auto
# If CV > 5%: writes evaluation.n_runs: 3 to config.yaml
```

**Seed studies** (`/turing:seed`):
Run the current best model across multiple random seeds. Reports mean, standard deviation, and 95% confidence interval. Flags seed-sensitive results (CV >= 5%).

**Reproducibility verification** (`/turing:reproduce`):
Re-run a logged experiment from its saved configuration and verify metrics fall within tolerance. Detects environment drift and non-deterministic behavior.

!!! warning "Lucky seeds"
    A single seed can produce an outlier result that looks like a genuine improvement. The convergence step automatically runs `seed_runner.py --quick` before declaring final results. Seed-sensitive results are reported as mean +/- std, not single-seed numbers.

## Layer 5: Tool Restriction

The `@ml-researcher` agent's Bash access is restricted to a whitelist:

| Allowed Pattern | Purpose |
|-----------------|---------|
| `python train.py:*` | Execute training |
| `python scripts/*:*` | Run utility scripts |
| `git:*` | Version control operations |
| `source .venv/bin/activate:*` | Environment activation |
| `pip:*` | Package installation (requires human approval) |

**Blocked by omission:**

- `cat`, `head`, `tail`, `less` -- prevents reading hidden files via shell
- `curl`, `wget` -- prevents data exfiltration
- Arbitrary command execution -- prevents escape from the sandbox

The agent's `Read` tool is separately governed by the file access tiers. Hidden files are denied at the tool level. The Bash whitelist closes the shell backdoor that would otherwise let the agent `cat evaluate.py`.

## Layer 6: Diff-Based History

The agent reads git diffs of recent discarded experiments during the OBSERVE step:

```bash
for branch in $(git branch --list 'exp/*' | tail -3); do
  echo "=== $branch ==="
  git diff main...$branch -- train.py config.yaml 2>/dev/null | head -40
done
```

This matters because the agent's memory of what it changed can drift from reality. Over many iterations, the agent may "remember" trying something it never actually tried, or forget a key detail of a past experiment. Git diffs are the ground truth.

The experiment log (`experiments/log.jsonl`) is append-only -- past entries cannot be modified. Combined with git history, this creates an immutable audit trail of every experiment, successful or not.

## Defense in Depth

No single layer is sufficient. The stack works because the layers are complementary:

| Attack Vector | Layers That Block It |
|--------------|---------------------|
| Modify evaluate.py | 1 (separation), 2 (hidden), 5 (tool restriction) |
| Read evaluation code via shell | 2 (hidden), 5 (Bash whitelist) |
| Exploit fixed seeds | 2 (hidden), 4 (seed studies) |
| Train a degenerate model | 3 (behavioral probes), 4 (multi-run) |
| Report lucky single-run result | 4 (statistical validation) |
| Fabricate experiment history | 6 (diff-based, append-only log) |
| Memorize test distribution | 2 (hidden), 3 (prediction diversity probe) |

## The Lesson

Prompt-based safety rules are suggestions. Code-based safety rules are constraints. When designing autonomous agent systems, encode your invariants in architecture, not in instructions.

```
"Do not modify evaluate.py"     <- prompt instruction (workaround-able)
evaluate.py not in tool scope   <- architectural constraint (holds)
```

The difference is not philosophical. It is the difference between a rule that the agent can reason about and decide to break, and a capability that the agent simply does not have.
