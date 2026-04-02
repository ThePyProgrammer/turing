---
title: "When Code Is Free, Research Is All That Matters"
description: "The philosophical foundations of Turing — why autonomous ML research demands taste, why evaluation must be immutable, and why the autoresearch pattern changes the division of labor between humans and agents."
---

# When Code Is Free, Research Is All That Matters

!!! quote "Amy Tam"

    "You're in a room with a quadrillion biased coins, and you want to maximize the number of heads in the shortest amount of time. Almost all coins are 'duds.' The novice coin-flipper might start flipping one-by-one, but heads come few and far between. The learned coin-flipper weaves through the quadrillion-coin room with a preternatural air; they flip many coins at once. What comes across as luck is really the refinement of taste: years of feeling faint differences in the weight of the metal, the subtle offsets of a mis-mint."

This is the most precise metaphor for ML research in the age of autonomous agents: a quadrillion-coin room where the researcher's value lies not in the mechanical act of flipping but in *choosing which coins to flip at all*.

---

## The Problem

The agentic coding tools consuming software engineering alive right now — Cursor, Claude Code, Codex — work precisely because engineering has a built-in feedback signal: a test to pass, a spec to meet, a benchmark to clear. You can RL on [SWE-bench](https://www.swebench.com/) because the ground truth exists.

**Research has no equivalent.** It is not clear what it means to RL on a research question, because it is not clear what definition of "ground truth" one should optimize for. The coin room has a quadrillion coins but no label telling you which ones are biased toward heads.

And yet Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) ran 126 experiments overnight on a single GPU: agents modifying LLM training code, running a five-minute training loop, checking if the result improved, and repeating. Tobias Lutke reported that after letting it run overnight, it executed 37 experiments and delivered a 19% performance gain. That is a lot more coins flipped than the average human in the same time.

---

## The New Division of Labor

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

---

## The Central Claim

*When anyone can build for free, the differentiator is knowing what's worth building and whether it's buildable at all.*

Turing handles the building. You bring the knowing.

!!! info "Why Autoresearch Matters"

    Karpathy's 126 experiments overnight demonstrated something important: the experiment loop in ML is mechanical enough to automate, given two constraints. First, the evaluation must be immutable — the agent cannot move the goalposts. Second, convergence must be detected — the agent must know when to stop flipping coins in one corner and move to another.

    Turing formalizes both constraints as code, not as prompts.

---

## Further Reading

- [The Taste-Leverage Loop](taste-leverage.md) — The bidirectional interface between taste and discipline
- [Separating Hypothesis from Measurement](separation.md) — Why the entity that generates hypotheses must not evaluate them
- [Experiment Tracking as Institutional Memory](memory.md) — Why agents need structured memory
- [Intellectual Heritage](heritage.md) — The ideas Turing builds on
