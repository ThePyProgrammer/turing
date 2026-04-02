---
title: "The Discovery Machine That Cannot Be Built"
description: "The philosophical foundations of Turing — why research taste resists automation, what six decades of philosophy of science predict about autonomous agents, and what contemporary AI leaders confirm."
---

# The Discovery Machine That Cannot Be Built

!!! quote "Hans Reichenbach, *Experience and Prediction* (1938)"

    "The act of discovery escapes logical analysis; there are no logical rules in terms of which a 'discovery machine' could be constructed that would take over the creative function of the genius."

Reichenbach wrote this in 1938 — six decades before the first neural network won an ImageNet competition, nine decades before an AI agent ran 126 experiments overnight. He was not making a prediction about compute. He was making a claim about the *logical structure of discovery itself*: there are no rules to follow. You cannot build a machine that generates the hypotheses worth testing, because generating hypotheses worth testing is not a rule-following activity.

And yet, in March 2026, Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) did exactly what Reichenbach said was impossible — or did it? The agent ran 700 experiments over two days, discovering 20 optimizations that produced an 11% speedup. Tobias Lutke reported a 19% performance gain from a single overnight session. That is a lot of coins flipped.

The resolution is that Reichenbach was right about discovery and Karpathy was right about execution. They are talking about different things. The agent did not *discover* anything — it *tested* hypotheses within a search space that a human defined. Karpathy himself is explicit about this:

> "You're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the program.md Markdown files that provide context to the AI agents and set up your autonomous research org."

The human writes `program.md` — the research agenda, the definition of "better," the stopping criteria. The agent flips coins. The discovery is in the program, not in the flipping.

---

## What Philosophy of Science Predicted

The distinction between *choosing what to investigate* and *mechanically testing it* is not new. It is the central structural feature of the scientific method, identified independently by every major 20th-century philosopher of science — who agreed on almost nothing else.

### The Context of Discovery vs. the Context of Justification

Karl Popper drew the foundational line. Hypothesis generation requires "a leap of the imagination" — it is creative, non-logical, and not susceptible to formal analysis. Only hypothesis *testing* (falsification) is rule-based and mechanical. Popper approvingly cites Einstein:

!!! quote "Albert Einstein, cited in Popper's *The Logic of Scientific Discovery*"

    "There is no logical path leading to [the highly universal laws of science]. They can only be reached by intuition, based upon something like an intellectual love of the objects of experience."

This maps directly onto what an autonomous ML training loop does. The loop *tests* hypotheses — runs experiments, evaluates metrics, keeps improvements, discards regressions. That is the context of justification. The context of discovery — *which* hypothesis to test, *why* it might matter, *when* to abandon the approach entirely — remains outside the loop.

### Tacit Knowledge: What We Know But Cannot Tell

Michael Polányi's concept of *tacit knowledge* provides the deeper explanation. In *The Tacit Dimension* (1966), he argues that scientific judgment depends on a form of knowing that cannot be articulated as explicit rules:

> "We can know more than we can tell."

A seasoned ML researcher *senses* that a learning rate is too high, that a loss curve looks wrong, that a model architecture is fighting the data distribution. This sensing is not expressible as a function — it is connoisseurship, transmitted by apprenticeship, not by precept. Polányi's claim is not that tacit knowledge is *currently* hard to formalize, but that it is *in principle* resistant to formalization.

Amy Tam captured the same insight in the language of her generation:

!!! quote "Amy Tam (2026)"

    "What comes across as luck is really the refinement of taste: years of feeling faint differences in the weight of the metal, the subtle offsets of a mis-mint."

Tam's "taste" is Polányi's "tacit knowledge" is Popper's "context of discovery." Three formulations, one insight: the part of research that matters most is the part you cannot write down as a recipe.

### Normal Science and the Paradigm Trap

Thomas Kuhn's *Structure of Scientific Revolutions* (1962) adds the sharpest warning. Most science — what Kuhn calls "normal science" — is puzzle-solving within an established paradigm. The rules are given, the methods are known, the criteria for success are defined. This is exactly what an autonomous training loop does: it solves puzzles (hyperparameter optimization, architecture search) within a given framework.

But *paradigm selection* — choosing the framework itself, recognizing when the current approach is exhausted, sensing that an anomaly is significant rather than noise — is explicitly non-algorithmic:

> "Perception of similarity cannot be reduced to rules of rationality."

The deepest risk of autonomous research is not getting the wrong answer. It is **efficiently optimizing within a degenerating paradigm** — solving puzzles with great precision in a framework that should have been abandoned. The agent tunes hyperparameters with superhuman discipline while the human should have switched to a different model family three hours ago.

Imre Lakatos formalized this as the distinction between *hard core* commitments (what to protect — human judgment) and *protective belt* adjustments (what to tweak — mechanical work). The agent adjusts the protective belt. The human evaluates whether the research programme is still progressive or has degenerated into post-hoc rationalization.

---

## What Contemporary AI Leaders Confirm

The philosophical predictions are not theoretical. Every major AI researcher who has spoken on the topic confirms the same structural insight — from opposite ends of the optimism spectrum.

**Demis Hassabis** (Nobel Prize in Chemistry, 2024, for AlphaFold):

> "The human ingenuity comes in first — asking the question, developing the hypothesis — and AI systems can't do any of that. It just sort of analyses data right now."

**Yann LeCun** (Turing Award, left Meta to co-found AMI Labs):

> Current AI structurally lacks reasoning, planning, persistent memory, and world understanding. These are "not bugs to be fixed with more data but fundamental architectural limitations."

**Yoshua Bengio** (launched LawZero, 2025, for non-agentic "Scientist AI"):

> "The Scientist AI is trained to understand, explain and predict, like a selfless idealized and platonic scientist."

Bengio draws the sharpest line: an AI that *understands* the world can accelerate science as a tool. An AI that *acts* in the world introduces agency risks. An autonomous training loop sits at this boundary — the agent acts (modifies code, runs experiments), but the human constrains the action space.

**The cautionary example:** Sakana AI's "AI Scientist" (2024) attempted fully autonomous research — generating hypotheses, writing code, running experiments, producing manuscripts. An independent evaluation found a 42% experiment failure rate, hallucinated numerical results, placeholder text like "Conclusions Here," and manuscripts with a median of five citations, mostly outdated. The reviewers concluded it operated "at the level of an unmotivated undergraduate student rushing to meet a deadline."

This is autoresearch without taste. Volume without validity is worse than useless — it is actively misleading.

---

## The Architecture of Taste

The resolution is architectural, not aspirational.

You cannot make an agent that has taste. But you can build a system where taste and execution are cleanly separated, each doing what it does best:

| Domain | Human (Taste) | Agent (Execution) |
|--------|---------------|-------------------|
| **Popper** | Context of discovery — "a leap of the imagination" | Context of justification — deductive falsification |
| **Polányi** | Tacit knowledge — connoisseurship | Explicit knowledge — formalizable procedures |
| **Kuhn** | Paradigm selection — anomaly recognition | Normal science — puzzle-solving within a paradigm |
| **Lakatos** | Hard core — programme evaluation | Protective belt — auxiliary hypothesis adjustment |
| **Karpathy** | `program.md` — the research agenda | The training loop — 700 experiments in 2 days |
| **Hassabis** | "Asking the question" | "Analysing data" |

Turing encodes this division as code, not as a suggestion. The human injects hypotheses (`/turing:try`). The agent executes them. The briefing reports results (`/turing:brief`). The human decides what to try next. The loop is bidirectional by design — not because the agent needs help, but because the human provides the one thing the agent structurally cannot: judgment about what matters.

---

## Further Reading

- [The Taste-Leverage Loop](taste-leverage.md) — The bidirectional interface between taste and discipline
- [Separating Hypothesis from Measurement](separation.md) — Why the entity that generates hypotheses must not evaluate them
- [Experiment Tracking as Institutional Memory](memory.md) — Why agents need structured memory
- [Intellectual Heritage](heritage.md) — The full lineage of ideas Turing builds on
