---
title: "The Taste-Leverage Thesis"
description: "The philosophical foundations of Turing. Amy Tam's insight that taste is the differentiator, what philosophy of science says about why, and what contemporary AI leaders confirm."
---

# The Taste-Leverage Thesis

!!! quote "Amy Tam (2026)"

    "You're in a room with a quadrillion biased coins, and you want to maximize the number of heads in the shortest amount of time. Almost all coins are 'duds.' The novice coin-flipper might start flipping one-by-one, but heads come few and far between. The learned coin-flipper weaves through the quadrillion-coin room with a preternatural air; they flip many coins at once. What comes across as luck is really the refinement of taste: years of feeling faint differences in the weight of the metal, the subtle offsets of a mis-mint."

This passage is why Turing exists.

Tam wrote it in the context of autonomous coding agents, but the metaphor cuts deeper than she intended. It describes ML research with eerie precision: a quadrillion-coin room where the researcher's value lies not in the mechanical act of flipping but in *choosing which coins to flip at all*. The novice hyperparameter-tunes one model at a time. The experienced researcher senses which model families are worth exploring, which data transformations will unlock signal, which loss functions are measuring the wrong thing. They sense it before the evidence is conclusive, through what Tam calls the "refinement of taste."

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) proved the second half of the metaphor. His agent ran 700 experiments over two days, discovering 20 optimizations that produced an 11% speedup. That is more coins flipped than any human could manage. But Karpathy is explicit about what the human still provides:

> "Writing a good program.md requires having done the research yourself. You need to know which directions are worth trying, what 'better' means for your problem, and when incremental gains have run their course."

The agent flips coins with superhuman speed. The human selects which coins are worth flipping. **Tam's thesis is that the selection, the taste, is the hard part, and it always was.** The agent just makes this visible by removing the false bottleneck of execution.

---

## Why Taste Resists Automation

Tam's insight is original, but the question it raises is not: *why can't you automate the selection?* Six decades of philosophy of science asked the same question and arrived at converging answers.

### "We Can Know More Than We Can Tell"

Michael Polányi, in *The Tacit Dimension* (1966), argued that scientific judgment depends on knowledge that cannot be articulated as explicit rules:

> "We can know more than we can tell."

The researcher who *senses* that a learning rate is wrong is exercising what Polányi calls connoisseurship, knowledge transmitted by apprenticeship, not by precept. His claim is not that this knowledge is *currently* hard to formalize, but that it is *in principle* resistant to formalization. This is the philosophical validation of Tam's "feeling faint differences in the weight of the metal." Those faint differences are real, they matter, and they cannot be written as a loss function.

### The Discovery Machine That Cannot Be Built

Hans Reichenbach, writing in 1938, stated the point with the precision of formal philosophy:

> "The act of discovery escapes logical analysis; there are no logical rules in terms of which a 'discovery machine' could be constructed that would take over the creative function of the genius."

Reichenbach was distinguishing the *context of discovery* (where hypotheses come from, creative and non-logical) from the *context of justification* (how hypotheses are tested, mechanical and rule-based). Karl Popper built his entire philosophy of science on the same distinction: hypothesis generation requires "a leap of the imagination"; only testing admits of formal treatment.

An autonomous ML training loop is, in Reichenbach's terms, a *justification machine*. It tests hypotheses with superhuman efficiency. But the discovery, which hypothesis is worth testing, is exactly the part that "escapes logical analysis." It is the part that requires taste.

### The Paradigm Trap

Thomas Kuhn's *Structure of Scientific Revolutions* (1962) adds the sharpest warning for autonomous systems. Most science, "normal science," is puzzle-solving within an established paradigm. The rules are given, the methods are known. This is what an autonomous training loop does: it solves puzzles within a framework the human defined.

But paradigm selection, choosing the framework and recognizing when it is exhausted, is not algorithmic:

> "Perception of similarity cannot be reduced to rules of rationality."

The deepest risk is not that the agent gets the wrong answer. It is **efficiently optimizing within a degenerating paradigm**, tuning hyperparameters with superhuman discipline inside a research programme that should have been abandoned. In Tam's metaphor: flipping coins very fast in a corner of the room where all the coins are duds. The taste to *leave that corner* is exactly what the agent lacks.

---

## What Contemporary AI Leaders Confirm

Every major AI researcher who has spoken on the topic arrives at a version of Tam's insight, often independently, sometimes in strikingly similar language.

**Demis Hassabis** (Nobel Prize in Chemistry, 2024, for AlphaFold):

> "The human ingenuity comes in first, asking the question, developing the hypothesis, and AI systems can't do any of that."

**Yann LeCun** (Turing Award, co-founded AMI Labs):

> Current AI structurally lacks reasoning, planning, persistent memory, and world understanding. These are "not bugs to be fixed with more data but fundamental architectural limitations."

**Yoshua Bengio** (launched LawZero, 2025):

> "The Scientist AI is trained to understand, explain and predict, like a selfless idealized and platonic scientist." He draws a sharp line between AI that explains and AI that has agency.

**Alan Aspuru-Guzik** (self-driving laboratories, University of Toronto):

> Scientists must "give up the driver's seat" to a computer, which allows them to "work on more important things," such as deciding *what research projects to pursue*.

**The cautionary example:** Sakana AI's "AI Scientist" (2024) attempted fully autonomous research without the taste layer, generating hypotheses, writing code, running experiments, and producing manuscripts with no human in the loop. An independent evaluation found a 42% experiment failure rate, hallucinated results, and manuscripts that operated "at the level of an unmotivated undergraduate student rushing to meet a deadline." This is what Tam's thesis predicts: without taste, you get volume without validity.

The meta-pattern across every voice: **autonomous AI research is a force multiplier for human judgment, not a replacement for it.** The hard part of research was never running the experiment.

---

## The Architecture of Taste

Turing's design is a direct encoding of Tam's thesis. You cannot make an agent that has taste. But you can build a system where taste and execution are cleanly separated, each doing what it does best:

| | Human (Taste) | Agent (Execution) |
|---|---|---|
| **Tam** | "Feeling faint differences in the weight of the metal" | Flipping coins at superhuman speed |
| **Polányi** | Tacit knowledge, connoisseurship | Explicit knowledge, formalizable procedures |
| **Reichenbach** | Context of discovery, "the creative function" | Context of justification, logical analysis |
| **Kuhn** | Paradigm selection: knowing when to leave the corner | Normal science: puzzle-solving within a paradigm |
| **Karpathy** | Writing `program.md`, the research agenda | The training loop: 700 experiments in 2 days |
| **Hassabis** | "Asking the question" | "Analysing data" |

The interface between the two columns is Turing's reason for existing. `/turing:try` is how taste reaches the agent. `/turing:brief` is how results reach the human. Everything else, the 72 other commands, is infrastructure to make that exchange richer, faster, and more informed.

!!! info "Tam's Thesis, Formalized"

    When execution cost approaches zero, the differentiator is research taste: the accumulated judgment about which problems are worth solving, which approaches are promising, and when a direction is exhausted. Turing handles the execution. You bring the taste.

---

## Further Reading

- [The Taste-Leverage Loop](taste-leverage.md): The bidirectional interface between taste and discipline
- [Separating Hypothesis from Measurement](separation.md): Why the entity that generates hypotheses must not evaluate them
- [Experiment Tracking as Institutional Memory](memory.md): Why agents need structured memory
