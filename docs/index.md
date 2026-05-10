---
title: "Turing: The Research Assistant That Can't Fool Itself"
description: "Autonomous ML research harness for Claude Code implementing the autoresearch pattern. Iterative training, immutable evaluation, and structured experiment loops with convergence detection and safety guardrails."
template: home.html
hide:
  - navigation
  - toc
  - path
---

# Turing

**The research assistant that can't fool itself.**

An autonomous ML research harness for Claude Code. Turing implements the autoresearch pattern: an AI agent that iteratively trains, evaluates, and improves machine learning models through a structured experiment loop with convergence detection, immutable evaluation infrastructure, and safety guardrails.

---

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

---

## What is Turing?

Turing is built around a single loop: the **taste-leverage loop**.

You have taste: the accumulated judgment about which problems are tractable, which metrics capture what you care about, which directions are dead ends. Turing has leverage: the discipline to run experiments without fatigue, track every result without amnesia, and measure without contamination.

The interface is built around two verbs: **try** and **brief**. `/turing:try` is how your taste reaches the agent. `/turing:brief` is how the agent's results reach you. Everything else is infrastructure connecting those two endpoints.

!!! info "The Taste-Leverage Loop"

    You inject hypotheses. The agent executes them with superhuman discipline. The briefing tells you what happened. You inject new hypotheses informed by the results. The agent never forgets what it tried. You never lose context between sessions.

Read more in [The Taste-Leverage Loop](philosophy/taste-leverage.md).

---

## At a Glance

| | |
|---|---|
| **74 commands** | Core loop, taste-leverage interface, reporting, validation, exploration, model surgery, experiment archaeology, research communication, and more |
| **2 agents** | `@ml-researcher` (read/write) and `@ml-evaluator` (read-only), strict capability boundary |
| **2010 tests** | Unit, integration, anti-pattern, and manifest coverage |
| **29 phases** | Complete development roadmap, fully delivered |

---

## Why "Turing"?

The name references Alan Turing, the person who first asked whether machines could think, then built the framework for answering the question. Turing the plugin does what Turing the person formalized: it **defines** a computational process, **executes** it mechanically, and **determines** whether the result constitutes an improvement.

---

## Installation

```bash
# Via npm
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

---

## Learn More

- [Philosophy](philosophy/index.md): Why Turing exists and the ideas behind it
- [The Taste-Leverage Loop](philosophy/taste-leverage.md): The bidirectional interface between human judgment and agent execution
- [Separating Hypothesis from Measurement](philosophy/separation.md): The epistemological foundation
- [Experiment Tracking as Institutional Memory](philosophy/memory.md): Why agents need structured memory
