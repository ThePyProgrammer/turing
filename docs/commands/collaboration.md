---
title: "Collaboration"
description: "Project onboarding for new collaborators, experiment packaging for handoff, and simulated peer review."
---

# Collaboration

Commands for working with other people -- getting new team members up to speed, packaging experiments for handoff, and stress-testing your work before external review.

---

### `/turing:onboard` -- Project onboarding

Generate a walkthrough for new collaborators that replaces a 1-hour onboarding meeting with a 5-minute read. Covers the task, experiment history, key decisions, and suggested next steps. Tailored by audience (researcher, engineer, stakeholder) and depth (brief or full).

**Syntax:** `/turing:onboard [--audience researcher|engineer|stakeholder] [--depth brief|full]`

**Examples:**
```
/turing:onboard
/turing:onboard --audience engineer --depth brief
```

---

### `/turing:share` -- Experiment packaging

Package experiments into portable archives for collaborator handoff or paper supplementary material. Each package includes config, metrics, seed study results, annotations, and reproduction instructions. Optionally includes model weights, figures, and code.

**Syntax:** `/turing:share <exp-ids...> [--include model,figures,code]`

**Examples:**
```
/turing:share exp-089
/turing:share exp-042 exp-089 --include model,figures
```

---

### `/turing:review` -- Peer review simulation

Simulate a conference reviewer before you submit. Generates likely reviewer objections with severity ratings and links each weakness to the specific `/turing:` command that fixes it. Supports venue-specific review styles and a `--harsh` mode for thorough stress testing.

**Syntax:** `/turing:review [--venue neurips|icml|general] [--harsh]`

**Examples:**
```
/turing:review
/turing:review --venue neurips --harsh
```
