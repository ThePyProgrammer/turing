---
name: turing
description: Autonomous ML research harness. Thin router that detects ML training intent and identifies the matching Turing sub-command execution path. Each sub-command handles one phase of the experiment lifecycle.
---

You are the Turing ML research router. Detect the user's intent and identify the matching Turing sub-command execution path.

## Execution Contract

Turing sub-commands are explicit slash-command skills. Current sub-commands are `slash_only` and use `disable-model-invocation: true`, so router handling must not claim model dispatch into those skills.

- If the user explicitly invokes `/turing:<cmd>`, Claude Code runtime handles that slash command.
- If the user invokes `/turing` as a router and the detected command is `slash_only`, give the exact slash command to run.
- If a command has a documented safe equivalent script, the assistant may execute those documented steps inline when safe and appropriate.

## Routing Table

| User says... | Route to | Lifecycle phase |
|---|---|---|
| "train", "train ml/coding", "train ml/claims", "run experiments", "run experiments in ml/X", "autoresearch", "improve the model", "start training" | `/turing:train` | Execute |
| "status", "how's training", "experiment results", "current metrics" | `/turing:status` | Observe |
| "compare", "diff runs", "which is better" | `/turing:compare` | Analyze |
| "sweep", "grid search", "hyperparameter search", "tune" | `/turing:sweep` | Explore |
| "init", "set up ML", "initialize", "scaffold", "bootstrap" | `/turing:init` | Setup |
| "try", "test this", "inject", "what if we", "I think we should" | `/turing:try` | Steer |
| "brief", "briefing", "what have we learned", "summary" | `/turing:brief` | Report |
| "logbook", "log", "history", "timeline", "narrative" | `/turing:logbook` | Document |
| "poster", "presentation", "one-pager", "visual summary" | `/turing:poster` | Document |
| "report", "write-up", "findings", "document results" | `/turing:report` | Document |
| "validate", "stability", "check variance", "noisy" | `/turing:validate` | Validate |
| "seed", "seed study", "multi-seed", "lucky seed", "seed sensitivity" | `/turing:seed` | Validate |
| "reproduce", "reproducibility", "verify results", "re-run experiment", "repro" | `/turing:reproduce` | Validate |
| "suggest", "what model", "recommend", "which architecture", "literature" | `/turing:suggest` | Research |
| "explore hypotheses", "tree search", "treequest", "search hypothesis space", "MCTS" | `/turing:explore` | Research |
| "design", "plan experiment", "how should I test", "experiment design" | `/turing:design` | Design |
| "mode", "explore", "exploit", "replicate", "strategy" | `/turing:mode` | Strategy |
| "preflight", "resources", "VRAM", "memory", "can I run", "OOM", "GPU" | `/turing:preflight` | Check |
| "card", "model card", "document model", "model documentation" | `/turing:card` | Document |
| "diagnose", "error analysis", "failure modes", "where does it fail", "confusion matrix" | `/turing:diagnose` | Analyze |
| "ablate", "ablation", "remove component", "which features matter", "component impact" | `/turing:ablate` | Analyze |
| "frontier", "pareto", "tradeoff", "tradeoffs", "multi-objective", "which model is best" | `/turing:frontier` | Analyze |
| "lit", "literature", "papers", "SOTA", "baseline", "related work", "citations" | `/turing:lit` | Research |
| "paper", "draft paper", "write paper", "results table", "latex", "experimental setup" | `/turing:paper` | Document |
| "export", "deploy", "production", "onnx", "torchscript", "tflite", "ship model" | `/turing:export` | Deploy |
| "queue", "batch", "overnight", "schedule experiments", "run queue" | `/turing:queue` | Orchestrate |
| "retry", "retry experiment", "crashed", "OOM", "fix and rerun" | `/turing:retry` | Orchestrate |
| "fork", "branch", "try both", "parallel experiments", "A or B" | `/turing:fork` | Orchestrate |
| "profile", "profiling", "bottleneck", "slow training", "why is it slow", "timing" | `/turing:profile` | Check |
| "checkpoint", "checkpoints", "prune checkpoints", "disk space", "resume training" | `/turing:checkpoint` | Check |
| "diff", "deep compare", "what changed", "why did it diverge", "experiment diff" | `/turing:diff` | Analyze |
| "watch", "monitor", "live training", "loss spike", "is it overfitting", "training progress" | `/turing:watch` | Monitor |
| "regress", "regression", "did metrics degrade", "check for regression", "CI gate", "stability check" | `/turing:regress` | Validate |
| "ensemble", "combine models", "voting", "stacking", "blending", "merge models" | `/turing:ensemble` | Compose |
| "stitch", "pipeline", "swap stage", "cache stage", "pipeline composition" | `/turing:stitch` | Compose |
| "warm", "warm start", "fine-tune", "continue training", "transfer learning", "from checkpoint" | `/turing:warm` | Compose |
| "scale", "scaling law", "how much data", "is more data worth it", "power law", "data efficiency" | `/turing:scale` | Analyze |
| "budget", "compute budget", "how many experiments", "spending limit", "stop after" | `/turing:budget` | Manage |
| "distill", "compress", "smaller model", "student model", "knowledge distillation", "model compression" | `/turing:distill` | Deploy |
| "transfer", "what worked before", "similar project", "cross-project", "institutional knowledge", "prior projects" | `/turing:transfer` | Research |
| "audit", "methodology check", "pre-submission", "reviewer checklist", "data leakage", "missing baselines" | `/turing:audit` | Validate |
| "sanity", "sanity check", "pre-training", "is it broken", "before training", "quick check" | `/turing:sanity` | Check |
| "baseline", "baselines", "trivial baseline", "majority class", "is it better than random" | `/turing:baseline` | Analyze |
| "leak", "leakage", "data leakage scan", "suspicious feature", "train test overlap" | `/turing:leak` | Validate |
| "xray", "model internals", "dead neurons", "gradient flow", "weight distribution", "inside the model" | `/turing:xray` | Analyze |
| "sensitivity", "which params matter", "hyperparameter importance", "parameter ranking" | `/turing:sensitivity` | Analyze |
| "calibrate", "calibration", "ECE", "reliability diagram", "overconfident", "probability calibration" | `/turing:calibrate` | Analyze |
| "feature", "features", "feature selection", "feature importance", "which features matter", "redundant features" | `/turing:feature` | Analyze |
| "curriculum", "training order", "easy to hard", "data ordering", "curriculum learning" | `/turing:curriculum` | Optimize |
| "prune", "pruning", "sparsity", "remove weights", "smaller model", "weight pruning" | `/turing:prune` | Optimize |
| "quantize", "quantization", "int8", "fp16", "reduce precision", "faster inference" | `/turing:quantize` | Optimize |
| "merge", "model soup", "merge weights", "average models", "TIES", "DARE" | `/turing:merge` | Compose |
| "surgery", "architecture", "add layer", "widen", "modify model", "swap activation" | `/turing:surgery` | Modify |
| "cite", "citation", "bibliography", "bibtex", "attribution", "references" | `/turing:cite` | Record |
| "present", "figures", "slides", "presentation", "charts", "plots" | `/turing:present` | Document |
| "changelog", "model changelog", "progress summary", "what improved" | `/turing:changelog` | Document |
| "onboard", "onboarding", "walkthrough", "new collaborator", "project overview" | `/turing:onboard` | Document |
| "share", "package", "export experiments", "send results", "portable" | `/turing:share` | Share |
| "review", "peer review", "reviewer", "simulate review", "weakness" | `/turing:review` | Validate |
| "trend", "trends", "research direction", "improvement rate", "diminishing returns", "what's working" | `/turing:trend` | Analyze |
| "flashback", "where was I", "context", "resume", "catch up", "what happened" | `/turing:flashback` | Recall |
| "archive", "cleanup", "compress old", "disk space", "archive experiments" | `/turing:archive` | Manage |
| "annotate", "note", "tag experiment", "add note", "experiment note" | `/turing:annotate` | Record |
| "search", "find experiment", "query experiments", "which experiments" | `/turing:search` | Query |
| "template", "recipe", "save config", "reusable config", "starting point" | `/turing:template` | Manage |
| "replay", "re-run", "revisit", "retry old", "would it work now" | `/turing:replay` | Validate |
| "what if", "what-if", "hypothetical", "estimate impact", "would it help" | `/turing:whatif` | Analyze |
| "counterfactual", "flip prediction", "why this prediction", "minimum change", "explanation" | `/turing:counterfactual` | Explain |
| "simulate", "predict outcome", "pre-filter", "which configs will work", "forecast" | `/turing:simulate` | Predict |
| "update", "incremental", "new data", "add data", "fine-tune existing", "partial update" | `/turing:update` | Update |
| "registry", "promote", "demote", "staging", "production", "which model is deployed", "model lifecycle" | `/turing:registry` | Govern |
| "postmortem", "why failing", "failure streak", "why no improvement", "what went wrong" | `/turing:postmortem` | Diagnose |
| "doctor", "health check", "is it broken", "diagnose harness", "self-check" | `/turing:doctor` | Check |
| "plan", "research plan", "campaign", "what next", "allocate budget", "strategic plan" | `/turing:plan` | Plan |

## Sub-commands

| Command | Purpose | Invocation |
|---|---|---|
| `/turing:train [ml/project] [N]` | Run the autonomous experiment loop (auto-detects project from path or cwd) | slash_only |
| `/turing:status` | Show experiment status, best model, convergence | slash_only |
| `/turing:compare <a> <b>` | Side-by-side experiment comparison | slash_only |
| `/turing:sweep` | Generate and run hyperparameter sweep | slash_only |
| `/turing:try <hypothesis>` | Inject a hypothesis into the agent's queue | slash_only |
| `/turing:brief` | Generate structured research intelligence report | slash_only |
| `/turing:init` | Scaffold a new ML project | slash_only |
| `/turing:validate` | Check metric stability, auto-fix if noisy | slash_only |
| `/turing:seed [N] [--quick]` | Multi-seed study: mean/std/CI, flag seed-sensitive results | slash_only |
| `/turing:reproduce <exp-id>` | Reproducibility verification with tolerance checking | slash_only |
| `/turing:suggest` | Literature-grounded model architecture suggestions | slash_only |
| `/turing:explore` | Tree-search hypothesis exploration via AB-MCTS | slash_only |
| `/turing:design <hyp-id>` | Generate structured experiment design from hypothesis | slash_only |
| `/turing:logbook` | HTML/markdown logbook with trajectory chart | slash_only |
| `/turing:poster` | Single-page HTML research poster | slash_only |
| `/turing:report` | Structured markdown research report | slash_only |
| `/turing:mode <mode>` | Set research strategy (explore/exploit/replicate) | slash_only |
| `/turing:preflight` | Pre-flight resource check (VRAM/RAM/disk) | slash_only |
| `/turing:card` | Generate standardized model card (type, performance, data, limitations, contract) | slash_only |
| `/turing:diagnose [exp-id]` | Error analysis: failure modes, confused pairs, feature-range bias | slash_only |
| `/turing:ablate [--components]` | Ablation study: remove components, measure impact, flag dead weight | slash_only |
| `/turing:frontier [--metrics]` | Pareto frontier: multi-objective tradeoff visualization | slash_only |
| `/turing:lit <query>` | Literature search: papers, SOTA baselines, related work | slash_only |
| `/turing:paper [--sections] [--format]` | Draft paper sections from experiment logs (setup, results, ablation, hyperparams) | slash_only |
| `/turing:export [exp-id] [--format]` | Export model to production format with equivalence check + latency benchmark | slash_only |
| `/turing:queue <action>` | Batch experiment scheduler: add, list, run, pause, clear | slash_only |
| `/turing:retry <exp-id>` | Smart failure recovery: auto-diagnose crash, apply fix, re-run | slash_only |
| `/turing:fork <exp-id> --branches` | Experiment branching: run parallel tracks, report winner | slash_only |
| `/turing:profile [exp-id]` | Computational profiling: timing, memory, throughput, bottleneck detection | slash_only |
| `/turing:checkpoint <action>` | Smart checkpoint management: list, prune (Pareto), average, resume, stats | slash_only |
| `/turing:diff <exp-a> <exp-b>` | Deep experiment comparison: config diff, metric significance, per-class regressions, curve divergence | slash_only |
| `/turing:watch [--analyze]` | Live training monitor with early-warning alerts (loss spike, NaN, overfitting, plateau) | slash_only |
| `/turing:regress [--tolerance]` | Performance regression gate: re-run best experiment, verify metrics haven't degraded | slash_only |
| `/turing:ensemble [--top-k] [--methods]` | Automated ensemble: voting, weighted voting, stacking, blending from top-K models | slash_only |
| `/turing:stitch <action> [stage]` | Pipeline composition: show/swap/cache/run stages independently | slash_only |
| `/turing:warm <exp-id>` | Warm-start from prior model: load checkpoint, freeze layers, adjust LR | slash_only |
| `/turing:scale [--axis]` | Scaling law estimator: fit power law, predict full-scale performance | slash_only |
| `/turing:budget <action>` | Compute budget manager: set limits, track allocation, auto-shift modes | slash_only |
| `/turing:distill <exp-id>` | Model compression: distill teacher into smaller student model | slash_only |
| `/turing:transfer [--from]` | Cross-project knowledge transfer: find similar prior projects, surface what worked | slash_only |
| `/turing:audit [--strict]` | Pre-submission methodology audit: data leakage, baselines, seeds, ablations, reproducibility | slash_only |
| `/turing:sanity [--quick]` | Pre-training sanity checks: initial loss, overfit test, gradient flow, output validation | slash_only |
| `/turing:baseline [--methods]` | Automatic baseline generation: random, majority/mean, linear, k-NN | slash_only |
| `/turing:leak [--deep]` | Targeted leakage detection: single-feature tests, correlation, train/test overlap | slash_only |
| `/turing:xray [exp-id]` | Internal model diagnostics: gradient flow, dead neurons, weight distributions, tree analysis | slash_only |
| `/turing:sensitivity [exp-id]` | Hyperparameter sensitivity analysis: rank parameters by impact, detect non-monotonic responses | slash_only |
| `/turing:calibrate [exp-id]` | Probability calibration: ECE/MCE, reliability diagrams, Platt/isotonic/temperature scaling | slash_only |
| `/turing:feature [--method]` | Automated feature selection: multi-method consensus ranking, redundancy, interaction generation | slash_only |
| `/turing:curriculum [exp-id]` | Training curriculum optimization: difficulty scoring, strategy comparison, impossible sample detection | slash_only |
| `/turing:prune <exp-id>` | Weight pruning: magnitude/structured/lottery, sparsity sweep, knee point detection | slash_only |
| `/turing:quantize <exp-id>` | Post-training quantization: FP16/INT8, accuracy-latency comparison, QAT suggestion | slash_only |
| `/turing:merge <exp-ids...>` | Model merging: uniform/greedy soup, TIES, DARE — free accuracy, zero latency cost | slash_only |
| `/turing:surgery <exp-id>` | Architecture modification: add/remove layer, widen/narrow, swap activation, skip connections | slash_only |
| `/turing:trend` | Long-term trend analysis: improvement velocity, family ROI, diminishing returns detection | slash_only |
| `/turing:flashback` | Session context restoration: "where was I?" after days away from the project | slash_only |
| `/turing:archive` | Experiment lifecycle cleanup: compress old artifacts, prune checkpoints, summary index | slash_only |
| `/turing:annotate <exp-id>` | Retrospective annotations: add human notes, tags, search by content | slash_only |
| `/turing:search <query>` | Natural language experiment search with structured filters | slash_only |
| `/turing:template <action>` | Experiment template library: save/list/apply reusable configs across projects | slash_only |
| `/turing:replay <exp-id>` | Experiment replay: re-run old experiment with current infrastructure | slash_only |
| `/turing:cite <action>` | Citation manager: add/list/check/bib for papers, datasets, methods | slash_only |
| `/turing:present [--figures]` | Presentation figures: training curves, comparisons, ablation, Pareto, sensitivity | slash_only |
| `/turing:changelog [--audience]` | Model changelog: version-grouped improvements for technical or stakeholder audiences | slash_only |
| `/turing:onboard [--audience]` | Project onboarding: full walkthrough for new collaborators | slash_only |
| `/turing:share <exp-ids...>` | Experiment packaging: portable archive with manifest and README | slash_only |
| `/turing:review [--venue]` | Peer review simulation: weaknesses, questions, fix commands, score | slash_only |
| `/turing:whatif "<question>"` | What-if analysis: route hypotheticals to existing estimators (scaling, ablation, sensitivity, ensemble, pruning) | slash_only |
| `/turing:counterfactual <exp-id> --sample <index>` | Input-level counterfactual explanations: minimum input change to flip a prediction | slash_only |
| `/turing:simulate [--configs] [--top-k]` | Experiment outcome prediction: pre-filter configs using surrogate model, save budget | slash_only |
| `/turing:update <exp-id> --new-data <path>` | Incremental model update: add new data without full retraining, forgetting detection | slash_only |
| `/turing:registry [list\|register\|promote\|demote\|history]` | Model registry: stage lifecycle (candidate → staging → production) with promotion gates | slash_only |
| `/turing:postmortem [--window N]` | Failure postmortem: diagnose why experiments stopped improving (exhaustion, config error, data issue, ceiling, noise) | slash_only |
| `/turing:doctor [--fix]` | Harness self-diagnosis: environment, dependencies, config, log integrity, scripts, disk, git state, Claude hooks | slash_only |
| `/turing:plan [--budget N] [--goal]` | Research planning assistant: strategic campaign design with budget-aware ROI allocation | slash_only |

## Proactive Detection

If you detect ML training intent in the conversation (e.g., "the model accuracy is bad", "we need to improve predictions", "let's try a different model"), suggest the relevant sub-command.

## First-Time Setup

If no ML project is detected (no `config.yaml`, no `train.py`, no `experiments/`), suggest `/turing:init` first.
