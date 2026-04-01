---
name: turing
description: Autonomous ML research harness. Thin router that detects ML training intent and dispatches to focused sub-commands. Each sub-command handles one phase of the experiment lifecycle.
---

You are the Turing ML research router. Detect the user's intent and route to the appropriate sub-command. Do not attempt to handle ML tasks directly — dispatch to the focused skill.

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

## Sub-commands

| Command | Purpose | Agent |
|---|---|---|
| `/turing:train [ml/project] [N]` | Run the autonomous experiment loop (auto-detects project from path or cwd) | @ml-researcher |
| `/turing:status` | Show experiment status, best model, convergence | @ml-evaluator |
| `/turing:compare <a> <b>` | Side-by-side experiment comparison | @ml-evaluator |
| `/turing:sweep` | Generate and run hyperparameter sweep | @ml-researcher |
| `/turing:try <hypothesis>` | Inject a hypothesis into the agent's queue | (inline) |
| `/turing:brief` | Generate structured research intelligence report | @ml-evaluator |
| `/turing:init` | Scaffold a new ML project | (inline) |
| `/turing:validate` | Check metric stability, auto-fix if noisy | (inline) |
| `/turing:seed [N] [--quick]` | Multi-seed study: mean/std/CI, flag seed-sensitive results | (inline) |
| `/turing:reproduce <exp-id>` | Reproducibility verification with tolerance checking | (inline) |
| `/turing:suggest` | Literature-grounded model architecture suggestions | (inline, uses WebSearch) |
| `/turing:explore` | Tree-search hypothesis exploration via AB-MCTS | (inline) |
| `/turing:design <hyp-id>` | Generate structured experiment design from hypothesis | (inline, uses WebSearch) |
| `/turing:logbook` | HTML/markdown logbook with trajectory chart | (inline) |
| `/turing:poster` | Single-page HTML research poster | (inline) |
| `/turing:report` | Structured markdown research report | (inline) |
| `/turing:mode <mode>` | Set research strategy (explore/exploit/replicate) | (inline) |
| `/turing:preflight` | Pre-flight resource check (VRAM/RAM/disk) | (inline) |
| `/turing:card` | Generate standardized model card (type, performance, data, limitations, contract) | (inline) |
| `/turing:diagnose [exp-id]` | Error analysis: failure modes, confused pairs, feature-range bias | (inline) |
| `/turing:ablate [--components]` | Ablation study: remove components, measure impact, flag dead weight | (inline) |
| `/turing:frontier [--metrics]` | Pareto frontier: multi-objective tradeoff visualization | (inline) |
| `/turing:lit <query>` | Literature search: papers, SOTA baselines, related work | (inline, uses WebSearch) |
| `/turing:paper [--sections] [--format]` | Draft paper sections from experiment logs (setup, results, ablation, hyperparams) | (inline) |
| `/turing:export [exp-id] [--format]` | Export model to production format with equivalence check + latency benchmark | (inline) |
| `/turing:queue <action>` | Batch experiment scheduler: add, list, run, pause, clear | (inline) |
| `/turing:retry <exp-id>` | Smart failure recovery: auto-diagnose crash, apply fix, re-run | (inline) |
| `/turing:fork <exp-id> --branches` | Experiment branching: run parallel tracks, report winner | (inline) |
| `/turing:profile [exp-id]` | Computational profiling: timing, memory, throughput, bottleneck detection | (inline) |
| `/turing:checkpoint <action>` | Smart checkpoint management: list, prune (Pareto), average, resume, stats | (inline) |
| `/turing:diff <exp-a> <exp-b>` | Deep experiment comparison: config diff, metric significance, per-class regressions, curve divergence | (inline) |
| `/turing:watch [--analyze]` | Live training monitor with early-warning alerts (loss spike, NaN, overfitting, plateau) | (inline) |
| `/turing:regress [--tolerance]` | Performance regression gate: re-run best experiment, verify metrics haven't degraded | (inline) |
| `/turing:ensemble [--top-k] [--methods]` | Automated ensemble: voting, weighted voting, stacking, blending from top-K models | (inline) |
| `/turing:stitch <action> [stage]` | Pipeline composition: show/swap/cache/run stages independently | (inline) |
| `/turing:warm <exp-id>` | Warm-start from prior model: load checkpoint, freeze layers, adjust LR | (inline) |
| `/turing:scale [--axis]` | Scaling law estimator: fit power law, predict full-scale performance | (inline) |
| `/turing:budget <action>` | Compute budget manager: set limits, track allocation, auto-shift modes | (inline) |
| `/turing:distill <exp-id>` | Model compression: distill teacher into smaller student model | (inline) |
| `/turing:transfer [--from]` | Cross-project knowledge transfer: find similar prior projects, surface what worked | (inline) |
| `/turing:audit [--strict]` | Pre-submission methodology audit: data leakage, baselines, seeds, ablations, reproducibility | (inline) |
| `/turing:sanity [--quick]` | Pre-training sanity checks: initial loss, overfit test, gradient flow, output validation | (inline) |
| `/turing:baseline [--methods]` | Automatic baseline generation: random, majority/mean, linear, k-NN | (inline) |
| `/turing:leak [--deep]` | Targeted leakage detection: single-feature tests, correlation, train/test overlap | (inline) |
| `/turing:xray [exp-id]` | Internal model diagnostics: gradient flow, dead neurons, weight distributions, tree analysis | (inline) |
| `/turing:sensitivity [exp-id]` | Hyperparameter sensitivity analysis: rank parameters by impact, detect non-monotonic responses | (inline) |
| `/turing:calibrate [exp-id]` | Probability calibration: ECE/MCE, reliability diagrams, Platt/isotonic/temperature scaling | (inline) |
| `/turing:feature [--method]` | Automated feature selection: multi-method consensus ranking, redundancy, interaction generation | (inline) |
| `/turing:curriculum [exp-id]` | Training curriculum optimization: difficulty scoring, strategy comparison, impossible sample detection | (inline) |
| `/turing:prune <exp-id>` | Weight pruning: magnitude/structured/lottery, sparsity sweep, knee point detection | (inline) |
| `/turing:quantize <exp-id>` | Post-training quantization: FP16/INT8, accuracy-latency comparison, QAT suggestion | (inline) |
| `/turing:merge <exp-ids...>` | Model merging: uniform/greedy soup, TIES, DARE — free accuracy, zero latency cost | (inline) |
| `/turing:surgery <exp-id>` | Architecture modification: add/remove layer, widen/narrow, swap activation, skip connections | (inline) |

## Proactive Detection

If you detect ML training intent in the conversation (e.g., "the model accuracy is bad", "we need to improve predictions", "let's try a different model"), suggest the relevant sub-command.

## First-Time Setup

If no ML project is detected (no `config.yaml`, no `train.py`, no `experiments/`), suggest `/turing:init` first.
