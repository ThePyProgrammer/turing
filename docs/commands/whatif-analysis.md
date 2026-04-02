---
title: "What-If Analysis"
description: "Answer hypothetical questions from existing data, generate counterfactual explanations, and predict experiment outcomes before running them."
---

# What-If Analysis

Commands for answering questions without running new experiments -- hypothetical scenario analysis from existing data, minimum-change counterfactual explanations for individual predictions, and outcome prediction for proposed configurations.

---

### `/turing:whatif` -- What-if scenario analysis

Answer "what if?" questions using existing experiment data without running new experiments. Routes to the right estimator automatically based on the question type: data scaling extrapolation, ablation study data, pipeline stitch estimation, sensitivity interpolation, ensemble correlation analysis, pruning sweep interpolation, and budget allocation comparison.

**Syntax:** `/turing:whatif "<question>" [--json]`

**Examples:**
```
/turing:whatif "what if I had 2x more data"
/turing:whatif "what if I removed class 3"
/turing:whatif "what if I combined exp-031 with exp-042"
/turing:whatif "what if learning_rate was 0.01" --json
```

---

### `/turing:counterfactual` -- Input-level counterfactual explanations

Find the smallest input change needed to flip a prediction. For individual samples, identifies which features need to change (and by how much) to produce a different classification. Uses both greedy perturbation (change one feature at a time) and prototype-based methods (find nearest training sample from target class), selecting the explanation with the smallest distance.

**Syntax:** `/turing:counterfactual <exp-id> --sample <index> [--target <class>] [--batch-misclassified] [--json]`

**Examples:**
```
/turing:counterfactual exp-042 --sample 1247
/turing:counterfactual exp-042 --sample 1247 --target 0
/turing:counterfactual exp-042 --batch-misclassified
/turing:counterfactual exp-042 --sample 500 --json
```

---

### `/turing:simulate` -- Experiment outcome prediction

Predict which configs will beat the current best before spending compute. Builds a surrogate model from experiment history (weighted k-NN), predicts metrics for each proposed config, applies a novelty penalty for configs far from the training distribution, and ranks the results. Only recommends configs predicted to improve over the current best.

**Syntax:** `/turing:simulate [--configs configs.yaml] [--top-k 5] [--threshold 0.001] [--json]`

**Examples:**
```
/turing:simulate --configs sweep_configs.yaml
/turing:simulate --configs candidates.yaml --top-k 3
/turing:simulate --configs proposals.yaml --threshold 0.005
/turing:simulate --configs sweep.yaml --json
```
