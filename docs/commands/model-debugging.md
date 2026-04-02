---
title: "Model Debugging"
description: "Internal model diagnostics, hyperparameter sensitivity analysis, probability calibration, feature selection, and curriculum optimization."
---

# Model Debugging

Commands for understanding why a model behaves the way it does: inspecting internals, identifying which hyperparameters matter, fixing probability estimates, evaluating features, and optimizing training data ordering.

---

### `/turing:xray`: Internal model diagnostics

See inside the model. When it underperforms, the fix depends on why. For neural networks: gradient magnitudes, activation stats, dead neuron percentage, weight distributions, and gradient-to-weight ratios. For tree models: depth utilization, leaf purity, and feature split dominance. For scikit-learn: coefficient magnitudes and feature importance concentration. Detects dead gradients, vanishing/exploding gradients, sparse weights, and overfitting risk.

**Syntax:** `/turing:xray [exp-id] [--layer encoder.layer.2] [--compare exp-a exp-b] [--json]`

**Examples:**
```
/turing:xray exp-042              # Full diagnostics
/turing:xray                      # Best experiment
```

---

### `/turing:sensitivity`: Hyperparameter sensitivity analysis

Rank hyperparameters by impact and identify which matter and which are noise. Each parameter gets a sensitivity rating (HIGH, MED, LOW, NONE) based on the metric range observed across its sweep. Includes monotonicity detection and actionable recommendations: focus tuning on high-sensitivity parameters, stop wasting time on the rest.

**Syntax:** `/turing:sensitivity [exp-id] [--params learning_rate,max_depth] [--json]`

**Examples:**
```
/turing:sensitivity exp-042                           # All tunable params
/turing:sensitivity --params "learning_rate,max_depth" # Specific params
```

---

### `/turing:calibrate`: Probability calibration

Make model probabilities trustworthy. Measures ECE and MCE, generates reliability diagrams, and applies calibration methods: Platt scaling (logistic regression on logits), isotonic regression (non-parametric, needs more data), and temperature scaling (single scalar). Auto mode tries all methods and picks the one with the lowest ECE.

**Syntax:** `/turing:calibrate [exp-id] [--method platt|isotonic|temperature|auto] [--json]`

**Examples:**
```
/turing:calibrate exp-042                  # Auto-select best method
/turing:calibrate exp-042 --method platt   # Platt scaling only
```

---

### `/turing:feature`: Feature selection and analysis

Systematically evaluate which features matter and which are noise. Builds a consensus ranking across multiple methods (mutual information, L1, tree-based), detects redundant feature pairs with high correlation, and generates candidate interaction features. Recommends dropping zero-consensus features outright.

**Syntax:** `/turing:feature [--method all|importance|selection|generation] [--top-k 20] [--json]`

**Examples:**
```
/turing:feature                      # Full analysis
/turing:feature --top-k 10           # Top-10 consensus
```

---

### `/turing:curriculum`: Training curriculum optimization

Does the order your model sees data matter? Find out systematically. Compares easy-to-hard (classic curriculum learning), hard-to-easy (anti-curriculum), self-paced (gradually include harder samples), and random (control) strategies. Reports convergence speedup and detects impossible samples that are likely mislabeled.

**Syntax:** `/turing:curriculum [exp-id] [--strategies easy_to_hard,hard_to_easy,self_paced,random] [--json]`

**Examples:**
```
/turing:curriculum exp-042                      # All strategies
/turing:curriculum --strategies easy_to_hard,random  # Specific strategies
```
