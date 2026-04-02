---
title: Configuration Reference
description: All configuration files that control Turing's behavior -- config.yaml, hypotheses.yaml, model_contract.md, and model_registry.yaml.
---

# Configuration Reference

Turing scaffolds four configuration files into each project. The agent reads these; you own them.

## `config.yaml`

The main configuration file. Every field is documented inline after scaffolding.

```yaml
data:
  source: "data/reviews.csv"          # Path to raw dataset
  splits_dir: "data/splits"           # Where train/val/test splits land
  target_column: "label"              # Prediction target column name
  split_ratios:
    train: 0.70                       # (1)
    val: 0.15
    test: 0.15
  random_state: 42                    # Reproducible splits

evaluation:
  primary_metric: "f1_weighted"       # The metric Turing optimizes
  metrics:                            # All metrics computed per experiment
    - "f1_weighted"
    - "accuracy"
    - "precision_weighted"
  lower_is_better: false              # (2)

  # Multi-seed configuration (/turing:seed)
  seed_seeds: [42, 123, 456, 789, 1024, 1337, 2048, 3141, 4096, 7919]
  seed_study_n_runs: 5               # Seeds per study
  seed_sensitivity_threshold: 5.0    # CV% above this = seed-sensitive

  # Reproducibility (/turing:reproduce)
  reproduce_tolerance: 0.02          # 2% relative tolerance
  reproduce_n_runs: 3                # Reproduction attempts

convergence:
  patience: 3                        # (3)
  improvement_threshold: 0.005       # 0.5% relative improvement required

model:
  type: "xgboost"                    # Key from model_registry.yaml
  hyperparams:                       # (4)
    n_estimators: 100
    max_depth: 4
    learning_rate: 0.1
    objective: "multi:softmax"
    num_class: 2
    eval_metric: "mlogloss"
    verbosity: 0

output:
  models_dir: "models"
  best_model_dir: "models/best"
  archive_dir: "models/archive"
  experiment_log: "experiments/log.jsonl"
  results_tsv: "experiments/results.tsv"

constraints:                          # (5)
  min_train_time: 5
  min_model_size_bytes: 100
```

1. Ratios must sum to 1.0. The test split is never touched during training.
2. Set to `true` for loss-style metrics (MAE, MSE, RMSE). Set to `false` for accuracy-style metrics (accuracy, F1, AUC).
3. How many consecutive non-improving experiments before the agent stops. Higher values let it explore more; lower values save compute.
4. The agent modifies this section during experiments. Your initial values serve as the baseline.
5. Anti-cheating constraints. If training finishes suspiciously fast or the model file is trivially small, the evaluation harness flags the result.

## `hypotheses.yaml`

The hypothesis queue. Each entry is a structured idea the agent tests in order.

```yaml
hypotheses:
  - id: "h001"
    text: "Increase n-gram range to (1,3) for richer text features"
    source: "user"              # "user" or "agent"
    priority: 1                 # Lower number = tested first
    status: "tested"            # pending | tested | rejected
    result_exp_id: "exp-003"

  - id: "h002"
    text: "Try LightGBM with dart boosting for regularization"
    source: "agent"
    priority: 5
    status: "pending"
```

You can edit this file directly, or use `/turing:try "your idea"` which appends an entry with `source: user` and `priority: 1`.

## `model_contract.md`

Defines the artifact schema for saved models. Consumers of your model (serving code, downstream pipelines) depend on this contract.

Key fields:

- **Bundle format** -- joblib file at `models/best/model.joblib` containing the fitted model, featurizer, config, and contract version
- **Metadata** -- JSON at `models/best/metadata.json` with metrics, feature names, and timestamps
- **Consumer contract** -- `.predict()` on the model, `.transform()` on the featurizer
- **Breaking changes** -- increment `contract_version` when the feature schema, label encoding, or bundle format changes

## `model_registry.yaml`

Catalog of available model architectures. The agent reads this when suggesting alternatives or comparing model families.

```yaml
models:
  xgboost:
    name: "XGBoost Classifier"
    family: "gradient_boosting"
    notes: "Default. Good for tabular data with mixed feature types."
    default_hyperparams:
      n_estimators: 100
      max_depth: 4
      learning_rate: 0.1

  lightgbm:
    name: "LightGBM Classifier"
    family: "gradient_boosting"
    notes: "Often faster than XGBoost. Leaf-wise growth."
    default_hyperparams:
      n_estimators: 100
      max_depth: -1
      learning_rate: 0.1
      num_leaves: 31

  random_forest:
    name: "Random Forest Classifier"
    family: "ensemble"
    notes: "Bagging ensemble. Good baseline."
    default_hyperparams:
      n_estimators: 100
      max_depth: null

  logistic_regression:
    name: "Logistic Regression"
    family: "linear"
    notes: "Simple linear baseline. Try first."
    default_hyperparams:
      C: 1.0
      max_iter: 1000

  mlp:
    name: "Multi-Layer Perceptron"
    family: "neural_network"
    notes: "Simple neural network. Needs feature scaling."
    default_hyperparams:
      hidden_layer_sizes: [100, 50]
      learning_rate_init: 0.001
      max_iter: 200
```

Add domain-specific models (transformers, custom architectures) by appending entries to this file. The agent discovers them automatically.
