# ADR-0009: XGBoost as Default Model with Pluggable Featurizer Pipeline

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-31 |
| **Author** | Prannaya Gupta |
| **Supersedes** | (none) |
| **Category** | Technology Choice |

## Context

Helios needs a default model that works out of the box for the widest range of ML tasks. The scaffolded `train.py` must include a working training pipeline that the autoresearch agent can immediately iterate on. The default must be:

1. **Reliable**: works on tabular data without extensive preprocessing
2. **Fast**: trains in seconds to minutes, not hours
3. **Competitive**: produces strong baselines that are hard to beat with simple alternatives
4. **Interpretable**: the agent can reason about what hyperparameters to change

Separately, the feature engineering pipeline must be extensible without modifying the evaluation infrastructure (which is READ-ONLY per ADR-0002).

## Options Considered

### Option 1: XGBoost with Composite Featurizer

XGBoost as the default gradient boosting implementation. Feature engineering via a pluggable featurizer pipeline (NumericFeaturizer + CategoricalFeaturizer composed via CompositeFeaturizer) with a scikit-learn-like fit/transform interface.

Trade-offs: XGBoost is the most mature gradient boosting library. The featurizer pipeline is simple but extensible.

### Option 2: LightGBM

Microsoft's gradient boosting framework. Often faster than XGBoost on large datasets.

Trade-offs: competitive performance. But installation can be problematic (requires libomp on macOS), and XGBoost has a longer track record.

### Option 3: scikit-learn GradientBoostingClassifier

Pure Python implementation in scikit-learn.

Trade-offs: no additional dependency (scikit-learn is already required for metrics). But significantly slower than XGBoost/LightGBM and missing key features (GPU support, built-in regularization).

### Option 4: Neural Network (PyTorch/TensorFlow)

Small MLP as the default model.

Trade-offs: more flexible architecture. But requires GPU for reasonable training speed on non-trivial datasets, adds large dependencies, and is harder to configure correctly as a default.

### Option 5: No Default — Empty train.py

Ship an empty training script that the agent builds from scratch.

Trade-offs: maximally flexible. But the cold-start problem is severe — the agent must write an entire pipeline before running its first experiment.

## Decision

**We will use XGBoost as the default model and a pluggable CompositeFeaturizer as the feature pipeline** because XGBoost provides the best combination of reliability, speed, and competitiveness for tabular data, and the featurizer pipeline separates feature engineering from model training while respecting the READ-ONLY boundary.

## Rationale

XGBoost is the default for a pragmatic reason: it is the single model most likely to produce a competitive baseline on arbitrary tabular data. Kaggle competition analysis shows that gradient boosting methods dominate tabular tasks, and XGBoost is the most widely deployed implementation.

The featurizer pipeline follows a deliberate design:
- `BaseFeaturizer` (abstract) → `NumericFeaturizer`, `CategoricalFeaturizer` → `CompositeFeaturizer`
- scikit-learn-like `fit(df) → transform(df)` interface
- The agent modifies how `train.py` *uses* the featurizer, not the featurizer itself (consistent with ADR-0002)
- The `get_default_featurizer()` function provides a single entry point

LightGBM is bundled in `requirements.txt` as an available alternative — the agent can switch to it by modifying `train.py`. This is listed as the second suggestion in `program.md`'s experiment ideas.

## Consequences

### Positive

- Working baseline from the first experiment — no cold start
- XGBoost handles mixed numeric/categorical data, missing values, and class imbalance
- Featurizer pipeline is extensible: add new featurizers by subclassing BaseFeaturizer
- Agent can switch model type by modifying train.py without touching featurizers

### Negative

- Bias toward gradient boosting — the default influences the agent's exploration trajectory
- XGBoost dependency adds ~100MB to the virtual environment
- The featurizer pipeline is simple (passthrough + one-hot) — real projects will need custom featurizers

### Neutral

- LightGBM is included in requirements.txt but not used by default — available for the agent's second experiment

## References

- [XGBoost](https://xgboost.readthedocs.io/) — Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016
- [Why Gradient Boosting Dominates Tabular Data](https://arxiv.org/abs/2207.08815) — Grinsztajn et al., 2022
- `templates/train.py` — default XGBoost training pipeline
- `templates/features/featurizers.py` — pluggable featurizer pipeline
- `templates/requirements.txt` — includes both XGBoost and LightGBM
