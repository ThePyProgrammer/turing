# ML Researcher Memory

## Goal

{{TASK_DESCRIPTION}}

Primary metric: {{TARGET_METRIC}} ({{METRIC_DIRECTION}} is better).

## Best Result

No experiments completed yet. Run `/helios:train` to begin.

## Observations

- Initial model: XGBoost with default hyperparams (n_estimators=100, max_depth=4, lr=0.1)
- Config file: config.yaml controls all hyperparameters
- Sweep tool available: `python scripts/sweep.py` for systematic grid search
- Per-experiment branches: `exp/NNN-description` preserves all variants

## Failed Approaches

(none yet)

## Promising Directions

- Hyperparameter sweep across n_estimators, max_depth, learning_rate
- LightGBM as alternative to XGBoost
- Feature engineering: add domain-specific features
- Try different model architectures (RandomForest, MLP)
