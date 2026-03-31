# Model Artifact Contract

Version: 1
Last updated: {{PROJECT_NAME}} initial scaffold

## Bundle Format

The trained model is saved as a joblib bundle at `models/best/model.joblib` containing:

```python
{
    "model": <fitted model object>,
    "featurizer": <fitted CompositeFeaturizer>,
    "config": <dict of training config>,
    "contract_version": 1
}
```

## Metadata

`models/best/metadata.json` contains:

```json
{
    "contract_version": 1,
    "model_type": "xgboost",
    "experiment_id": "exp-001",
    "metrics": {"{{TARGET_METRIC}}": 0.0},
    "feature_names": [],
    "created_at": "ISO-8601"
}
```

## Consumer Contract

Any service loading this model expects:
- `bundle["model"]` has a `.predict()` method accepting a feature matrix
- `bundle["featurizer"]` has a `.transform(df)` method returning a DataFrame
- `bundle.get("contract_version", 0)` must equal 1

If `contract_version` doesn't match, the consumer should log a warning and fall back to a default/rules-based approach.

## Breaking Changes

Increment `contract_version` when changing:
- Feature schema (different featurizer output shape)
- Label encoding (different label_map)
- Bundle key names
- Model input/output format
