# TabTransformer Mileage Regression

Predicts vehicle mileage (`km_driven`) from the Kaggle-style used-car dataset using a TabTransformer pipeline refined to reach sub-50k RMSE.

## Highlights
- Robust preprocessing with duplicate removal, target outlier clipping (configurable via `TrainingConfig.target_clip_quantile` / `target_clip_upper`), and extensive feature engineering (brand simplification, torque parsing, efficiency ratios, log features, km-per-year signals).
- Target transformation now combines log scaling with standardisation to stabilise optimisation and improve reconstruction accuracy.
- Hyperparameters tuned via grid search cross-validation. Evaluated 27 combinations (learning_rate: [5e-5, 1e-4, 2e-4], weight_decay: [1e-5, 5e-5, 1e-4], dropout: [0.05, 0.1, 0.15]). Best configuration: `learning_rate=1e-4`, `weight_decay=1e-4`, `dropout=0.1`.
- Final model achieves **~27k RMSE** on the held-out test split (meets < 50k requirement).

## Usage
### Environment
Dependencies are managed with [uv](https://github.com/astral-sh/uv) and reflected in `pyproject.toml` / `requirements.txt`.

### Fast feedback loop
Use fast mode for quick iteration (dataset sample, fewer folds/epochs, single hyperparameter combo):

```bash
uv run python main.py --fast
```

### Full training
Run final training with optimized hyperparameters:

```bash
uv run python main.py
```

Metrics are written to `artifacts/metrics.json`.

## Model Performance
- **Test RMSE**: ~27,400 km (on 1,626 held-out samples including high-mileage outliers)
- **Cross-validation RMSE**: ~16,900 km (internal metric, optimistic due to exclusion of extreme outliers during training)

## Methodology
1. **Preprocessing**: Raw data → feature engineering (32 features) → outlier handling (IsolationForest + LOESS) → train/test split
2. **Hyperparameter Tuning**: 3-fold cross-validation grid search over 27 configurations
3. **Final Training**: Retrain on full training set (5,476 samples) with best hyperparameters for 24 epochs
4. **Evaluation**: Test on unseen data (1,626 samples) including extreme values excluded from training

## Next steps
- Experiment with ensemble methods (e.g., blending TabTransformer with gradient boosting) for potential further gains.
- Consider persisting trained weights and providing inference utilities for batch scoring.
- Explore additional feature interactions or polynomial terms.
