# TabTransformer Mileage Regression

Predicts vehicle mileage (`km_driven`) from the Kaggle-style used-car dataset using a TabTransformer pipeline refined to reach sub-50k RMSE.

## Highlights
- Robust preprocessing with duplicate removal, target outlier clipping (configurable via `TrainingConfig.target_clip_quantile` / `target_clip_upper`), and extensive feature engineering (brand simplification, torque parsing, efficiency ratios, log features, km-per-year signals).
- Target transformation now combines log scaling with standardisation to stabilise optimisation and improve reconstruction accuracy.
- Cross-validation tuned hyperparameters (`dropout=0.1`, `learning_rate=1e-4`, `weight_decay=1e-4`) with a lighter CV schedule (`cv_max_epochs`) to keep full runs tractable.
- Final model achieves **15.5k RMSE** on the held-out test split (`artifacts/metrics.json`).

## Usage
### Environment
Dependencies are managed with [uv](https://github.com/astral-sh/uv) and reflected in `pyproject.toml` / `requirements.txt`.

### Fast feedback loop
Use fast mode for quick iteration (dataset sample, fewer folds/epochs):

```bash
uv run python main.py --fast
```

### Full training
Run full cross-validation + final fit:

```bash
uv run python main.py
```

Metrics are written to `artifacts/metrics.json`.

## Next steps
- Experiment with wider hyperparameter grids or ensembling (e.g., blending with tree-based models) for potential additional gains.
- Consider persisting trained weights and providing inference utilities for batch scoring.
