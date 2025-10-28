from __future__ import annotations

import json
import random
import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from tabular_ml.config import TrainingConfig
from tabular_ml.data import load_dataset, train_test_split
from tabular_ml.model import get_device
from tabular_ml.preprocessing import (
    ModelInputs,
    PreprocessingArtifacts,
    build_artifacts,
    get_categorical_cardinalities,
    inverse_transform_target,
    prepare_model_inputs,
    preprocess_full_dataset,
)
from tabular_ml.train import cross_validate, evaluate, train_final_model


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _summarize_cardinalities(cardinalities: Dict[str, int]) -> Dict[str, int]:
    # Only keep a subset for readability in logs.
    preview = dict(list(cardinalities.items())[:5])
    preview["total_categorical_columns"] = len(cardinalities)
    return preview


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TabTransformer mileage regression model")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable a quicker configuration with fewer folds/epochs and a sampled dataset for rapid feedback.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = TrainingConfig()
    if args.fast:
        fast_hyperparams = {
            "learning_rate": [config.learning_rate],
            "weight_decay": [config.weight_decay],
            "dropout": [config.dropout],
        }
        config = replace(
            config,
            max_epochs=max(6, config.max_epochs // 3),
            patience=2,
            cv_folds=3,
            batch_size=max(config.batch_size, 256),
            hyperparams=fast_hyperparams,
        )
        print("Fast mode enabled: using reduced epochs, folds, and hyperparameter combinations.")

    _set_global_seed(config.random_state)
    dataset_path = Path(config.data_path)
    print(f"Loading dataset from {dataset_path.resolve()}")
    raw_df = load_dataset(config)
    print(f"Initial shape: {raw_df.shape}")

    if args.fast and len(raw_df) > 2000:
        sample_size = min(2000, len(raw_df))
        raw_df = raw_df.sample(sample_size, random_state=config.random_state).reset_index(drop=True)
        print(f"Fast mode: sampled {sample_size} rows for training.")

    train_df, test_df = train_test_split(config, raw_df)
    print(f"Training rows (raw): {len(train_df)} | Test rows (raw): {len(test_df)}")

    train_df = preprocess_full_dataset(train_df, config)
    print(f"Training rows after preprocessing: {len(train_df)}")
    
    # Build artifacts from training data (including seats mode)
    artifacts: PreprocessingArtifacts = build_artifacts(train_df, config)
    
    # Apply ONLY feature cleaning/engineering to test (no outlier removal/target clipping)
    # Use training set's seats mode for consistency
    from tabular_ml.preprocessing import clean_dataframe, feature_engineering
    test_df = clean_dataframe(test_df, config, apply_target_clipping=False, seats_mode=artifacts.seats_mode)
    test_df = feature_engineering(test_df)

    model_inputs: ModelInputs = prepare_model_inputs(train_df, test_df, artifacts, config)
    categorical_cardinalities = get_categorical_cardinalities(artifacts)

    print("Categorical embedding summary:")
    print(json.dumps(_summarize_cardinalities(categorical_cardinalities), indent=2))
    print(f"Numeric feature count: {model_inputs.train_numeric.shape[1]}")

    inverse_target_fn = lambda arr: inverse_transform_target(arr, config, model_inputs.target_stats)

    # Hyperparameter search via cross-validation
    print("Running cross-validation for hyperparameter tuning...")
    best_params, tuned_config_template, cv_rmse = cross_validate(
        config,
        model_inputs.train_categorical,
        model_inputs.train_numeric,
        model_inputs.y_train,
        model_inputs.y_train_raw,
        categorical_cardinalities,
        inverse_target_fn,
    )
    tuned_config = tuned_config_template.config
    tuned_values: Dict[str, float] = {}
    hyperparam_keys = (config.hyperparams or {}).keys()
    if hyperparam_keys:
        tuned_values = {
            key: getattr(tuned_config, key)
            for key in hyperparam_keys
            if hasattr(tuned_config, key)
        }
    if not tuned_values:
        tuned_values = {
            "dropout": tuned_config.dropout,
            "weight_decay": tuned_config.weight_decay,
        }
    config = replace(config, **tuned_values)
    print("Best hyperparameters:")
    print(json.dumps(best_params, indent=2))
    print(f"Cross-validation RMSE: {cv_rmse:.2f} km")

    # Final training on full training data
    print("Training final TabTransformer model...")
    model, final_loss = train_final_model(
        config,
        model_inputs.train_categorical,
        model_inputs.train_numeric,
        model_inputs.y_train,
        categorical_cardinalities,
    )
    print(f"Final training loss (MSE): {final_loss:.4f}")

    device = get_device(config)
    test_rmse = evaluate(
        model,
        model_inputs.test_categorical,
        model_inputs.test_numeric,
        model_inputs.y_test,
        model_inputs.y_test_raw,
        config,
        device,
        inverse_target_fn,
    )
    print(f"Test RMSE: {test_rmse:.2f} km")

    report = {
        "dataset_path": str(dataset_path),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "final_train_loss": float(final_loss),
        "test_rmse": float(test_rmse),
        "cv_rmse": float(cv_rmse),
        "best_hyperparameters": best_params,
    }
    report_path = Path("artifacts")
    report_path.mkdir(exist_ok=True)
    with open(report_path / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved metrics to {report_path / 'metrics.json'}")


if __name__ == "__main__":
    main()
