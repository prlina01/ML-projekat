from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold, ParameterGrid
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from .config import TrainingConfig
from .model import TabTransformer, TabTransformerConfig, get_device


@dataclass
class Batch:
    categorical: Dict[str, torch.Tensor]
    numeric: torch.Tensor
    target: torch.Tensor
    target_raw: Optional[torch.Tensor] = None


def _apply_hyperparameters(config: TrainingConfig, params: Dict[str, float]) -> TrainingConfig:
    if not params:
        return config
    allowed = {k: v for k, v in params.items() if hasattr(config, k) and k != "hyperparams"}
    if not allowed:
        return config
    tuned = replace(config, **allowed)
    cv_epochs = tuned.max_epochs
    if config.cv_max_epochs is not None:
        cv_epochs = config.cv_max_epochs
    return replace(tuned, max_epochs=cv_epochs)


def _iterate_batches(
    categorical_data: Dict[str, np.ndarray],
    numeric_data: np.ndarray,
    targets: np.ndarray,
    config: TrainingConfig,
    shuffle: bool = False,
    seed: int | None = None,
    raw_targets: Optional[np.ndarray] = None,
) -> Iterable[Batch]:
    num_samples = targets.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, num_samples, config.batch_size):
        end = min(start + config.batch_size, num_samples)
        batch_idx = indices[start:end]
        cat_batch = {col: torch.from_numpy(data[batch_idx]).long() for col, data in categorical_data.items()}
        num_batch = torch.from_numpy(numeric_data[batch_idx]).float()
        target_batch = torch.from_numpy(targets[batch_idx]).float()
        raw_batch = None
        if raw_targets is not None:
            raw_batch = torch.from_numpy(raw_targets[batch_idx]).float()
        yield Batch(categorical=cat_batch, numeric=num_batch, target=target_batch, target_raw=raw_batch)


def train_one_epoch(
    model: TabTransformer,
    categorical_data: Dict[str, np.ndarray],
    numeric_data: np.ndarray,
    targets: np.ndarray,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    losses: List[float] = []
    for batch in _iterate_batches(categorical_data, numeric_data, targets, config, shuffle=True, seed=seed):
        optimizer.zero_grad()
        cat_inputs = {k: v.to(device) for k, v in batch.categorical.items()}
        num_inputs = batch.numeric.to(device)
        target = batch.target.to(device)
        preds = model(cat_inputs, num_inputs)
        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate(
    model: TabTransformer,
    categorical_data: Dict[str, np.ndarray],
    numeric_data: np.ndarray,
    targets: np.ndarray,
    targets_raw: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
    inverse_target_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    model.eval()
    preds_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in _iterate_batches(
            categorical_data,
            numeric_data,
            targets,
            config,
            shuffle=False,
            raw_targets=targets_raw,
        ):
            cat_inputs = {k: v.to(device) for k, v in batch.categorical.items()}
            num_inputs = batch.numeric.to(device)
            target = batch.target.to(device)
            preds = model(cat_inputs, num_inputs)
            preds_all.append(preds.cpu().numpy())
            if batch.target_raw is None:
                raise ValueError("Raw targets required for evaluation but missing in batch.")
            targets_all.append(batch.target_raw.cpu().numpy())
    transformed_preds = np.concatenate(preds_all)
    truth_raw = np.concatenate(targets_all)
    preds_raw = inverse_target_fn(transformed_preds)
    rmse = np.sqrt(np.mean((truth_raw - preds_raw) ** 2))
    return float(rmse)


def cross_validate(
    config: TrainingConfig,
    categorical_data: Dict[str, np.ndarray],
    numeric_data: np.ndarray,
    targets: np.ndarray,
    targets_raw: np.ndarray,
    categorical_cardinalities: Dict[str, int],
    inverse_target_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[Dict[str, float], TabTransformerConfig, float]:
    device = get_device(config)
    hyperparams = config.hyperparams or {
        "weight_decay": [config.weight_decay, max(config.weight_decay * 0.1, 1e-6)],
        "dropout": [config.dropout, min(0.3, config.dropout + 0.1)],
    }
    param_grid = list(ParameterGrid(hyperparams)) or [{}]
    best_score = float("inf")
    best_params: Dict[str, float] = {}
    best_model_config: TabTransformerConfig | None = None
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    planned_epochs = max(len(param_grid) * config.cv_folds * config.max_epochs, 1)
    with tqdm(total=planned_epochs, desc="Cross-validation", unit="epoch") as progress:
        for params in param_grid:
            tuned_config = _apply_hyperparameters(config, params)
            fold_scores: List[float] = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(targets), start=1):
                train_cat = {col: data[train_idx] for col, data in categorical_data.items()}
                val_cat = {col: data[val_idx] for col, data in categorical_data.items()}
                train_num = numeric_data[train_idx]
                val_num = numeric_data[val_idx]
                train_target = targets[train_idx]
                val_target = targets[val_idx]
                train_target_raw = targets_raw[train_idx]
                val_target_raw = targets_raw[val_idx]

                tt_config = TabTransformerConfig(
                    categorical_cardinalities=categorical_cardinalities,
                    numeric_feature_dim=numeric_data.shape[1],
                    config=tuned_config,
                )
                model = TabTransformer(tt_config=tt_config).to(device)
                optimizer = AdamW(
                    model.parameters(),
                    lr=tuned_config.learning_rate,
                    weight_decay=tuned_config.weight_decay,
                )
                scheduler = CosineAnnealingLR(optimizer, T_max=max(1, tuned_config.max_epochs))

                best_val = float("inf")
                epochs_no_improve = 0
                for epoch in range(tuned_config.max_epochs):
                    seed = tuned_config.random_state + epoch + fold_idx
                    train_one_epoch(model, train_cat, train_num, train_target, tuned_config, optimizer, device, seed)
                    scheduler.step()
                    val_rmse = evaluate(
                        model,
                        val_cat,
                        val_num,
                        val_target,
                        val_target_raw,
                        tuned_config,
                        device,
                        inverse_target_fn,
                    )
                    if val_rmse + 1e-6 < best_val:
                        best_val = val_rmse
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    progress.update(1)
                    progress.set_postfix({"fold": f"{fold_idx}/{config.cv_folds}", "rmse": f"{val_rmse:.0f}"}, refresh=False)
                    if epochs_no_improve >= tuned_config.patience:
                        break
                fold_scores.append(best_val)
            mean_score = float(np.mean(fold_scores))
            if mean_score < best_score:
                best_score = mean_score
                best_params = params
                best_model_config = TabTransformerConfig(
                    categorical_cardinalities=categorical_cardinalities,
                    numeric_feature_dim=numeric_data.shape[1],
                    config=tuned_config,
                )
        progress.total = max(progress.n, 1)
        progress.refresh()
    if best_model_config is None:
        raise RuntimeError("No model configuration selected during cross-validation")
    return best_params, best_model_config, best_score


def train_final_model(
    config: TrainingConfig,
    categorical_data: Dict[str, np.ndarray],
    numeric_data: np.ndarray,
    targets: np.ndarray,
    categorical_cardinalities: Dict[str, int],
) -> Tuple[TabTransformer, float]:
    device = get_device(config)
    tt_config = TabTransformerConfig(
        categorical_cardinalities=categorical_cardinalities,
        numeric_feature_dim=numeric_data.shape[1],
        config=config,
    )
    model = TabTransformer(tt_config=tt_config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.max_epochs))

    best_loss = float("inf")
    epochs_no_improve = 0
    planned_epochs = max(config.max_epochs, 1)
    with tqdm(total=planned_epochs, desc="Final training", unit="epoch") as progress:
        for epoch in range(config.max_epochs):
            seed = config.random_state + epoch
            epoch_loss = train_one_epoch(model, categorical_data, numeric_data, targets, config, optimizer, device, seed)
            scheduler.step()
            if epoch_loss + 1e-6 < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            progress.update(1)
            progress.set_postfix({"loss": f"{epoch_loss:.4f}"}, refresh=False)
            if epochs_no_improve >= config.patience:
                break
        progress.total = max(progress.n, 1)
        progress.refresh()
    return model, best_loss
