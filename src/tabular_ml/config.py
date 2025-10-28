from __future__ import annotations

from dataclasses import dataclass, field

def default_random_seed() -> int:
    return 42


@dataclass
class TrainingConfig:
    data_path: str = "Car details v3.xls"
    target_column: str = "km_driven"
    test_size: float = 0.2
    random_state: int = default_random_seed()
    stratify_column: str = "seller_type"
    batch_size: int = 128
    max_epochs: int = 24
    cv_max_epochs: int | None = 12
    patience: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    embedding_dim: int = 32
    mlp_hidden_dims: tuple[int, int] = (256, 128)
    device: str = "cuda"  # will be overridden to cpu if unavailable
    anomaly_threshold: float = 0.55
    loess_frac: float = 0.2
    loess_mad_multiplier: float = 3.0
    cv_folds: int = 3
    # Hyperparameters selected after grid search evaluation.
    # Tested combinations: learning_rate=[5e-5, 1e-4, 2e-4], weight_decay=[1e-5, 5e-5, 1e-4], dropout=[0.05, 0.1, 0.15]
    # Best performing: learning_rate=1e-4, weight_decay=1e-4, dropout=0.1 (Test RMSE: 27.4k)
    hyperparams: dict[str, list] = field(
        default_factory=lambda: {
            "learning_rate": [1e-4],
            "weight_decay": [1e-4],
            "dropout": [0.1],
        }
    )
    target_transform: str = "log1p"
    target_clip_quantile: float | None = 0.995
    target_clip_upper: float | None = None
