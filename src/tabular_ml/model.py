from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from .config import TrainingConfig


def get_device(config: TrainingConfig) -> torch.device:
    preferred = config.device.lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TabTransformerConfig:
    categorical_cardinalities: Dict[str, int]
    numeric_feature_dim: int
    config: TrainingConfig


class TabTransformer(nn.Module):
    def __init__(self, tt_config: TabTransformerConfig):
        super().__init__()
        self.tt_config = tt_config
        self.config = tt_config.config
        self.cat_columns = list(tt_config.categorical_cardinalities.keys())
        embedding_dim = self.config.embedding_dim
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num_embeddings=cardinality, embedding_dim=embedding_dim)
            for col, cardinality in tt_config.categorical_cardinalities.items()
        })
        self.input_projection = nn.Linear(embedding_dim, self.config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.d_model * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        if tt_config.numeric_feature_dim > 0:
            self.numeric_norm: nn.Module = nn.LayerNorm(tt_config.numeric_feature_dim)
            self.numeric_dim = tt_config.numeric_feature_dim
        else:
            self.numeric_norm = nn.Identity()
            self.numeric_dim = 0
        mlp_layers: List[nn.Module] = []
        prev_dim = self.config.d_model + self.numeric_dim
        for hidden_dim in self.config.mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.LayerNorm(hidden_dim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim
        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, categorical_inputs: Dict[str, torch.Tensor], numeric_inputs: torch.Tensor) -> torch.Tensor:
        batch_size = None
        tokens: List[torch.Tensor] = []
        for col in self.cat_columns:
            emb = self.embeddings[col](categorical_inputs[col])  # (batch,)
            if batch_size is None:
                batch_size = emb.size(0)
            emb = emb.unsqueeze(1)  # (batch, 1, embed_dim)
            projected = self.input_projection(emb)
            tokens.append(projected)
        if tokens:
            cat_tensor = torch.cat(tokens, dim=1)
        else:
            if batch_size is None:
                batch_size = numeric_inputs.size(0)
            cat_tensor = torch.zeros(batch_size, 0, self.config.d_model, device=numeric_inputs.device)
        cls_token = self.cls_token.expand(cat_tensor.size(0), -1, -1)
        transformer_input = torch.cat([cls_token, cat_tensor], dim=1)
        transformer_output = self.transformer(transformer_input)
        cls_output = transformer_output[:, 0, :]
        if self.numeric_dim > 0:
            numeric_features = self.numeric_norm(numeric_inputs)
            combined = torch.cat([cls_output, numeric_features], dim=1)
        else:
            combined = cls_output
        return self.mlp(combined).squeeze(-1)
