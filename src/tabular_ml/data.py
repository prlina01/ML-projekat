from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import TrainingConfig


def load_dataset(config: TrainingConfig) -> pd.DataFrame:
    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    suffix = path.suffix.lower()
    df: pd.DataFrame | None = None
    if suffix in {".csv"}:
        df = pd.read_csv(path)
    else:
        # Some distributions of the dataset ship a CSV with an .xls extension.
        try:
            df = pd.read_csv(path)
        except Exception:
            # Fallback to Excel readers when the content is a true spreadsheet.
            engine = "openpyxl" if suffix == ".xlsx" else None
            df = pd.read_excel(path, engine=engine)
    return df


def train_test_split(config: TrainingConfig, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    stratify_vals = None
    if config.stratify_column in df.columns:
        stratify_vals = df[config.stratify_column]
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_vals,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
