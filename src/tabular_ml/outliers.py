from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

from .config import TrainingConfig


def isolation_forest_filter(df: pd.DataFrame, numeric_columns: List[str], config: TrainingConfig) -> pd.DataFrame:
    if not numeric_columns:
        return df
    numeric_subset = df[numeric_columns].copy()
    # Fill missing with column medians to keep the estimator stable.
    medians = numeric_subset.median()
    numeric_subset = numeric_subset.fillna(medians)
    clf = IsolationForest(random_state=config.random_state, contamination="auto")
    clf.fit(numeric_subset)
    scores = clf.score_samples(numeric_subset)
    normalized_scores = MinMaxScaler().fit_transform((-scores).reshape(-1, 1)).reshape(-1)
    mask = normalized_scores <= config.anomaly_threshold
    filtered_df = df.loc[mask].reset_index(drop=True)
    return filtered_df


def loess_soft_clipping(df: pd.DataFrame, numeric_columns: List[str], config: TrainingConfig) -> pd.DataFrame:
    adjusted = df.copy()
    for col in numeric_columns:
        series = adjusted[col].astype(float)
        if series.isna().all():
            continue
        x = np.arange(len(series))
        valid_mask = series.notna()
        if valid_mask.sum() < 5:
            continue
        loess_fit = lowess(series[valid_mask], x[valid_mask], frac=config.loess_frac, it=0, return_sorted=False)
        residuals = series[valid_mask] - loess_fit
        mad = np.median(np.abs(residuals - np.median(residuals)))
        if mad == 0:
            continue
        limit = config.loess_mad_multiplier * mad
        upper = loess_fit + limit
        lower = loess_fit - limit
        clipped = np.clip(series[valid_mask], lower, upper)
        series.loc[valid_mask] = clipped
        adjusted[col] = series
    return adjusted
