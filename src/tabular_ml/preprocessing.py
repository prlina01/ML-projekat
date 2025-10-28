from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import TrainingConfig
from .outliers import isolation_forest_filter, loess_soft_clipping


NUMERIC_REGEX = re.compile(r"([-+]?[0-9]*\.?[0-9]+)")
TORQUE_VALUE_REGEX = re.compile(r"([-+]?\d*\.?\d+)")


@dataclass
class PreprocessingArtifacts:
    numeric_columns: List[str]
    categorical_columns: List[str]
    scaler: RobustScaler
    category_maps: Dict[str, Dict[str, int]]
    numeric_imputers: Dict[str, float]


@dataclass
class ModelInputs:
    train_categorical: Dict[str, np.ndarray]
    train_numeric: np.ndarray
    y_train: np.ndarray
    y_train_raw: np.ndarray
    test_categorical: Dict[str, np.ndarray]
    test_numeric: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray
    target_stats: "TargetStats"


@dataclass
class TargetStats:
    mean: float
    std: float


def _extract_numeric(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.replace(",", "", regex=False)
    values = extracted.str.extract(NUMERIC_REGEX, expand=False)
    return pd.to_numeric(values, errors="coerce")


def _parse_torque(raw_value: str | float | int) -> Tuple[float, float]:
    if pd.isna(raw_value):
        return (np.nan, np.nan)
    text = str(raw_value).lower().replace(",", "")
    matches = TORQUE_VALUE_REGEX.findall(text)
    torque_nm = np.nan
    rpm = np.nan
    if matches:
        try:
            torque_val = float(matches[0])
        except ValueError:
            torque_val = np.nan
        if not np.isnan(torque_val):
            if "kgm" in text or "kg m" in text:
                torque_nm = torque_val * 9.80665
            else:
                torque_nm = torque_val
    if len(matches) > 1:
        try:
            rpm = float(matches[1])
        except ValueError:
            rpm = np.nan
    return (torque_nm, rpm)


def clean_dataframe(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.dropna(subset=[config.target_column])

    upper_bound: float | None = None
    if config.target_clip_quantile is not None:
        quantile_value = float(cleaned[config.target_column].quantile(config.target_clip_quantile))
        upper_bound = quantile_value
    if config.target_clip_upper is not None:
        upper_bound = min(upper_bound, config.target_clip_upper) if upper_bound is not None else float(config.target_clip_upper)
    if upper_bound is not None:
        original_rows = len(cleaned)
        cleaned = cleaned[cleaned[config.target_column] <= upper_bound].reset_index(drop=True)
        removed = original_rows - len(cleaned)
        if removed > 0:
            print(f"Removed {removed} rows exceeding target upper bound {upper_bound:.0f}.")

    if "mileage" in cleaned.columns:
        cleaned["mileage"] = _extract_numeric(cleaned["mileage"])
    if "engine" in cleaned.columns:
        cleaned["engine"] = _extract_numeric(cleaned["engine"])
    if "max_power" in cleaned.columns:
        cleaned["max_power"] = _extract_numeric(cleaned["max_power"])
    if "torque" in cleaned.columns:
        torque_components = cleaned["torque"].apply(_parse_torque)
        torque_df = pd.DataFrame(list(torque_components), columns=["torque_nm", "torque_rpm"])
        cleaned["torque_nm"] = torque_df["torque_nm"].astype(float)
        cleaned["torque_rpm"] = torque_df["torque_rpm"].astype(float)
        cleaned["torque"] = _extract_numeric(cleaned["torque"])
    if "seats" in cleaned.columns:
        cleaned["seats"] = cleaned["seats"].fillna(cleaned["seats"].mode(dropna=True).iloc[0])

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != config.target_column:
            cleaned[col] = cleaned[col].astype(float)

    return cleaned


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    current_year = 2025
    if "year" in enriched.columns:
        enriched["age"] = (current_year - enriched["year"]).clip(lower=0)
    else:
        enriched["age"] = np.nan
    if "selling_price" in enriched.columns:
        age = enriched["age"].replace(0, np.nan)
        enriched["price_per_year"] = enriched["selling_price"] / age
        enriched["price_per_year"] = enriched["price_per_year"].fillna(enriched["selling_price"])
    else:
        enriched["price_per_year"] = np.nan

    if {"km_driven", "age"}.issubset(enriched.columns):
        age_years = enriched["age"].where(enriched["age"] > 0, np.nan)
        enriched["km_per_year"] = enriched["km_driven"] / age_years
        enriched["km_per_year"] = enriched["km_per_year"].replace([np.inf, -np.inf], np.nan)
        enriched["log_km_per_year"] = np.log1p(enriched["km_per_year"].clip(lower=0))
    else:
        enriched["km_per_year"] = np.nan
        enriched["log_km_per_year"] = np.nan

    if "name" in enriched.columns:
        simplified = (
            enriched["name"].astype(str).str.strip().str.split().apply(lambda parts: " ".join(parts[:2]).lower() if parts else "unknown")
        )
        enriched["brand"] = (
            enriched["name"].astype(str).str.strip().str.split().str[0].str.lower().fillna("unknown")
        )
        enriched["name"] = simplified.fillna("unknown")

    if {"max_power", "engine"}.issubset(enriched.columns):
        engine_litres = (enriched["engine"] / 1000.0).replace(0, np.nan)
        enriched["power_per_litre"] = enriched["max_power"] / engine_litres
        enriched["power_per_litre"] = enriched["power_per_litre"].replace([np.inf, -np.inf], np.nan)
    if {"torque_nm", "engine"}.issubset(enriched.columns):
        enriched["torque_per_litre"] = enriched["torque_nm"] / enriched["engine"]
        enriched["torque_per_litre"] = enriched["torque_per_litre"].replace([np.inf, -np.inf], np.nan)
    if {"selling_price", "max_power"}.issubset(enriched.columns):
        enriched["price_per_power"] = enriched["selling_price"] / enriched["max_power"].replace(0, np.nan)
        enriched["price_per_power"] = enriched["price_per_power"].replace([np.inf, -np.inf], np.nan)
    if {"selling_price", "engine"}.issubset(enriched.columns):
        enriched["price_per_engine"] = enriched["selling_price"] / enriched["engine"].replace(0, np.nan)
        enriched["price_per_engine"] = enriched["price_per_engine"].replace([np.inf, -np.inf], np.nan)
    if {"selling_price", "km_driven"}.issubset(enriched.columns):
        enriched["price_per_km"] = enriched["selling_price"] / enriched["km_driven"].replace(0, np.nan)
        enriched["price_per_km"] = enriched["price_per_km"].replace([np.inf, -np.inf], np.nan)

    for col in ["selling_price", "engine", "max_power", "mileage", "torque_nm"]:
        if col in enriched.columns:
            enriched[f"log_{col}"] = np.log1p(enriched[col].clip(lower=0))

    if "age" in enriched.columns:
        enriched["age_squared"] = enriched["age"] ** 2
        enriched["age_log"] = np.log1p(enriched["age"])
    return enriched


def apply_outlier_strategies(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if config.target_column in numeric_cols:
        numeric_cols.remove(config.target_column)
    filtered = isolation_forest_filter(df, numeric_cols, config)
    clipped = loess_soft_clipping(filtered, numeric_cols, config)
    return clipped


def preprocess_full_dataset(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    cleaned = clean_dataframe(df, config)
    engineered = feature_engineering(cleaned)
    processed = apply_outlier_strategies(engineered, config)
    return processed.reset_index(drop=True)


def identify_columns(df: pd.DataFrame, config: TrainingConfig) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if config.target_column in numeric_cols:
        numeric_cols.remove(config.target_column)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def fit_scaler(train_numeric: pd.DataFrame) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(train_numeric.values)
    return scaler


def encode_categories(df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Dict[str, int]]:
    category_maps: Dict[str, Dict[str, int]] = {}
    for col in categorical_columns:
        values = df[col].astype(str).fillna("<UNK>")
        unique_vals = pd.Index(sorted(values.unique()))
        mapping = {"<UNK>": 0}
        next_index = 1
        for val in unique_vals:
            if val == "<UNK>":
                continue
            mapping[val] = next_index
            next_index += 1
        category_maps[col] = mapping
    return category_maps


def build_artifacts(train_df: pd.DataFrame, config: TrainingConfig) -> PreprocessingArtifacts:
    numeric_cols, categorical_cols = identify_columns(train_df, config)
    numeric_imputers: Dict[str, float] = {}
    for col in numeric_cols:
        numeric_imputers[col] = float(train_df[col].median(skipna=True))
    train_numeric = train_df[numeric_cols].fillna(pd.Series(numeric_imputers))
    scaler = fit_scaler(train_numeric)
    category_maps = encode_categories(train_df, categorical_cols)
    return PreprocessingArtifacts(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        scaler=scaler,
        category_maps=category_maps,
        numeric_imputers=numeric_imputers,
    )


def transform_with_artifacts(
    df: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    numeric_df = df[artifacts.numeric_columns].copy()
    for col, value in artifacts.numeric_imputers.items():
        numeric_df[col] = numeric_df[col].fillna(value)
    scaled_numeric = artifacts.scaler.transform(numeric_df.values)

    categorical_arrays: Dict[str, np.ndarray] = {}
    for col in artifacts.categorical_columns:
        values = df[col].astype(str).fillna("<UNK>")
        mapping = artifacts.category_maps[col]
        encoded = values.map(mapping).fillna(0).astype(int).to_numpy()
        categorical_arrays[col] = encoded

    return categorical_arrays, scaled_numeric.astype(np.float32)


def _base_transform(values: np.ndarray, config: TrainingConfig) -> np.ndarray:
    arr = values.astype(np.float32)
    if config.target_transform == "log1p":
        arr = np.log1p(np.clip(arr, a_min=0.0, a_max=None))
    return arr


def _base_inverse(values: np.ndarray, config: TrainingConfig) -> np.ndarray:
    if config.target_transform == "log1p":
        return np.expm1(values)
    return values


def fit_target_transform(values: np.ndarray, config: TrainingConfig) -> Tuple[np.ndarray, TargetStats]:
    transformed = _base_transform(values, config)
    mean = float(transformed.mean())
    std = float(transformed.std())
    if std == 0:
        std = 1.0
    normalized = (transformed - mean) / std
    return normalized.astype(np.float32), TargetStats(mean=mean, std=std)


def transform_target(values: np.ndarray, config: TrainingConfig, stats: TargetStats) -> np.ndarray:
    transformed = _base_transform(values, config)
    normalized = (transformed - stats.mean) / stats.std
    return normalized.astype(np.float32)


def inverse_transform_target(values: np.ndarray, config: TrainingConfig, stats: TargetStats) -> np.ndarray:
    unscaled = values * stats.std + stats.mean
    return _base_inverse(unscaled, config)


def prepare_model_inputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
    config: TrainingConfig,
    ) -> ModelInputs:
    train_cats, train_nums = transform_with_artifacts(train_df, artifacts)
    test_cats, test_nums = transform_with_artifacts(test_df, artifacts)
    y_train_raw = train_df[config.target_column].to_numpy(dtype=np.float32)
    y_test_raw = test_df[config.target_column].to_numpy(dtype=np.float32)
    y_train, stats = fit_target_transform(y_train_raw, config)
    y_test = transform_target(y_test_raw, config, stats)
    return ModelInputs(
        train_categorical=train_cats,
        train_numeric=train_nums,
        y_train=y_train,
        y_train_raw=y_train_raw,
        test_categorical=test_cats,
        test_numeric=test_nums,
        y_test=y_test,
        y_test_raw=y_test_raw,
        target_stats=stats,
    )


def get_categorical_cardinalities(artifacts: PreprocessingArtifacts) -> Dict[str, int]:
    return {
        col: int(max(mapping.values()) + 1)
        for col, mapping in artifacts.category_maps.items()
    }
