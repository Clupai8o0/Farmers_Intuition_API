from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import (
    BASE_CATEGORICAL_FEATURE_COLUMNS,
    BASE_NUMERIC_FEATURE_COLUMNS,
    FERTILIZER_COLUMNS,
    LAG_SOURCE_COLUMNS,
    LAG_STEPS,
    ROLLING_WINDOWS,
    SEASON_BY_QUARTER,
    TARGET_COLUMN,
)
from src.data.preprocess import sort_by_farm_time


@dataclass(frozen=True)
class FeatureSpec:
    categorical_features: list[str]
    numeric_features: list[str]
    target_column: str

    @property
    def all_features(self) -> list[str]:
        return self.numeric_features + self.categorical_features


def _chronological_week_index(df: pd.DataFrame) -> pd.Series:
    quarter_number = df["Quarter"].str.replace("Q", "", regex=False).astype(int)
    return ((df["Year"] - df["Year"].min()) * 52) + ((quarter_number - 1) * 13) + (df["Week"] - 1)


def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["week_sin"] = np.sin(2 * np.pi * result["Week"] / 13.0)
    result["week_cos"] = np.cos(2 * np.pi * result["Week"] / 13.0)
    quarter_number = result["Quarter"].str.replace("Q", "", regex=False).astype(int)
    result["quarter_sin"] = np.sin(2 * np.pi * quarter_number / 4.0)
    result["quarter_cos"] = np.cos(2 * np.pi * quarter_number / 4.0)
    return result


def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["total_fertiliser"] = result[FERTILIZER_COLUMNS].sum(axis=1)
    result["temperature_x_humidity"] = (
        result["Temperature_Avg_C"] * result["Humidity_Percent"]
    )
    result["sunlight_x_temperature"] = (
        result["Sunlight_Hours"] * result["Temperature_Avg_C"]
    )
    return result


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    grouped = result.groupby("Farm_ID", group_keys=False)
    for column in LAG_SOURCE_COLUMNS:
        for lag in LAG_STEPS:
            result[f"{column}_lag_{lag}"] = grouped[column].shift(lag)
        for window in ROLLING_WINDOWS:
            result[f"{column}_rolling_mean_{window}"] = grouped[column].transform(
                lambda series: series.shift(1).rolling(window=window, min_periods=1).mean()
            )
    return result


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
    result = sort_by_farm_time(df)
    result["Season"] = result["Quarter"].map(SEASON_BY_QUARTER)
    result["time_index"] = _chronological_week_index(result)
    result["farm_time_index"] = result.groupby("Farm_ID").cumcount()
    result = _add_cyclical_features(result)
    result = _add_interactions(result)
    result = _add_lag_features(result)

    numeric_features = list(BASE_NUMERIC_FEATURE_COLUMNS) + [
        "time_index",
        "farm_time_index",
        "week_sin",
        "week_cos",
        "quarter_sin",
        "quarter_cos",
        "total_fertiliser",
        "temperature_x_humidity",
        "sunlight_x_temperature",
    ]

    for column in LAG_SOURCE_COLUMNS:
        for lag in LAG_STEPS:
            numeric_features.append(f"{column}_lag_{lag}")
        for window in ROLLING_WINDOWS:
            numeric_features.append(f"{column}_rolling_mean_{window}")

    categorical_features = list(BASE_CATEGORICAL_FEATURE_COLUMNS)
    spec = FeatureSpec(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        target_column=TARGET_COLUMN,
    )
    return result, spec


def prepare_features_for_prediction(
    *,
    historical_df: pd.DataFrame,
    incoming_records: Sequence[dict],
) -> tuple[pd.DataFrame, FeatureSpec]:
    incoming_df = pd.DataFrame(incoming_records)
    incoming_df["__is_inference__"] = True
    combined_df = historical_df.copy()
    combined_df["__is_inference__"] = False
    merged = pd.concat([combined_df, incoming_df], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(
        subset=["Farm_ID", "Year", "Quarter", "Week"],
        keep="last",
    )
    feature_df, spec = engineer_features(merged)
    prediction_rows = feature_df.loc[feature_df["__is_inference__"]].copy()
    if prediction_rows.empty:
        raise ValueError("No inference rows remained after feature preparation.")
    return prediction_rows, spec
