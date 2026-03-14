from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TEST_YEAR


def sort_by_farm_time(df: pd.DataFrame) -> pd.DataFrame:
    quarter_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    result = df.copy()
    result["_quarter_order"] = result["Quarter"].map(quarter_order)
    result = result.sort_values(
        by=["Farm_ID", "Year", "_quarter_order", "Week"],
        ascending=True,
    ).drop(columns="_quarter_order")
    return result.reset_index(drop=True)


def split_train_test_by_year(
    df: pd.DataFrame,
    *,
    test_year: int = TEST_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[df["Year"] < test_year].copy()
    test_df = df.loc[df["Year"] == test_year].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Time-aware split failed. Need rows before {test_year} and rows in {test_year}."
        )
    return train_df, test_df


def build_preprocessor(
    *,
    categorical_columns: Iterable[str],
    numeric_columns: Iterable[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ]
    )

