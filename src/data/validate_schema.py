from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import (
    NUMERIC_COLUMNS,
    PRIMARY_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    TRAIN_END_YEAR,
    TRAIN_START_YEAR,
)


class SchemaValidationError(ValueError):
    """Raised when the input dataset does not satisfy the expected schema."""


def normalize_quarter(value: Any) -> str:
    text = str(value).strip().upper()
    mapping = {
        "1": "Q1",
        "Q1": "Q1",
        "QUARTER1": "Q1",
        "QUARTER 1": "Q1",
        "2": "Q2",
        "Q2": "Q2",
        "QUARTER2": "Q2",
        "QUARTER 2": "Q2",
        "3": "Q3",
        "Q3": "Q3",
        "QUARTER3": "Q3",
        "QUARTER 3": "Q3",
        "4": "Q4",
        "Q4": "Q4",
        "QUARTER4": "Q4",
        "QUARTER 4": "Q4",
    }
    if text not in mapping:
        raise SchemaValidationError(
            f"Invalid Quarter value '{value}'. Expected one of Q1, Q2, Q3, Q4."
        )
    return mapping[text]


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: str(column).strip() for column in df.columns}
    return df.rename(columns=renamed)


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise SchemaValidationError(f"Dataset is missing required columns: {missing}")


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    bad_columns: list[str] = []
    for column in NUMERIC_COLUMNS:
        series = pd.to_numeric(result[column], errors="coerce")
        invalid_mask = result[column].notna() & series.isna()
        if invalid_mask.any():
            bad_columns.append(column)
        result[column] = series
    if bad_columns:
        raise SchemaValidationError(
            "The following columns contain non-numeric values that could not be coerced: "
            + ", ".join(sorted(set(bad_columns)))
        )
    return result


def _validate_ranges(df: pd.DataFrame, enforce_training_years: bool) -> None:
    if df["Week"].dropna().between(1, 13).all() is False:
        raise SchemaValidationError("Week values must be between 1 and 13.")
    if enforce_training_years:
        if df["Year"].dropna().between(TRAIN_START_YEAR, TRAIN_END_YEAR).all() is False:
            raise SchemaValidationError(
                f"Year values for training data must be between {TRAIN_START_YEAR} and {TRAIN_END_YEAR}."
            )
    elif (df["Year"].dropna() < TRAIN_START_YEAR).any():
        raise SchemaValidationError(
            f"Year values must be greater than or equal to {TRAIN_START_YEAR}."
        )


def _validate_duplicates(df: pd.DataFrame) -> None:
    duplicate_mask = df.duplicated(subset=PRIMARY_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, PRIMARY_KEY_COLUMNS].to_dict(orient="records")
        raise SchemaValidationError(
            "Duplicate farm chronology rows detected for Farm_ID + Year + Quarter + Week: "
            f"{duplicates[:5]}"
        )


def validate_dataset_schema(
    df: pd.DataFrame,
    *,
    enforce_training_years: bool = True,
) -> pd.DataFrame:
    """Validate schema, normalize quarter values, and return a cleaned copy."""
    result = _strip_columns(df)
    _validate_required_columns(result)
    result = _coerce_numeric_columns(result)
    result["Quarter"] = result["Quarter"].map(normalize_quarter)
    _validate_ranges(result, enforce_training_years=enforce_training_years)
    _validate_duplicates(result)
    return result

