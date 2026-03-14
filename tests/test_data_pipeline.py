from __future__ import annotations

import pandas as pd
import pytest

from src.data.feature_engineering import engineer_features
from src.data.preprocess import build_preprocessor
from src.data.validate_schema import SchemaValidationError, validate_dataset_schema
from src.ml.predict import predict_from_dict


def test_schema_validation_accepts_valid_dataset(sample_dataset_df: pd.DataFrame) -> None:
    validated = validate_dataset_schema(sample_dataset_df)
    assert validated.shape == sample_dataset_df.shape
    assert set(validated["Quarter"].unique()) == {"Q1", "Q2", "Q3", "Q4"}


def test_schema_validation_rejects_duplicate_rows(sample_dataset_df: pd.DataFrame) -> None:
    duplicated = pd.concat([sample_dataset_df, sample_dataset_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaValidationError):
        validate_dataset_schema(duplicated)


def test_feature_engineering_adds_expected_columns(sample_dataset_df: pd.DataFrame) -> None:
    feature_df, spec = engineer_features(sample_dataset_df)
    assert feature_df.shape[0] == sample_dataset_df.shape[0]
    assert "Water_Weekly_L_lag_1" in feature_df.columns
    assert "Temperature_Avg_C_rolling_mean_4" in feature_df.columns
    assert "total_fertiliser" in spec.numeric_features
    assert "Season" in spec.categorical_features


def test_preprocessor_fits_engineered_feature_frame(sample_dataset_df: pd.DataFrame) -> None:
    feature_df, spec = engineer_features(sample_dataset_df)
    preprocessor = build_preprocessor(
        categorical_columns=spec.categorical_features,
        numeric_columns=spec.numeric_features,
        scale_numeric=True,
    )
    transformed = preprocessor.fit_transform(feature_df[spec.all_features])
    assert transformed.shape[0] == feature_df.shape[0]
    assert transformed.shape[1] > len(spec.numeric_features)


def test_model_prediction_is_positive(trained_artifact: dict, prediction_payload: dict) -> None:
    result = predict_from_dict(prediction_payload, artifact=trained_artifact)
    assert result["predicted_weekly_l"] > 0
    assert result["predicted_daily_l"] > 0

