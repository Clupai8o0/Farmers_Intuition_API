from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_ARTIFACT_PATH, REQUIRED_COLUMNS
from src.data.feature_engineering import prepare_features_for_prediction
from src.data.validate_schema import validate_dataset_schema


class ModelArtifactNotFoundError(FileNotFoundError):
    """Raised when prediction is requested before a trained artifact exists."""


def load_model_artifact(path: Path = MODEL_ARTIFACT_PATH) -> dict[str, Any]:
    if not path.exists():
        raise ModelArtifactNotFoundError(
            f"Model artifact not found at {path}. Train the model before predicting."
        )
    artifact = joblib.load(path)
    if "pipeline" not in artifact or "feature_spec" not in artifact:
        raise ValueError("Invalid model artifact structure.")
    return artifact


def _build_inference_frame(input_features: dict[str, Any]) -> pd.DataFrame:
    record = {column: np.nan for column in REQUIRED_COLUMNS}
    record.update(
        {
            "Country": "Australia",
            "State": "Victoria",
            "Region": input_features.get("Region"),
            "Farm_ID": input_features.get("Farm_ID"),
            "Year": input_features.get("Year"),
            "Quarter": input_features.get("Quarter"),
            "Week": input_features.get("Week"),
            "Nitrogen_Weekly": input_features.get("Nitrogen_Weekly"),
            "Phosphorus_Weekly": input_features.get("Phosphorus_Weekly"),
            "Potassium_Weekly": input_features.get("Potassium_Weekly"),
            "Calcium_Weekly": input_features.get("Calcium_Weekly"),
            "Magnesium_Weekly": input_features.get("Magnesium_Weekly"),
            "Temperature_Avg_C": input_features.get("Temperature_Avg_C"),
            "Sunlight_Hours": input_features.get("Sunlight_Hours"),
            "Humidity_Percent": input_features.get("Humidity_Percent"),
        }
    )
    frame = pd.DataFrame([record])
    return validate_dataset_schema(frame, enforce_training_years=False)


def predict_from_dict(
    input_features: dict[str, Any],
    *,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_artifact = artifact or load_model_artifact()
    historical_df = active_artifact["historical_data"].copy()
    inference_df = _build_inference_frame(input_features)
    prediction_rows, _ = prepare_features_for_prediction(
        historical_df=historical_df,
        incoming_records=inference_df.to_dict(orient="records"),
    )
    spec = active_artifact["feature_spec"]
    baseline_prediction = float(
        active_artifact["pipeline"].predict(prediction_rows[spec.all_features])[0]
    )
    return {
        "predicted_weekly_l": baseline_prediction,
        "predicted_daily_l": baseline_prediction / 7.0,
        "model_name": active_artifact["model_name"],
    }

