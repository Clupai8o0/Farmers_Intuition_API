from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.config import (
    METRICS_SUMMARY_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_COMPARISON_PATH,
    PROCESSED_FEATURE_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data.feature_engineering import FeatureSpec, engineer_features
from src.data.load_data import load_dataset
from src.data.preprocess import build_preprocessor, split_train_test_by_year
from src.ml.evaluate import compute_regression_metrics, summarise_walk_forward_metrics
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def get_candidate_models() -> dict[str, Callable[[], RegressorMixin]]:
    candidates: dict[str, Callable[[], RegressorMixin]] = {
        "linear_regression": lambda: LinearRegression(),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBRegressor  # type: ignore

        candidates["xgboost"] = lambda: XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
        )
        LOGGER.info("XGBoost is available and will be included in model comparison.")
    except Exception:
        LOGGER.info("XGBoost not available; using scikit-learn models only.")
    return candidates


def build_model_pipeline(model_name: str, estimator: RegressorMixin, spec: FeatureSpec) -> Pipeline:
    preprocessor = build_preprocessor(
        categorical_columns=spec.categorical_features,
        numeric_columns=spec.numeric_features,
        scale_numeric=model_name == "linear_regression",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def _walk_forward_years(frame: pd.DataFrame) -> list[int]:
    years = sorted(frame["Year"].unique().tolist())
    return [year for year in years if year > min(years)]


def _evaluate_candidate(
    *,
    model_name: str,
    model_factory: Callable[[], RegressorMixin],
    feature_frame: pd.DataFrame,
    spec: FeatureSpec,
) -> dict:
    train_df, test_df = split_train_test_by_year(feature_frame)
    X_train = train_df[spec.all_features]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[spec.all_features]
    y_test = test_df[TARGET_COLUMN]

    holdout_pipeline = build_model_pipeline(model_name, model_factory(), spec)
    holdout_pipeline.fit(X_train, y_train)
    holdout_predictions = holdout_pipeline.predict(X_test)
    holdout_metrics = compute_regression_metrics(y_test, holdout_predictions)

    walk_forward_records: list[dict[str, float]] = []
    for evaluation_year in _walk_forward_years(feature_frame):
        fold_train = feature_frame.loc[feature_frame["Year"] < evaluation_year]
        fold_test = feature_frame.loc[feature_frame["Year"] == evaluation_year]
        if fold_train.empty or fold_test.empty:
            continue
        fold_pipeline = build_model_pipeline(model_name, model_factory(), spec)
        fold_pipeline.fit(fold_train[spec.all_features], fold_train[TARGET_COLUMN])
        fold_predictions = fold_pipeline.predict(fold_test[spec.all_features])
        fold_metrics = compute_regression_metrics(fold_test[TARGET_COLUMN], fold_predictions)
        walk_forward_records.append(fold_metrics)

    walk_forward_mean = summarise_walk_forward_metrics(walk_forward_records)
    return {
        "model_name": model_name,
        "holdout_rmse": holdout_metrics["rmse"],
        "holdout_mae": holdout_metrics["mae"],
        "holdout_r2": holdout_metrics["r2"],
        "holdout_mape": holdout_metrics["mape"],
        "walk_forward_rmse": walk_forward_mean["rmse"],
        "walk_forward_mae": walk_forward_mean["mae"],
        "walk_forward_r2": walk_forward_mean["r2"],
        "walk_forward_mape": walk_forward_mean["mape"],
    }


def _select_best_model(comparison_df: pd.DataFrame) -> pd.DataFrame:
    ranked = comparison_df.copy()
    ranked["selection_score"] = (
        ranked["holdout_rmse"].rank(method="dense", ascending=True)
        + ranked["holdout_mae"].rank(method="dense", ascending=True)
        + ranked["walk_forward_rmse"].rank(method="dense", ascending=True)
        + ranked["walk_forward_mae"].rank(method="dense", ascending=True)
        + ranked["holdout_r2"].rank(method="dense", ascending=False)
        + ranked["walk_forward_r2"].rank(method="dense", ascending=False)
    )
    return ranked.sort_values(
        by=["selection_score", "holdout_rmse", "walk_forward_rmse", "holdout_mae"],
        ascending=[True, True, True, True],
    )


def train_and_select_model(dataset_path: Path | None = None) -> dict:
    raw_df = load_dataset(dataset_path, enforce_training_years=True)
    feature_frame, spec = engineer_features(raw_df)
    PROCESSED_FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(PROCESSED_FEATURE_DATA_PATH, index=False)

    results = []
    for model_name, model_factory in get_candidate_models().items():
        LOGGER.info("Training candidate model: %s", model_name)
        results.append(
            _evaluate_candidate(
                model_name=model_name,
                model_factory=model_factory,
                feature_frame=feature_frame,
                spec=spec,
            )
        )

    comparison_df = _select_best_model(pd.DataFrame(results))
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    best_model_name = str(comparison_df.iloc[0]["model_name"])

    final_model_factory = get_candidate_models()[best_model_name]
    final_pipeline = build_model_pipeline(best_model_name, final_model_factory(), spec)
    final_pipeline.fit(feature_frame[spec.all_features], feature_frame[TARGET_COLUMN])

    selected_summary = comparison_df.iloc[0].to_dict()
    artifact = {
        "model_name": best_model_name,
        "pipeline": final_pipeline,
        "feature_spec": deepcopy(spec),
        "historical_data": raw_df.copy(),
        "comparison_table": comparison_df.to_dict(orient="records"),
        "selected_model_metrics": selected_summary,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_row_count": int(len(raw_df)),
        "limitations": [
            "Training data is a simulated dataset covering 3 farms and 780 rows.",
            "The supervised model predicts Water_Weekly_L using historical farm-level absolute water volumes.",
            "Crop type, land area, rainfall, soil moisture, evapotranspiration, and growth stage are not present in the training target schema.",
        ],
    }
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)

    metrics_summary = {
        "selected_model": best_model_name,
        "selected_model_metrics": selected_summary,
        "trained_at_utc": artifact["trained_at_utc"],
        "dataset_row_count": artifact["dataset_row_count"],
    }
    with METRICS_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics_summary, handle, indent=2)
    LOGGER.info("Saved trained model artifact to %s", MODEL_ARTIFACT_PATH)
    return artifact


if __name__ == "__main__":
    train_and_select_model()
