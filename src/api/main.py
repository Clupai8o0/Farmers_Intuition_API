from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ChatInput,
    ChatOutput,
    EnvironmentInput,
    EnvironmentResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
    RetrainRequest,
    RetrainResponse,
)
from src.config import MODEL_ARTIFACT_PATH

app = FastAPI(
    title="Farmers Intuition Irrigation API",
    version="0.2.0",
    description=(
        "Production-minded MVP for baseline irrigation demand prediction and "
        "recommendation using simulated Victorian farm history, with dashboard "
        "integration and Gemini-powered voice assistant."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://farmers-intuition.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory environment state
# ---------------------------------------------------------------------------
_environment_state: dict[str, Any] = {}
_previous_environment_state: dict[str, Any] = {}

_DEMO_DEFAULTS: dict[str, Any] = {
    "country": "Australia",
    "state": "Victoria",
    "farm_id": "FARM_YARRA_001",
    "year": datetime.now(timezone.utc).year,
    "quarter": f"Q{(datetime.now(timezone.utc).month - 1) // 3 + 1}",
    "week": 1,
    "nitrogen_weekly": 45.0,
    "phosphorus_weekly": 18.0,
    "potassium_weekly": 22.0,
    "calcium_weekly": 10.0,
    "magnesium_weekly": 7.0,
    "sunlight_hours": 60.0,
}


_CONFIG_FIELDS = ("land_area_ha", "variety", "growth_stage", "region")


def _config_changed(current: dict[str, Any], previous: dict[str, Any]) -> bool:
    """Return True if any configuration field changed between updates."""
    return any(
        current.get(f) != previous.get(f) for f in _CONFIG_FIELDS
    )


def _check_alerts(
    current: dict[str, Any], previous: dict[str, Any]
) -> tuple[bool, list[str]]:
    alerts: list[str] = []
    if current.get("soil_moisture", 100) < 30:
        alerts.append(
            f"Soil moisture critically low at {current['soil_moisture']}%"
        )
    if current.get("temperature", 0) > 35:
        alerts.append(
            f"Heat stress risk \u2014 temperature at {current['temperature']}\u00b0C"
        )
    if (
        current.get("humidity", 0) > 80
        and current.get("temperature", 0) > 10
        and current.get("rainfall", 0) > 2
    ):
        alerts.append(
            "Downy mildew conditions detected \u2014 high humidity + warm temp + rainfall"
        )
    if current.get("soil_moisture", 0) > 85:
        alerts.append(
            f"Waterlogging risk \u2014 soil moisture at {current['soil_moisture']}%"
        )
    # Only flag irrigation shift when it's caused by actual weather/sensor
    # changes, not by config changes (land area, variety, growth stage, region).
    if previous and previous.get("predicted_daily_l") and not _config_changed(current, previous):
        prev_daily = previous["predicted_daily_l"]
        curr_daily = current.get("predicted_daily_l", prev_daily)
        pct_change = abs(curr_daily - prev_daily) / max(prev_daily, 1)
        if pct_change > 0.2:
            alerts.append(
                f"Irrigation need shifted significantly ({pct_change:.0%} change)"
            )
    return len(alerts) > 0, alerts


def _build_recommend_input(env: EnvironmentInput) -> dict[str, Any]:
    """Map dashboard environment values into the existing /recommend payload."""
    return {
        "Country": _DEMO_DEFAULTS["country"],
        "State": _DEMO_DEFAULTS["state"],
        "Region": env.region.replace("_", " ").title(),
        "Farm_ID": _DEMO_DEFAULTS["farm_id"],
        "Year": _DEMO_DEFAULTS["year"],
        "Quarter": _DEMO_DEFAULTS["quarter"],
        "Week": _DEMO_DEFAULTS["week"],
        "Nitrogen_Weekly": _DEMO_DEFAULTS["nitrogen_weekly"],
        "Phosphorus_Weekly": _DEMO_DEFAULTS["phosphorus_weekly"],
        "Potassium_Weekly": _DEMO_DEFAULTS["potassium_weekly"],
        "Calcium_Weekly": _DEMO_DEFAULTS["calcium_weekly"],
        "Magnesium_Weekly": _DEMO_DEFAULTS["magnesium_weekly"],
        "Temperature_Avg_C": env.temperature,
        "Sunlight_Hours": _DEMO_DEFAULTS["sunlight_hours"],
        "Humidity_Percent": env.humidity,
        "land_area_ha": env.land_area_ha,
        "crop_type": env.variety,
        "growth_stage": env.growth_stage,
        "rainfall_mm": env.rainfall,
        "soil_moisture_percent": env.soil_moisture,
    }


def get_current_environment() -> dict[str, Any]:
    return _environment_state.copy()


# ---------------------------------------------------------------------------
# Debug endpoint — remove after deployment is stable
# ---------------------------------------------------------------------------


@app.get("/debug")
def debug() -> dict[str, Any]:
    """Return diagnostic info for debugging Vercel deployment issues."""
    import sys

    info: dict[str, Any] = {
        "python_version": sys.version,
        "model_artifact_path": str(MODEL_ARTIFACT_PATH),
        "model_artifact_exists": MODEL_ARTIFACT_PATH.exists(),
        "cwd": str(__import__("pathlib").Path.cwd()),
        "config_project_root": str(__import__("src.config", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT),
    }
    try:
        from src.data.validate_schema import SchemaValidationError  # noqa: F401
        info["validate_schema_import"] = "ok"
    except Exception as exc:
        info["validate_schema_import"] = str(exc)
    try:
        from src.ml.predict import load_model_artifact  # noqa: F401
        info["predict_import"] = "ok"
    except Exception as exc:
        info["predict_import"] = str(exc)
    try:
        from src.ml.recommend import recommend_water  # noqa: F401
        info["recommend_import"] = "ok"
    except Exception as exc:
        info["recommend_import"] = str(exc)
    try:
        from src.ml.train import train_and_select_model  # noqa: F401
        info["train_import"] = "ok"
    except Exception as exc:
        info["train_import"] = str(exc)
    try:
        from src.api.chat import generate_response  # noqa: F401
        info["chat_import"] = "ok"
    except Exception as exc:
        info["chat_import"] = str(exc)
    return info


# ---------------------------------------------------------------------------
# Original endpoints
# ---------------------------------------------------------------------------


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Return metadata about the trained model artifact."""
    if not MODEL_ARTIFACT_PATH.exists():
        raise HTTPException(status_code=503, detail="No model artifact found. Run /retrain first.")

    from src.ml.predict import load_model_artifact

    artifact = load_model_artifact()
    return {
        "model_name": artifact.get("model_name"),
        "trained_at_utc": artifact.get("trained_at_utc"),
        "dataset_row_count": artifact.get("dataset_row_count"),
        "selected_model_metrics": artifact.get("selected_model_metrics"),
        "comparison_table": artifact.get("comparison_table"),
        "limitations": artifact.get("limitations", []),
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_loaded = MODEL_ARTIFACT_PATH.exists()
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_path=str(MODEL_ARTIFACT_PATH),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        from src.data.validate_schema import SchemaValidationError
        from src.ml.predict import ModelArtifactNotFoundError, load_model_artifact, predict_from_dict

        artifact = load_model_artifact()
        result = predict_from_dict(request.to_model_input(), artifact=artifact)
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        if "ModelArtifactNotFoundError" in type(exc).__name__:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    try:
        from src.data.validate_schema import SchemaValidationError
        from src.ml.predict import ModelArtifactNotFoundError, load_model_artifact
        from src.ml.recommend import recommend_water

        artifact = load_model_artifact()
        result = recommend_water(request.to_model_input(), artifact=artifact)
        return RecommendationResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        if "ModelArtifactNotFoundError" in type(exc).__name__:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc


@app.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest) -> RetrainResponse:
    try:
        from src.ml.train import train_and_select_model

        artifact = train_and_select_model(request.dataset_path)
        return RetrainResponse(
            selected_model=artifact["model_name"],
            trained_at_utc=artifact["trained_at_utc"],
            dataset_row_count=artifact["dataset_row_count"],
            selected_model_metrics=artifact["selected_model_metrics"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Environment state manager
# ---------------------------------------------------------------------------


@app.post("/environment", response_model=EnvironmentResponse)
def post_environment(env: EnvironmentInput) -> EnvironmentResponse:
    global _environment_state, _previous_environment_state

    try:
        from src.ml.predict import ModelArtifactNotFoundError, load_model_artifact
        from src.ml.recommend import recommend_water

        artifact = load_model_artifact()
        recommend_input = _build_recommend_input(env)
        recommendation = recommend_water(recommend_input, artifact=artifact)
    except Exception as exc:
        if "ModelArtifactNotFoundError" in type(exc).__name__:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        raise HTTPException(
            status_code=500, detail=f"Environment processing failed: {exc}"
        ) from exc

    _previous_environment_state = _environment_state.copy()

    _environment_state = {
        "temperature": env.temperature,
        "humidity": env.humidity,
        "soil_moisture": env.soil_moisture,
        "rainfall": env.rainfall,
        "wind_speed": env.wind_speed,
        "land_area_ha": env.land_area_ha,
        "growth_stage": env.growth_stage,
        "variety": env.variety,
        "region": env.region,
        "predicted_daily_l": recommendation["recommended_daily_l"],
        "predicted_weekly_l": recommendation["recommended_weekly_l"],
        "confidence_level": recommendation["confidence_level"],
        "warnings": recommendation["warnings"],
        "assumptions": recommendation["assumptions"],
        "model_name": recommendation["model_name"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    should_alert, alerts = _check_alerts(
        _environment_state, _previous_environment_state
    )
    _environment_state["should_alert"] = should_alert
    _environment_state["alerts"] = alerts

    return EnvironmentResponse(
        status="ok",
        recommendation=recommendation,
        should_alert=should_alert,
        alerts=alerts,
        environment=_environment_state,
    )


@app.get("/environment")
def get_environment() -> dict[str, Any]:
    if not _environment_state:
        return {"status": "no_data", "message": "No sensor data received yet."}
    return {"status": "ok", "environment": _environment_state}


# ---------------------------------------------------------------------------
# Gemini LLM chat endpoint
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatOutput)
async def chat(data: ChatInput) -> ChatOutput:
    env = get_current_environment()
    if not env:
        return ChatOutput(
            response="No sensor data available yet. Send environment data first.",
            session_id="",
            is_alert=False,
            environment={},
        )

    try:
        from src.api.chat import generate_response

        text, session_id = await generate_response(
            data.message, env, session_id=data.session_id
        )
    except Exception as exc:
        text = f"Voice assistant unavailable: {exc}"
        session_id = data.session_id or ""

    return ChatOutput(
        response=text,
        session_id=session_id,
        is_alert=env.get("should_alert", False),
        environment=env,
    )
