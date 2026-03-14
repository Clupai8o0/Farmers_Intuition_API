from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
    RetrainRequest,
    RetrainResponse,
)
from src.config import MODEL_ARTIFACT_PATH
from src.data.validate_schema import SchemaValidationError
from src.ml.predict import ModelArtifactNotFoundError, load_model_artifact, predict_from_dict
from src.ml.recommend import recommend_water
from src.ml.train import train_and_select_model

app = FastAPI(
    title="Farmers Intuition Irrigation API",
    version="0.1.0",
    description=(
        "Production-minded MVP for baseline irrigation demand prediction and "
        "recommendation using simulated Victorian farm history."
    ),
)


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
        artifact = load_model_artifact()
        result = predict_from_dict(request.to_model_input(), artifact=artifact)
        return PredictionResponse(**result)
    except ModelArtifactNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (SchemaValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    try:
        artifact = load_model_artifact()
        result = recommend_water(request.to_model_input(), artifact=artifact)
        return RecommendationResponse(**result)
    except ModelArtifactNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (SchemaValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc


@app.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest) -> RetrainResponse:
    try:
        artifact = train_and_select_model(request.dataset_path)
        return RetrainResponse(
            selected_model=artifact["model_name"],
            trained_at_utc=artifact["trained_at_utc"],
            dataset_row_count=artifact["dataset_row_count"],
            selected_model_metrics=artifact["selected_model_metrics"],
        )
    except (FileNotFoundError, SchemaValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}") from exc

