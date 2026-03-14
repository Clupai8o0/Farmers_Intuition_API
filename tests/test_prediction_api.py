from __future__ import annotations

from fastapi.testclient import TestClient

import src.api.main as api_main
from src.api.main import app


def test_predict_endpoint_success(
    monkeypatch,
    trained_artifact: dict,
) -> None:
    client = TestClient(app)
    monkeypatch.setattr(api_main, "load_model_artifact", lambda: trained_artifact)

    response = client.post(
        "/predict",
        json={
            "region": "Gippsland",
            "farm_id": "FARM_GIPPSLAND_001",
            "year": 2026,
            "quarter": "Q1",
            "week": 1,
            "nitrogen_weekly": 47.5,
            "phosphorus_weekly": 18.2,
            "potassium_weekly": 24.1,
            "calcium_weekly": 11.0,
            "magnesium_weekly": 7.1,
            "temperature_avg_c": 24.8,
            "sunlight_hours": 59.6,
            "humidity_percent": 56.4,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_weekly_l"] > 0
    assert body["model_name"]


def test_predict_endpoint_validation_failure() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "region": "Gippsland",
            "farm_id": "FARM_GIPPSLAND_001",
            "year": 2026,
            "quarter": "Q5",
            "week": 1,
            "nitrogen_weekly": 47.5,
            "phosphorus_weekly": 18.2,
            "potassium_weekly": 24.1,
            "calcium_weekly": 11.0,
            "magnesium_weekly": 7.1,
            "temperature_avg_c": 24.8,
            "sunlight_hours": 59.6,
            "humidity_percent": 56.4,
        },
    )
    assert response.status_code == 422

