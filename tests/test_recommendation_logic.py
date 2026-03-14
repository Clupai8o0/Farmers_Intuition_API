from __future__ import annotations

from src.ml.recommend import recommend_water


def test_recommendation_without_optional_inputs(
    trained_artifact: dict,
    prediction_payload: dict,
) -> None:
    result = recommend_water(prediction_payload, artifact=trained_artifact)
    assert result["baseline_weekly_l"] > 0
    assert result["recommended_weekly_l"] > 0
    assert result["confidence_level"] == "low"
    assert result["feature_availability_summary"]["crop_type_provided"] is False


def test_recommendation_with_optional_inputs(
    trained_artifact: dict,
    prediction_payload: dict,
) -> None:
    enriched_payload = {
        **prediction_payload,
        "land_area_ha": 2.0,
        "crop_type": "generic_vegetable",
        "growth_stage": "flowering",
        "rainfall_mm": 10.0,
        "soil_moisture_percent": 44.0,
    }
    result = recommend_water(enriched_payload, artifact=trained_artifact)
    assert result["baseline_weekly_l"] > 0
    assert result["recommended_weekly_l"] > 0
    assert result["feature_availability_summary"]["land_area_ha_provided"] is True
    assert any("provisional" in item for item in result["assumptions"])

