from __future__ import annotations

from typing import Any

import numpy as np

from src.config import (
    CROP_TYPE_ADJUSTMENTS,
    GROWTH_STAGE_ADJUSTMENTS,
    PROVISIONAL_AREA_SCALE_MAX,
    PROVISIONAL_AREA_SCALE_MIN,
    PROVISIONAL_REFERENCE_AREA_HA,
    RAINFALL_REDUCTION_CAP,
    RAINFALL_REDUCTION_PER_MM,
    SOIL_MOISTURE_REDUCTION_CAP,
    SOIL_MOISTURE_REDUCTION_PER_POINT,
    SOIL_MOISTURE_REDUCTION_START,
)
from src.ml.predict import predict_from_dict


def _normalize_lookup(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return float(max(min_value, min(max_value, value)))


def _derive_confidence_level(
    *,
    feature_availability_summary: dict[str, bool],
    provisional_scaling_applied: bool,
) -> str:
    provided_count = sum(int(value) for value in feature_availability_summary.values())
    if provisional_scaling_applied:
        return "low"
    if provided_count >= 3:
        return "medium"
    return "low"


def recommend_water(
    input_features: dict[str, Any],
    *,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction = predict_from_dict(input_features, artifact=artifact)
    baseline_weekly_l = float(prediction["predicted_weekly_l"])
    recommended_weekly_l = baseline_weekly_l

    warnings = [
        "model trained on limited simulated dataset with 3 farms and no explicit agronomic crop-water drivers",
    ]
    assumptions: list[str] = []

    land_area_ha = input_features.get("land_area_ha")
    crop_type = input_features.get("crop_type")
    growth_stage = input_features.get("growth_stage")
    rainfall_mm = input_features.get("rainfall_mm")
    soil_moisture_percent = input_features.get("soil_moisture_percent")

    feature_summary = {
        "land_area_ha_provided": land_area_ha is not None,
        "crop_type_provided": crop_type is not None,
        "growth_stage_provided": growth_stage is not None,
        "rainfall_mm_provided": rainfall_mm is not None,
        "soil_moisture_percent_provided": soil_moisture_percent is not None,
    }

    provisional_scaling_applied = False
    if land_area_ha is not None:
        scale_ratio = _clamp(
            float(land_area_ha) / PROVISIONAL_REFERENCE_AREA_HA,
            PROVISIONAL_AREA_SCALE_MIN,
            PROVISIONAL_AREA_SCALE_MAX,
        )
        recommended_weekly_l *= scale_ratio
        provisional_scaling_applied = True
        assumptions.append(
            "land area scaling is provisional because the model was trained on absolute farm-level water targets without area normalization"
        )
        warnings.append(
            "land area scaling uses a nominal 1.0 ha reference and capped ratios; treat as an interim heuristic only"
        )
    else:
        assumptions.append("land_area_ha not supplied; no area-based adjustment applied")

    if crop_type is not None:
        crop_key = _normalize_lookup(str(crop_type))
        factor = CROP_TYPE_ADJUSTMENTS.get(crop_key)
        if factor is not None:
            recommended_weekly_l *= factor
            assumptions.append(
                f"crop_type adjustment applied using configurable placeholder factor '{crop_key}'"
            )
        else:
            warnings.append(
                f"crop_type '{crop_type}' is not configured; no crop-specific adjustment applied"
            )
    else:
        assumptions.append("crop_type not supplied; no crop-specific adjustment applied")

    if growth_stage is not None:
        growth_stage_key = _normalize_lookup(str(growth_stage))
        factor = GROWTH_STAGE_ADJUSTMENTS.get(growth_stage_key)
        if factor is not None:
            recommended_weekly_l *= factor
            assumptions.append(
                f"growth_stage adjustment applied using configurable placeholder factor '{growth_stage_key}'"
            )
        else:
            warnings.append(
                f"growth_stage '{growth_stage}' is not configured; no growth-stage adjustment applied"
            )
    else:
        assumptions.append("growth_stage not supplied; no growth-stage adjustment applied")

    if rainfall_mm is not None:
        rainfall_reduction = min(RAINFALL_REDUCTION_CAP, float(rainfall_mm) * RAINFALL_REDUCTION_PER_MM)
        recommended_weekly_l *= 1.0 - rainfall_reduction
        assumptions.append(
            "rainfall adjustment applied using a conservative configurable reduction heuristic"
        )
    else:
        assumptions.append("rainfall_mm not supplied; no rainfall-based adjustment applied")

    if soil_moisture_percent is not None:
        surplus = max(0.0, float(soil_moisture_percent) - SOIL_MOISTURE_REDUCTION_START)
        soil_reduction = min(
            SOIL_MOISTURE_REDUCTION_CAP,
            surplus * SOIL_MOISTURE_REDUCTION_PER_POINT,
        )
        recommended_weekly_l *= 1.0 - soil_reduction
        assumptions.append(
            "soil moisture adjustment applied using a conservative configurable reduction heuristic"
        )
    else:
        assumptions.append(
            "soil_moisture_percent not supplied; no soil-moisture-based adjustment applied"
        )

    if not feature_summary["rainfall_mm_provided"] and not feature_summary["soil_moisture_percent_provided"]:
        warnings.append(
            "confidence is lower because rainfall and soil moisture are missing from the current feature set"
        )

    confidence_level = _derive_confidence_level(
        feature_availability_summary=feature_summary,
        provisional_scaling_applied=provisional_scaling_applied,
    )

    return {
        "baseline_weekly_l": round(baseline_weekly_l, 2),
        "recommended_weekly_l": round(max(recommended_weekly_l, 0.0), 2),
        "recommended_daily_l": round(max(recommended_weekly_l, 0.0) / 7.0, 2),
        "confidence_level": confidence_level,
        "assumptions": assumptions,
        "warnings": warnings,
        "feature_availability_summary": feature_summary,
        "model_name": prediction["model_name"],
    }

