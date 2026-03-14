from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data.validate_schema import normalize_quarter


class WeeklyObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country: str = Field(default="Australia")
    state: str = Field(default="Victoria")
    region: str
    farm_id: str
    year: int = Field(ge=2021)
    quarter: str
    week: int = Field(ge=1, le=13)
    nitrogen_weekly: float
    phosphorus_weekly: float
    potassium_weekly: float
    calcium_weekly: float
    magnesium_weekly: float
    temperature_avg_c: float
    sunlight_hours: float
    humidity_percent: float = Field(ge=0, le=100)

    @field_validator("quarter")
    @classmethod
    def validate_quarter(cls, value: str) -> str:
        return normalize_quarter(value)

    def to_model_input(self) -> dict[str, Any]:
        return {
            "Country": self.country,
            "State": self.state,
            "Region": self.region,
            "Farm_ID": self.farm_id,
            "Year": self.year,
            "Quarter": self.quarter,
            "Week": self.week,
            "Nitrogen_Weekly": self.nitrogen_weekly,
            "Phosphorus_Weekly": self.phosphorus_weekly,
            "Potassium_Weekly": self.potassium_weekly,
            "Calcium_Weekly": self.calcium_weekly,
            "Magnesium_Weekly": self.magnesium_weekly,
            "Temperature_Avg_C": self.temperature_avg_c,
            "Sunlight_Hours": self.sunlight_hours,
            "Humidity_Percent": self.humidity_percent,
        }


class PredictionRequest(WeeklyObservationRequest):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "region": "Gippsland",
                "farm_id": "FARM_GIPPSLAND_001",
                "year": 2026,
                "quarter": "Q1",
                "week": 1,
                "nitrogen_weekly": 48.0,
                "phosphorus_weekly": 18.0,
                "potassium_weekly": 24.0,
                "calcium_weekly": 11.5,
                "magnesium_weekly": 7.8,
                "temperature_avg_c": 24.0,
                "sunlight_hours": 67.0,
                "humidity_percent": 56.0,
            }
        },
    )


class RecommendationRequest(WeeklyObservationRequest):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "region": "Gippsland",
                "farm_id": "FARM_GIPPSLAND_001",
                "year": 2026,
                "quarter": "Q1",
                "week": 1,
                "nitrogen_weekly": 48.0,
                "phosphorus_weekly": 18.0,
                "potassium_weekly": 24.0,
                "calcium_weekly": 11.5,
                "magnesium_weekly": 7.8,
                "temperature_avg_c": 24.0,
                "sunlight_hours": 67.0,
                "humidity_percent": 56.0,
                "land_area_ha": 2.5,
                "rainfall_mm": 8.0,
                "soil_moisture_percent": 43.0,
            }
        },
    )

    land_area_ha: Optional[float] = Field(default=None, gt=0)
    crop_type: Optional[str] = None
    growth_stage: Optional[str] = None
    rainfall_mm: Optional[float] = Field(default=None, ge=0)
    soil_moisture_percent: Optional[float] = Field(default=None, ge=0, le=100)

    def to_model_input(self) -> dict[str, Any]:
        payload = super().to_model_input()
        payload.update(
            {
                "land_area_ha": self.land_area_ha,
                "crop_type": self.crop_type,
                "growth_stage": self.growth_stage,
                "rainfall_mm": self.rainfall_mm,
                "soil_moisture_percent": self.soil_moisture_percent,
            }
        )
        return payload


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_weekly_l: float
    predicted_daily_l: float
    model_name: str


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_weekly_l: float
    recommended_weekly_l: float
    recommended_daily_l: float
    confidence_level: str
    assumptions: list[str]
    warnings: list[str]
    feature_availability_summary: dict[str, bool]
    model_name: str


class RetrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: Optional[Path] = None


class RetrainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_model: str
    trained_at_utc: str
    dataset_row_count: int
    selected_model_metrics: dict[str, Any]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    model_loaded: bool
    model_path: str


class EnvironmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(ge=0, le=50)
    humidity: float = Field(ge=0, le=100)
    soil_moisture: float = Field(ge=0, le=100)
    rainfall: float = Field(ge=0, le=100)
    wind_speed: float = Field(ge=0, le=150)
    growth_stage: str = "veraison"
    variety: str = "shiraz"
    region: str = "yarra_valley"


class EnvironmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    recommendation: dict[str, Any]
    should_alert: bool
    alerts: list[str]
    environment: dict[str, Any]


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: Optional[str] = None


class ChatOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str
    is_alert: bool
    environment: dict[str, Any]

