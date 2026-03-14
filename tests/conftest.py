from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.ml.train as train_module


@pytest.fixture()
def sample_dataset_df() -> pd.DataFrame:
    regions = [
        ("Gippsland", "FARM_GIPPSLAND_001", 12000.0),
        ("Mallee", "FARM_MALLEE_001", 9200.0),
        ("Wimmera", "FARM_WIMMERA_001", 10400.0),
    ]
    quarter_factor = {"Q1": 1.15, "Q2": 0.95, "Q3": 0.82, "Q4": 1.02}
    records: list[dict] = []
    for region, farm_id, base_water in regions:
        regional_temperature_offset = {"Gippsland": 0.8, "Mallee": 2.4, "Wimmera": 1.4}[region]
        regional_humidity_offset = {"Gippsland": 7.0, "Mallee": -6.0, "Wimmera": -1.5}[region]
        for year in range(2021, 2026):
            for quarter_number, quarter in enumerate(["Q1", "Q2", "Q3", "Q4"], start=1):
                for week in range(1, 14):
                    time_index = (year - 2021) * 52 + (quarter_number - 1) * 13 + (week - 1)
                    temperature = (
                        18.5
                        + regional_temperature_offset
                        + quarter_factor[quarter] * 4.0
                        + (week / 13.0) * 1.2
                    )
                    sunlight = 48.0 + quarter_factor[quarter] * 8.0 + (week / 13.0) * 3.0
                    humidity = 58.0 + regional_humidity_offset - quarter_factor[quarter] * 6.0 + (week / 13.0)
                    nitrogen = 40.0 + quarter_number * 1.5 + week * 0.3
                    phosphorus = 16.0 + quarter_number * 0.8 + week * 0.1
                    potassium = 22.0 + quarter_number * 0.9 + week * 0.12
                    calcium = 10.0 + quarter_number * 0.5 + week * 0.05
                    magnesium = 6.0 + quarter_number * 0.3 + week * 0.04
                    total_fertiliser = nitrogen + phosphorus + potassium + calcium + magnesium
                    water = (
                        base_water * quarter_factor[quarter]
                        + temperature * 110.0
                        + sunlight * 26.0
                        - humidity * 32.0
                        + total_fertiliser * 12.0
                        + time_index * 9.0
                    )
                    records.append(
                        {
                            "Country": "Australia",
                            "State": "Victoria",
                            "Region": region,
                            "Farm_ID": farm_id,
                            "Year": year,
                            "Quarter": quarter,
                            "Week": week,
                            "Water_Weekly_L": round(water, 2),
                            "Water_Daily_Avg_L": round(water / 7.0, 2),
                            "Nitrogen_Weekly": round(nitrogen, 2),
                            "Phosphorus_Weekly": round(phosphorus, 2),
                            "Potassium_Weekly": round(potassium, 2),
                            "Calcium_Weekly": round(calcium, 2),
                            "Magnesium_Weekly": round(magnesium, 2),
                            "Temperature_Avg_C": round(temperature, 2),
                            "Sunlight_Hours": round(sunlight, 2),
                            "Humidity_Percent": round(humidity, 2),
                        }
                    )
    return pd.DataFrame(records)


@pytest.fixture()
def sample_dataset_path(tmp_path: Path, sample_dataset_df: pd.DataFrame) -> Path:
    path = tmp_path / "sample_irrigation_data.csv"
    sample_dataset_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def trained_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_dataset_path: Path,
):
    models_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(train_module, "MODEL_ARTIFACT_PATH", models_dir / "irrigation_recommender.joblib")
    monkeypatch.setattr(train_module, "MODEL_COMPARISON_PATH", models_dir / "model_comparison.csv")
    monkeypatch.setattr(train_module, "METRICS_SUMMARY_PATH", models_dir / "evaluation_summary.json")
    monkeypatch.setattr(train_module, "PROCESSED_FEATURE_DATA_PATH", processed_dir / "feature_dataset.csv")
    return train_module.train_and_select_model(sample_dataset_path)


@pytest.fixture()
def prediction_payload() -> dict:
    return {
        "Region": "Gippsland",
        "Farm_ID": "FARM_GIPPSLAND_001",
        "Year": 2026,
        "Quarter": "Q1",
        "Week": 1,
        "Nitrogen_Weekly": 47.5,
        "Phosphorus_Weekly": 18.2,
        "Potassium_Weekly": 24.1,
        "Calcium_Weekly": 11.0,
        "Magnesium_Weekly": 7.1,
        "Temperature_Avg_C": 24.8,
        "Sunlight_Hours": 59.6,
        "Humidity_Percent": 56.4,
    }
