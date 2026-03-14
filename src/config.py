from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DEFAULT_DATASET_CSV = RAW_DATA_DIR / "victoria_farmland_history.csv"
DEFAULT_DATASET_XLSX = RAW_DATA_DIR / "victoria_farmland_history.xlsx"
ROOT_LEVEL_DATASET_XLSX = PROJECT_ROOT / "victoria_farmland_history.xlsx"

MODEL_ARTIFACT_PATH = MODELS_DIR / "irrigation_recommender.joblib"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
METRICS_SUMMARY_PATH = MODELS_DIR / "evaluation_summary.json"
PROCESSED_FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "feature_dataset.csv"
EDA_OUTPUT_DIR = PROCESSED_DATA_DIR / "eda"

PRIMARY_KEY_COLUMNS = ["Farm_ID", "Year", "Quarter", "Week"]
REQUIRED_COLUMNS = [
    "Country",
    "State",
    "Region",
    "Farm_ID",
    "Year",
    "Quarter",
    "Week",
    "Water_Weekly_L",
    "Water_Daily_Avg_L",
    "Nitrogen_Weekly",
    "Phosphorus_Weekly",
    "Potassium_Weekly",
    "Calcium_Weekly",
    "Magnesium_Weekly",
    "Temperature_Avg_C",
    "Sunlight_Hours",
    "Humidity_Percent",
]
ALTERNATE_WEEKLY_SOURCE_COLUMNS = [
    "Country",
    "State",
    "City_or_Region",
    "Farmland",
    "Year",
    "Quarter",
    "Week_In_Quarter",
    "Weekly_Water_Consumption_Liters",
    "Avg_Daily_Water_Consumption_Liters",
    "Weekly_Nitrogen_kg_ha",
    "Weekly_Phosphorus_kg_ha",
    "Weekly_Potassium_kg_ha",
    "Weekly_Calcium_kg_ha",
    "Weekly_Magnesium_kg_ha",
    "Avg_Daily_Temperature_C",
    "Avg_Daily_Sunlight_Hours",
    "Avg_Daily_Humidity_Pct",
]
NUMERIC_COLUMNS = [
    "Year",
    "Week",
    "Water_Weekly_L",
    "Water_Daily_Avg_L",
    "Nitrogen_Weekly",
    "Phosphorus_Weekly",
    "Potassium_Weekly",
    "Calcium_Weekly",
    "Magnesium_Weekly",
    "Temperature_Avg_C",
    "Sunlight_Hours",
    "Humidity_Percent",
]
FERTILIZER_COLUMNS = [
    "Nitrogen_Weekly",
    "Phosphorus_Weekly",
    "Potassium_Weekly",
    "Calcium_Weekly",
    "Magnesium_Weekly",
]
TARGET_COLUMN = "Water_Weekly_L"
BASE_NUMERIC_FEATURE_COLUMNS = [
    "Year",
    "Week",
    "Nitrogen_Weekly",
    "Phosphorus_Weekly",
    "Potassium_Weekly",
    "Calcium_Weekly",
    "Magnesium_Weekly",
    "Temperature_Avg_C",
    "Sunlight_Hours",
    "Humidity_Percent",
]
BASE_CATEGORICAL_FEATURE_COLUMNS = ["Region", "Farm_ID", "Quarter", "Season"]
KEY_CATEGORICAL_COLUMNS = ["Country", "State", "Region", "Farm_ID", "Quarter"]
TRAIN_START_YEAR = 2021
TRAIN_END_YEAR = 2025
TEST_YEAR = 2025
RANDOM_STATE = 42

LAG_SOURCE_COLUMNS = [
    "Water_Weekly_L",
    "Temperature_Avg_C",
    "Sunlight_Hours",
    "Humidity_Percent",
    "Nitrogen_Weekly",
    "Phosphorus_Weekly",
    "Potassium_Weekly",
    "Calcium_Weekly",
    "Magnesium_Weekly",
]
LAG_STEPS = [1, 2, 4, 8]
ROLLING_WINDOWS = [2, 4, 8]

# Placeholder crop factors for the recommendation layer. These are configurable
# multipliers only and are not intended to represent agronomic ground truth.
CROP_TYPE_ADJUSTMENTS = {
    "generic_grain": 0.95,
    "generic_vegetable": 1.08,
    "generic_fruit": 1.12,
    # Vineyard varieties
    "shiraz": 1.0,
    "cabernet_sauvignon": 0.95,
    "pinot_noir": 1.05,
    "chardonnay": 0.92,
    "merlot": 0.98,
}
GROWTH_STAGE_ADJUSTMENTS = {
    "establishment": 0.95,
    "vegetative": 1.0,
    "flowering": 1.08,
    "maturity": 0.9,
    # Vineyard growth stages
    "budburst": 0.7,
    "veraison": 1.15,
    "harvest": 0.85,
    "dormancy": 0.3,
}

# The model is trained on absolute farm-level liters and does not know land area.
# This nominal reference area is only used for a provisional scaling fallback.
PROVISIONAL_REFERENCE_AREA_HA = 25.0
PROVISIONAL_AREA_SCALE_MIN = 0.001
PROVISIONAL_AREA_SCALE_MAX = 100.0

# Placeholder environmental adjustments. These are conservative heuristics used
# only to make the recommendation wrapper extensible until richer agronomic data
# is available.
RAINFALL_REDUCTION_PER_MM = 0.004
RAINFALL_REDUCTION_CAP = 0.35
SOIL_MOISTURE_REDUCTION_START = 35.0
SOIL_MOISTURE_REDUCTION_PER_POINT = 0.005
SOIL_MOISTURE_REDUCTION_CAP = 0.25

SEASON_BY_QUARTER = {
    "Q1": "Summer",
    "Q2": "Autumn",
    "Q3": "Winter",
    "Q4": "Spring",
}
