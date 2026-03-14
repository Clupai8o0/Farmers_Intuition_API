# Farmers Intuition MVP

Production-minded MVP for a baseline irrigation recommendation backend built from a simulated historical farm dataset for Victoria, Australia.

## Purpose

This project trains a supervised regression model on the current dataset schema to estimate weekly irrigation demand, then wraps that baseline prediction in a recommendation layer that can accept future agronomic inputs such as land area, crop type, rainfall, soil moisture, and growth stage.

The design is intentionally split into two layers:

1. `Baseline ML model`
   Predicts `Water_Weekly_L` using only fields that actually exist in the current dataset.
2. `Recommendation wrapper`
   Applies transparent, configurable adjustments when optional agronomic inputs are supplied.

This is a baseline irrigation recommender, not a scientifically complete crop-water balance model.

## Dataset

Expected training schema:

- `Country`
- `State`
- `Region`
- `Farm_ID`
- `Year`
- `Quarter`
- `Week`
- `Water_Weekly_L`
- `Water_Daily_Avg_L`
- `Nitrogen_Weekly`
- `Phosphorus_Weekly`
- `Potassium_Weekly`
- `Calcium_Weekly`
- `Magnesium_Weekly`
- `Temperature_Avg_C`
- `Sunlight_Hours`
- `Humidity_Percent`

Current business limitations in the historical dataset:

- No `Crop_Type`
- No `Land_Area_Ha`
- No `Rainfall_mm`
- No `Soil_Moisture_Percent`
- No `Growth_Stage`
- No `ET0` / evapotranspiration
- No `Soil_Type`

The repository currently contains `victoria_farmland_history.xlsx`. The loader will bootstrap `data/raw/victoria_farmland_history.csv` from the workbook if a CSV is not already present.

### Bundled Workbook Normalization

The included workbook does not exactly match the target CSV header names. The loader supports the workbook's `Weekly_Data` sheet and normalizes it into the canonical training schema:

- `City_or_Region` -> `Region`
- `Farmland` -> surrogate `Farm_ID`
- `Week_In_Quarter` -> `Week`
- `Weekly_Water_Consumption_Liters` -> `Water_Weekly_L`
- `Avg_Daily_Water_Consumption_Liters` -> `Water_Daily_Avg_L`
- `Weekly_Nitrogen_kg_ha` -> `Nitrogen_Weekly`
- `Weekly_Phosphorus_kg_ha` -> `Phosphorus_Weekly`
- `Weekly_Potassium_kg_ha` -> `Potassium_Weekly`
- `Weekly_Calcium_kg_ha` -> `Calcium_Weekly`
- `Weekly_Magnesium_kg_ha` -> `Magnesium_Weekly`
- `Avg_Daily_Temperature_C` -> `Temperature_Avg_C`
- `Avg_Daily_Sunlight_Hours` -> `Sunlight_Hours`
- `Avg_Daily_Humidity_Pct` -> `Humidity_Percent`

Because the workbook has no explicit `Farm_ID`, the loader derives a stable surrogate ID from the farm name. That normalization is explicit in code and should be replaced with a true farm identifier if one becomes available.

## Project Structure

```text
project_root/
  data/
    raw/
    processed/
  models/
  notebooks/
    eda_report.py
  src/
    __init__.py
    config.py
    data/
      __init__.py
      load_data.py
      validate_schema.py
      preprocess.py
      feature_engineering.py
    ml/
      __init__.py
      train.py
      evaluate.py
      predict.py
      recommend.py
    api/
      __init__.py
      main.py
      schemas.py
    utils/
      __init__.py
      logging_utils.py
  tests/
    conftest.py
    test_data_pipeline.py
    test_prediction_api.py
    test_recommendation_logic.py
  requirements.txt
  README.md
  .gitignore
```

## Assumptions

- The current supervised target is `Water_Weekly_L`.
- `Water_Daily_Avg_L` is not used as a model feature because it is target-derived and would leak the label.
- Time-aware lag and rolling features are computed per `Farm_ID` using only prior rows.
- Training uses a time-aware split: train on `2021-2024`, test on `2025`.
- Additional walk-forward yearly evaluation is used to avoid selecting a model from one holdout metric alone.

## Missing-Value Strategy

- Schema validation allows missing numeric values as long as the column is numeric.
- The model preprocessing pipeline imputes numeric features with the median and categorical features with the most frequent value.
- Missing optional recommendation inputs do not block inference; they reduce recommendation confidence.

## Land Area Limitation

The training target is absolute farm-level liters, not liters per hectare. If `land_area_ha` is provided, the recommendation layer applies a provisional scaling rule using a nominal `1.0 ha` reference and capped scaling ratios. This is an interim heuristic only. It should not be treated as agronomic truth.

## Crop and Environmental Adjustment Limitation

Crop type, growth stage, rainfall, and soil moisture adjustments are implemented as configurable placeholder heuristics in `src/config.py`. They exist to make the system extensible and transparent. They are not scientifically authoritative.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.ml.train
```

Artifacts written by training:

- `models/irrigation_recommender.joblib`
- `models/model_comparison.csv`
- `models/evaluation_summary.json`
- `data/processed/feature_dataset.csv`

## Run EDA

```bash
python notebooks/eda_report.py
```

Outputs are written to `data/processed/eda/`.

## Run API

```bash
uvicorn src.api.main:app --reload
```

## API Endpoints

- `GET /health`
- `POST /predict`
- `POST /recommend`
- `POST /retrain`

### Example `curl` for `/predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "humidity_percent": 56.4
  }'
```

### Example `curl` for `/recommend`

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
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
    "land_area_ha": 2.5,
    "crop_type": "generic_vegetable",
    "growth_stage": "flowering",
    "rainfall_mm": 8.0,
    "soil_moisture_percent": 43.0
  }'
```

Representative response shape:

```json
{
  "baseline_weekly_l": 12000.5,
  "recommended_weekly_l": 10850.2,
  "recommended_daily_l": 1550.03,
  "confidence_level": "medium",
  "assumptions": [
    "crop_type not supplied; no crop-specific adjustment applied",
    "land area scaling is provisional because the model was trained on absolute farm-level water targets without area normalization"
  ],
  "warnings": [
    "model trained on limited simulated dataset with 3 farms and no explicit agronomic crop-water drivers"
  ],
  "feature_availability_summary": {
    "land_area_ha_provided": true,
    "crop_type_provided": false,
    "growth_stage_provided": false,
    "rainfall_mm_provided": true,
    "soil_moisture_percent_provided": false
  },
  "model_name": "random_forest"
}
```

## Testing

```bash
pytest
```

## What The Baseline Model Does

- Validates the incoming training dataset schema.
- Engineers time-aware lag and rolling features per farm.
- Trains and compares `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`, and `HistGradientBoostingRegressor`.
- Includes `XGBoostRegressor` only if `xgboost` is already available.
- Selects a model using combined holdout and walk-forward performance.
- Persists the fitted pipeline and historical reference data for future inference.

## Why This Is Not A Full Agronomic Model

- It does not estimate evapotranspiration.
- It does not model soil texture, infiltration, drainage, or root-zone storage.
- It does not know crop coefficients or actual crop development stages from training data.
- It predicts historical water demand patterns, not biophysical water balance.

## Recommended Future Columns

- `Land_Area_Ha`
- `Crop_Type`
- `Soil_Moisture_Percent`
- `Rainfall_mm`
- `Growth_Stage`
- `ET0`
- `Soil_Type`

## Next Modeling Improvements

- Retrain on area-normalized targets such as liters per hectare or millimeters applied.
- Add measured rainfall, soil moisture, and ET0 to replace wrapper heuristics with learned effects.
- Introduce sequential validation over longer time horizons and farm cold-start evaluation.
- Store and replay new weekly observations so inference lags use post-training history without immediate retraining.
- Add model monitoring for drift, forecast error, and feature completeness.
