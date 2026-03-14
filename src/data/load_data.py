from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import pandas as pd

from src.config import (
    ALTERNATE_WEEKLY_SOURCE_COLUMNS,
    DEFAULT_DATASET_CSV,
    DEFAULT_DATASET_XLSX,
    RAW_DATA_DIR,
    REQUIRED_COLUMNS,
    ROOT_LEVEL_DATASET_XLSX,
)
from src.data.validate_schema import validate_dataset_schema
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _slugify_farm_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().upper()).strip("_")
    return f"FARM_{cleaned}"


def _normalize_alternate_weekly_schema(df: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(
        {
            "Country": df["Country"],
            "State": df["State"],
            "Region": df["City_or_Region"],
            # The workbook does not contain Farm_ID, so derive a stable surrogate
            # from the farm name and document that this is a loader-side normalization.
            "Farm_ID": df["Farmland"].map(_slugify_farm_name),
            "Year": df["Year"],
            "Quarter": df["Quarter"],
            "Week": df["Week_In_Quarter"],
            "Water_Weekly_L": df["Weekly_Water_Consumption_Liters"],
            "Water_Daily_Avg_L": df["Avg_Daily_Water_Consumption_Liters"],
            "Nitrogen_Weekly": df["Weekly_Nitrogen_kg_ha"],
            "Phosphorus_Weekly": df["Weekly_Phosphorus_kg_ha"],
            "Potassium_Weekly": df["Weekly_Potassium_kg_ha"],
            "Calcium_Weekly": df["Weekly_Calcium_kg_ha"],
            "Magnesium_Weekly": df["Weekly_Magnesium_kg_ha"],
            "Temperature_Avg_C": df["Avg_Daily_Temperature_C"],
            "Sunlight_Hours": df["Avg_Daily_Sunlight_Hours"],
            "Humidity_Percent": df["Avg_Daily_Humidity_Pct"],
        }
    )
    return normalized


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if set(REQUIRED_COLUMNS).issubset(df.columns):
        return df[REQUIRED_COLUMNS].copy()
    if set(ALTERNATE_WEEKLY_SOURCE_COLUMNS).issubset(df.columns):
        return _normalize_alternate_weekly_schema(df)
    raise ValueError(
        "Dataset does not match either the canonical training schema or the supported workbook weekly schema."
    )


def _read_tabular_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _standardize_columns(pd.read_csv(path))
    if suffix in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        for sheet_name in workbook.sheet_names:
            candidate = pd.read_excel(path, sheet_name=sheet_name)
            if set(REQUIRED_COLUMNS).issubset(set(candidate.columns)) or set(
                ALTERNATE_WEEKLY_SOURCE_COLUMNS
            ).issubset(set(candidate.columns)):
                LOGGER.info("Using workbook sheet '%s' from %s", sheet_name, path)
                return _standardize_columns(candidate)
        raise ValueError(
            f"No worksheet in {path} contains the canonical schema or supported workbook weekly schema."
        )
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def bootstrap_csv_from_workbook(
    *,
    workbook_path: Optional[Path] = None,
    output_csv_path: Path = DEFAULT_DATASET_CSV,
) -> Path:
    source_path = workbook_path or DEFAULT_DATASET_XLSX
    if not source_path.exists():
        if ROOT_LEVEL_DATASET_XLSX.exists():
            source_path = ROOT_LEVEL_DATASET_XLSX
        else:
            raise FileNotFoundError(
                "No source workbook found. Expected a CSV in data/raw/ or an XLSX workbook "
                "at data/raw/victoria_farmland_history.xlsx or project root."
            )
    LOGGER.info("Bootstrapping CSV dataset from workbook %s", source_path)
    df = _read_tabular_file(source_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def resolve_dataset_path(preferred_path: Optional[Path] = None) -> Path:
    if preferred_path and preferred_path.exists():
        return preferred_path
    if DEFAULT_DATASET_CSV.exists():
        return DEFAULT_DATASET_CSV
    if DEFAULT_DATASET_XLSX.exists():
        return DEFAULT_DATASET_XLSX
    if ROOT_LEVEL_DATASET_XLSX.exists():
        return bootstrap_csv_from_workbook(workbook_path=ROOT_LEVEL_DATASET_XLSX)
    candidates = list(RAW_DATA_DIR.glob("*.csv")) + list(RAW_DATA_DIR.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError("No dataset file found under data/raw/.")
    return candidates[0]


def load_dataset(
    path: Optional[Path] = None,
    *,
    enforce_training_years: bool = True,
) -> pd.DataFrame:
    dataset_path = resolve_dataset_path(path)
    df = _read_tabular_file(dataset_path)
    return validate_dataset_schema(df, enforce_training_years=enforce_training_years)
