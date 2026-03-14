from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib_config_dir = PROJECT_ROOT / ".matplotlib"
matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import EDA_OUTPUT_DIR
from src.data.load_data import load_dataset
from src.data.preprocess import sort_by_farm_time


def run_eda(dataset_path: Path | None = None) -> None:
    df = load_dataset(dataset_path)
    df = sort_by_farm_time(df)
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset shape:", df.shape)
    print("\nSchema:")
    print(df.dtypes.astype(str))

    print("\nRegion/Farm summary:")
    print(
        df.groupby(["Region", "Farm_ID"])
        .agg(
            rows=("Farm_ID", "size"),
            water_weekly_mean=("Water_Weekly_L", "mean"),
            temperature_mean=("Temperature_Avg_C", "mean"),
            humidity_mean=("Humidity_Percent", "mean"),
        )
        .round(2)
    )

    df["time_label"] = (
        df["Year"].astype(str)
        + "-"
        + df["Quarter"].astype(str)
        + "-W"
        + df["Week"].astype(int).astype(str).str.zfill(2)
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for farm_id, farm_df in df.groupby("Farm_ID"):
        ax.plot(farm_df["time_label"], farm_df["Water_Weekly_L"], label=farm_id)
    ax.set_title("Weekly Water Usage Over Time")
    ax.set_ylabel("Water_Weekly_L")
    ax.set_xticks(ax.get_xticks()[::20])
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "water_usage_trends.png")
    plt.close(fig)

    climate_columns = ["Temperature_Avg_C", "Sunlight_Hours", "Humidity_Percent"]
    fig, axes = plt.subplots(len(climate_columns), 1, figsize=(14, 12), sharex=True)
    for axis, column in zip(axes, climate_columns, strict=True):
        for farm_id, farm_df in df.groupby("Farm_ID"):
            axis.plot(farm_df["time_label"], farm_df[column], label=farm_id)
        axis.set_title(f"{column} Over Time")
        axis.legend()
    axes[-1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "climate_trends.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column="Water_Weekly_L", by="Quarter", ax=ax)
    ax.set_title("Water Usage by Quarter")
    ax.set_ylabel("Water_Weekly_L")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "quarterly_water_usage.png")
    plt.close(fig)

    correlation_frame = df.select_dtypes(include="number").corr(numeric_only=True)
    correlation_frame.to_csv(EDA_OUTPUT_DIR / "correlations.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(correlation_frame, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(correlation_frame.columns)))
    ax.set_xticklabels(correlation_frame.columns, rotation=90)
    ax.set_yticks(range(len(correlation_frame.index)))
    ax.set_yticklabels(correlation_frame.index)
    ax.set_title("Numeric Feature Correlations")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "correlations.png")
    plt.close(fig)

    print("\nQuarterly seasonality summary:")
    print(
        df.groupby(["Region", "Quarter"])["Water_Weekly_L"]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )

    print("\nModeling limitations:")
    print("- Dataset covers 3 farms only, so spatial generalisation is weak.")
    print("- Rows are simulated historical observations, not measured crop-water balance outcomes.")
    print("- No crop type, land area, rainfall, soil moisture, ET0, or growth stage columns exist.")
    print(f"- Saved plots and tables to {EDA_OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
