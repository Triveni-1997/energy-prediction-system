"""
EDA Part 1: Basic statistics and missing value analysis.
Run: python -m src.eda.basic_stats
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.load_data import load_energy_data, load_weather_data


def main():
    print("=" * 60)
    print("EDA PART 1: BASIC STATISTICS")
    print("=" * 60)

    energy = load_energy_data()
    weather = load_weather_data()

    print("\n=== ENERGY DATA ===")
    print(f"Shape: {energy.shape}")
    print(f"Date range: {energy.index.min()} to {energy.index.max()}")
    print(f"\nData types:\n{energy.dtypes}")

    print("\nFirst 5 rows:")
    print(energy.head())

    print("\nDescriptive statistics:")
    print(energy.describe().round(3))

    print("\n=== MISSING VALUES ===")
    missing = energy.isnull().sum()
    missing_pct = (missing / len(energy) * 100).round(3)
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    }).sort_values("Missing Count", ascending=False)
    print(missing_df)

    zero_count = (energy["global_active_power"] == 0).sum()
    print(f"\nZero consumption rows: {zero_count} ({zero_count/len(energy)*100:.2f}%)")

    print("\n=== HOURLY RESAMPLE ===")
    energy_hourly = energy["global_active_power"].resample("h").mean()
    print(f"Hourly data points: {len(energy_hourly)}")
    print(f"Null after resample: {energy_hourly.isnull().sum()}")
    print("\nFirst 5 hourly values:")
    print(energy_hourly.head())

    print("\n=== WEATHER DATA ===")
    print(f"Shape: {weather.shape}")
    print(f"Date range: {weather.index.min()} to {weather.index.max()}")
    print(f"\nWeather missing values:\n{weather.isnull().sum()}")

    print("\n" + "=" * 60)
    print("EDA PART 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
