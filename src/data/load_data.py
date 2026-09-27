"""
Data loading utilities for the energy prediction project.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
WEATHER_DATA_PATH = PROJECT_ROOT / "data" / "external" / "weather.csv"


def load_energy_data(path=RAW_DATA_PATH):
    """Load raw energy consumption data from the UCI dataset."""
    print(f"Loading energy data from {path}...")

    df = pd.read_csv(
        path,
        sep=";",
        low_memory=False,
        na_values=["?", "nan"]
    )

    # Combine Date + Time into a single timestamp
    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"])

    # Rename columns
    df.columns = [
        "global_active_power",
        "global_reactive_power",
        "voltage",
        "global_intensity",
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3",
        "timestamp"
    ]

    # Reorder and set index
    cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
    df = df[cols].set_index("timestamp").sort_index()

    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Missing values total: {df.isnull().sum().sum()}")

    return df


def load_weather_data(path=WEATHER_DATA_PATH):
    """Load weather data from CSV."""
    print(f"Loading weather data from {path}...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    print(f"Loaded {len(df)} weather records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df


if __name__ == "__main__":
    energy = load_energy_data()
    print("\n--- Energy Data Sample ---")
    print(energy.head())

    weather = load_weather_data()
    print("\n--- Weather Data Sample ---")
    print(weather.head())
