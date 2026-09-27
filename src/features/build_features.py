"""
Feature engineering for energy consumption prediction.

Day 1 scope: time-based features only.

Run: python -m src.features.build_features
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "energy_clean.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"


def load_clean_data(path=PROCESSED_PATH):
    """Load the cleaned hourly energy dataset."""
    print(f"Loading cleaned data from {path}...")
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df


def add_time_features(df):
    """Add calendar-based features from the timestamp index."""
    print("Adding time-based features...")
    idx = df.index

    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["dayofyear"] = idx.dayofyear
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    print(f"Added 10 time features. New shape: {df.shape}")
    return df


def main():
    print("=" * 60)
    print("FEATURE ENGINEERING - DAY 1 (TIME FEATURES)")
    print("=" * 60)

    df = load_clean_data()
    df = add_time_features(df)

    print("\nSample with new features:")
    cols_to_show = ["global_active_power", "hour", "dayofweek", "month", "is_weekend"]
    print(df[cols_to_show].head(10))

    print("\n" + "=" * 60)
    print("DAY 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
