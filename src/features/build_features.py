"""
Feature engineering for energy consumption prediction.

Scope:
- Time-based features
- Lag features (t-1h, t-24h, t-168h)
- Rolling averages (24h, 168h)

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
    """Add calendar-based features."""
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

    print(f"  -> shape now: {df.shape}")
    return df


def add_lag_features(df, target="global_active_power"):
    """Add lagged values of the target."""
    print("Adding lag features (t-1h, t-2h, t-3h, t-24h, t-48h, t-168h)...")

    df["lag_1h"] = df[target].shift(1)
    df["lag_2h"] = df[target].shift(2)
    df["lag_3h"] = df[target].shift(3)
    df["lag_24h"] = df[target].shift(24)
    df["lag_48h"] = df[target].shift(48)
    df["lag_168h"] = df[target].shift(168)  # 1 week

    print(f"  -> shape now: {df.shape}")
    return df


def add_rolling_features(df, target="global_active_power"):
    """Add rolling mean and std over 24h and 168h windows."""
    print("Adding rolling features (24h mean/std, 168h mean/std)...")

    # Shift by 1 first so we don't leak the current value
    shifted = df[target].shift(1)

    df["rolling_24h_mean"] = shifted.rolling(window=24).mean()
    df["rolling_24h_std"] = shifted.rolling(window=24).std()
    df["rolling_168h_mean"] = shifted.rolling(window=168).mean()
    df["rolling_168h_std"] = shifted.rolling(window=168).std()

    print(f"  -> shape now: {df.shape}")
    return df


def drop_na_rows(df):
    """Drop rows with NaN introduced by lag/rolling."""
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with NaN (from lags/rolling)")
    print(f"Final shape: {df.shape}")
    return df


def save_features(df, path=FEATURES_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"Saved features to {path}")


def main():
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = load_clean_data()
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = drop_na_rows(df)

    print("\nFinal columns:")
    print(df.columns.tolist())

    print("\nSample:")
    print(df[["global_active_power", "lag_1h", "lag_24h",
              "rolling_24h_mean", "rolling_168h_mean"]].head())

    save_features(df)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()