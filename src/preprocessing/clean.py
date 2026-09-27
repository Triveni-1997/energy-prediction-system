"""
Data cleaning pipeline for energy consumption data.

Steps:
1. Load raw energy data
2. Convert to numeric, drop rows with missing target
3. Remove duplicate timestamps
4. Cap extreme outliers (IQR * 3)
5. Resample to hourly (mean)
6. Save to data/processed/energy_clean.csv

Run: python -m src.preprocessing.clean
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.load_data import load_energy_data


PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "energy_clean.csv"


def clean_energy_data(df=None):
    """Clean the raw energy consumption data and resample hourly."""
    if df is None:
        df = load_energy_data()

    print(f"\n[1] Original shape: {df.shape}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["global_active_power"])
    print(f"[2] Dropped {before - len(df)} rows with missing target")

    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    print(f"[3] Dropped {before - len(df)} duplicate timestamps")

    Q1 = df["global_active_power"].quantile(0.25)
    Q3 = df["global_active_power"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR

    before = len(df)
    df = df[(df["global_active_power"] >= lower) & (df["global_active_power"] <= upper)]
    print(f"[4] Dropped {before - len(df)} outlier rows")
    print(f"    Target range after cleaning: {df['global_active_power'].min():.3f} - {df['global_active_power'].max():.3f}")

    print("[5] Resampling to hourly...")
    df_hourly = df.resample("h").mean()
    df_hourly = df_hourly.dropna()
    print(f"    Final shape: {df_hourly.shape}")
    print(f"    Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")

    return df_hourly


def save_processed(df, path=PROCESSED_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"\n[6] Saved cleaned data to {path}")


def main():
    print("=" * 60)
    print("DATA CLEANING PIPELINE")
    print("=" * 60)

    df_clean = clean_energy_data()
    save_processed(df_clean)

    print("\nSample of cleaned data:")
    print(df_clean.head())

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
