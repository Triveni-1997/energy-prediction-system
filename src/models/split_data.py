"""
Time-based train/test split for energy consumption prediction.

Instead of random split, we split chronologically:
- Train: first 80% of the timeline
- Test:  last 20% of the timeline

This prevents data leakage and reflects real-world forecasting.

Run: python -m src.models.split_data
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = SPLIT_DIR / "train.csv"
TEST_PATH = SPLIT_DIR / "test.csv"


def load_features(path=FEATURES_PATH):
    """Load the engineered features."""
    print(f"Loading features from {path}...")
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total columns: {len(df.columns)}")
    return df


def time_split(df, train_frac=0.8):
    """Chronological split: first train_frac rows go to train."""
    n = len(df)
    split_idx = int(n * train_frac)

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    print(f"\nSplit at row {split_idx} (train_frac={train_frac})")
    print(f"  Train: {len(train)} rows ({train.index.min()} to {train.index.max()})")
    print(f"  Test:  {len(test)} rows ({test.index.min()} to {test.index.max()})")

    return train, test


def save_splits(train, test, train_path=TRAIN_PATH, test_path=TEST_PATH):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path)
    test.to_csv(test_path)
    print(f"\nSaved train to {train_path}")
    print(f"Saved test  to {test_path}")


def main():
    print("=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    df = load_features()
    train, test = time_split(df, train_frac=0.8)
    save_splits(train, test)

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
