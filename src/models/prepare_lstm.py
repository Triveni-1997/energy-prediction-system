"""
Prepare sequence data for LSTM training.

Creates 3D tensors of shape: (samples, timesteps=24, features)

Run: python -m src.models.prepare_lstm
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models_saved"

TARGET = "global_active_power"
DROP_COLS = [
    "global_active_power", "sub_metering_1", "sub_metering_2", "sub_metering_3",
    "global_intensity", "global_reactive_power", "voltage",
]

TIMESTEPS = 24


def load_splits():
    print("Loading train/test splits...")
    train = pd.read_csv(PROCESSED / "train.csv",
                        parse_dates=["timestamp"], index_col="timestamp")
    test = pd.read_csv(PROCESSED / "test.csv",
                       parse_dates=["timestamp"], index_col="timestamp")
    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")
    return train, test


def make_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i - timesteps:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def main():
    print("=" * 60)
    print("PREPARE LSTM SEQUENCES")
    print("=" * 60)

    train, test = load_splits()

    X_train_df = train.drop(columns=DROP_COLS, errors="ignore")
    y_train_s = train[TARGET]
    X_test_df = test.drop(columns=DROP_COLS, errors="ignore")
    y_test_s = test[TARGET]

    print(f"\nFeatures: {X_train_df.shape[1]}")

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_df)
    X_test_scaled = scaler_X.transform(X_test_df)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_s.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test_s.values.reshape(-1, 1)).flatten()

    print(f"\nCreating sequences (timesteps={TIMESTEPS})...")
    X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train_scaled, TIMESTEPS)
    X_test_seq, y_test_seq = make_sequences(X_test_scaled, y_test_scaled, TIMESTEPS)

    print(f"  X_train_seq: {X_train_seq.shape}")
    print(f"  y_train_seq: {y_train_seq.shape}")
    print(f"  X_test_seq:  {X_test_seq.shape}")
    print(f"  y_test_seq:  {y_test_seq.shape}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / "lstm_X_train.npy", X_train_seq)
    np.save(MODELS_DIR / "lstm_y_train.npy", y_train_seq)
    np.save(MODELS_DIR / "lstm_X_test.npy", X_test_seq)
    np.save(MODELS_DIR / "lstm_y_test.npy", y_test_seq)
    joblib.dump(scaler_X, MODELS_DIR / "scaler_X.pkl")
    joblib.dump(scaler_y, MODELS_DIR / "scaler_y.pkl")

    print(f"\nSaved to {MODELS_DIR}/:")
    print("  lstm_X_train.npy")
    print("  lstm_y_train.npy")
    print("  lstm_X_test.npy")
    print("  lstm_y_test.npy")
    print("  scaler_X.pkl")
    print("  scaler_y.pkl")

    print("\n" + "=" * 60)
    print("LSTM DATA READY")
    print("=" * 60)


if __name__ == "__main__":
    main()
