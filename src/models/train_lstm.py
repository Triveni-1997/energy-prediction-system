"""
Train LSTM model for energy consumption prediction.

Run: python -m src.models.train_lstm
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

MODELS_DIR = PROJECT_ROOT / "models_saved"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "model_results.csv"

np.random.seed(42)
tf.random.set_seed(42)


def load_sequences():
    print("Loading sequences...")
    X_train = np.load(MODELS_DIR / "lstm_X_train.npy")
    y_train = np.load(MODELS_DIR / "lstm_y_train.npy")
    X_test = np.load(MODELS_DIR / "lstm_X_test.npy")
    y_test = np.load(MODELS_DIR / "lstm_y_test.npy")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    return X_train, y_train, X_test, y_test


def build_model(timesteps, n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(timesteps, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    print("=" * 60)
    print("LSTM TRAINING")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_sequences()
    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    model = build_model(timesteps, n_features)
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    print("\nTraining LSTM...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=30,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1,
    )
    print(f"Trained in {time.time()-t0:.1f}s")

    y_pred_train = model.predict(X_train, verbose=0).flatten()
    y_pred_test = model.predict(X_test, verbose=0).flatten()

    scaler_y = joblib.load(MODELS_DIR / "scaler_y.pkl")
    y_train_real = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_train_real = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
    y_pred_test_real = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

    def eval_set(y_true, y_pred, name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n--- {name} ---")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
        return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}

    results = [
        eval_set(y_train_real, y_pred_train_real, "LSTM (train)"),
        eval_set(y_test_real, y_pred_test_real, "LSTM (test)"),
    ]

    model.save(MODELS_DIR / "lstm_model.keras")
    print(f"\nSaved model to {MODELS_DIR / 'lstm_model.keras'}")

    if RESULTS_PATH.exists():
        existing = pd.read_csv(RESULTS_PATH)
        results = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"Saved results to {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("LSTM COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
