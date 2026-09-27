"""
Train LightGBM baseline.

Run: python -m src.models.train_lgbm
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models_saved"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "model_results.csv"

TARGET = "global_active_power"
DROP_COLS = [
    "global_active_power", "sub_metering_1", "sub_metering_2", "sub_metering_3",
    "global_intensity", "global_reactive_power", "voltage",
]


def load_splits():
    print("Loading train/test splits...")
    train = pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"], index_col="timestamp")
    test = pd.read_csv(TEST_PATH, parse_dates=["timestamp"], index_col="timestamp")
    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")
    return train, test


def prepare_xy(df):
    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df[TARGET]
    return X, y


def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {model_name} ---")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


def main():
    print("=" * 60)
    print("LIGHTGBM BASELINE")
    print("=" * 60)

    train, test = load_splits()
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)

    print(f"\nFeatures ({len(X_train.columns)})")

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    print("\nTraining LightGBM...")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = [
        evaluate(y_train, y_pred_train, "LightGBM (train)"),
        evaluate(y_test, y_pred_test, "LightGBM (test)"),
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "lightgbm.pkl")
    print(f"\nSaved model to {MODELS_DIR / 'lightgbm.pkl'}")

    if RESULTS_PATH.exists():
        existing = pd.read_csv(RESULTS_PATH)
        results = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"Saved results to {RESULTS_PATH}")

    print("\nTop 10 features (LightGBM importance):")
    imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(10)
    print(imp.to_string(index=False))

    print("\n" + "=" * 60)
    print("LIGHTGBM COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
