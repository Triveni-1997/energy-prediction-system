"""
Train tree-based models: Random Forest + Gradient Boosting.

Run: python -m src.models.train_tree
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models_saved"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "model_results.csv"

TARGET = "global_active_power"

DROP_COLS = [
    "global_active_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
    "global_intensity",
    "global_reactive_power",
    "voltage",
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

    print(f"\n--- {model_name} Performance ---")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")

    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


def train_and_eval(model, model_name, X_train, y_train, X_test, y_test):
    print(f"\nTraining {model_name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  -> Trained in {elapsed:.1f}s")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = [
        evaluate(y_train, y_pred_train, f"{model_name} (train)"),
        evaluate(y_test, y_pred_test, f"{model_name} (test)"),
    ]
    return model, results


def main():
    print("=" * 60)
    print("TREE-BASED MODELS - DAY 6")
    print("=" * 60)

    train, test = load_splits()
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)

    all_results = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Random Forest ----
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42,
    )
    rf, rf_results = train_and_eval(rf, "RandomForest", X_train, y_train, X_test, y_test)
    all_results.extend(rf_results)

    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")
    print(f"  Saved to {MODELS_DIR / 'random_forest.pkl'}")

    # ---- 2. Gradient Boosting ----
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
    )
    gb, gb_results = train_and_eval(gb, "GradientBoosting", X_train, y_train, X_test, y_test)
    all_results.extend(gb_results)

    joblib.dump(gb, MODELS_DIR / "gradient_boosting.pkl")
    print(f"  Saved to {MODELS_DIR / 'gradient_boosting.pkl'}")

    # ---- Save results ----
    df_results = pd.DataFrame(all_results)

    if RESULTS_PATH.exists():
        existing = pd.read_csv(RESULTS_PATH)
        df_results = pd.concat([existing, df_results], ignore_index=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved all results to {RESULTS_PATH}")

    # ---- Feature importance (Random Forest) ----
    print("\nTop 10 features (Random Forest importance):")
    imp_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).head(10)
    print(imp_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("DAY 6 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
