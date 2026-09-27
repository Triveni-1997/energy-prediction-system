"""
Train baseline models for energy consumption prediction.

Day 5 scope: Linear Regression only.

Run: python -m src.models.train
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
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
    # Leaky features — correlated with target at the same timestamp
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
    """Separate features (X) and target (y)."""
    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df[TARGET]
    return X, y


def evaluate(y_true, y_pred, model_name):
    """Compute RMSE, MAE, R2."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {model_name} Performance ---")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")

    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


def train_linear_regression(X_train, y_train):
    print("\nTraining Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("  -> Trained.")
    return model


def main():
    print("=" * 60)
    print("BASELINE MODEL - LINEAR REGRESSION")
    print("=" * 60)

    train, test = load_splits()
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)

    print(f"\nFeature columns ({len(X_train.columns)}):")
    print(X_train.columns.tolist())

    model = train_linear_regression(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = []
    results.append(evaluate(y_train, y_pred_train, "LinearRegression (train)"))
    results.append(evaluate(y_test, y_pred_test, "LinearRegression (test)"))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "linear_regression.pkl"
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"Saved results to {RESULTS_PATH}")

    print("\nTop 10 most important features (by |coef|):")
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coef": model.coef_
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).head(10)
    print(coef_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("DAY 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
