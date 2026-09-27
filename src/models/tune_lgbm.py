"""
Hyperparameter tuning for LightGBM with Optuna.

Run: python -m src.models.tune_lgbm
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import optuna
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

N_TRIALS = 30


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


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 3),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))


def main():
    print("=" * 60)
    print("LIGHTGBM HYPERPARAMETER TUNING (Optuna)")
    print("=" * 60)

    train, test = load_splits()
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)

    val_size = int(0.15 * len(X_train))
    X_tr = X_train.iloc[:-val_size]
    y_tr = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]

    print(f"\nTrain: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize", study_name="lgbm_tuning")
    print(f"\nRunning {N_TRIALS} trials...")
    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )
    print(f"\nTuning completed in {time.time()-t0:.1f}s")
    print(f"Best RMSE (val): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    print("\nTraining final model on full training set...")
    best = lgb.LGBMRegressor(
        **study.best_params,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    best.fit(X_train, y_train)

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)

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
        eval_set(y_train, y_pred_train, "LightGBM-Tuned (train)"),
        eval_set(y_test, y_pred_test, "LightGBM-Tuned (test)"),
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODELS_DIR / "lightgbm_tuned.pkl")
    print(f"\nSaved tuned model to {MODELS_DIR / 'lightgbm_tuned.pkl'}")

    if RESULTS_PATH.exists():
        existing = pd.read_csv(RESULTS_PATH)
        results = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"Saved results to {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
