"""
Prediction service - loads production model and provides inference.
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


FEATURE_COLUMNS = [
    "hour", "dayofweek", "month", "quarter", "dayofyear", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h",
    "rolling_24h_mean", "rolling_24h_std", "rolling_168h_mean", "rolling_168h_std",
    "temperature", "humidity", "wind_speed",
]


class PredictionService:
    """Loads and serves predictions from the trained LightGBM model."""

    def __init__(self, model_path, scaler_X_path=None, scaler_y_path=None):
        self.model_path = Path(model_path)
        self.scaler_X_path = Path(scaler_X_path) if scaler_X_path else None
        self.scaler_y_path = Path(scaler_y_path) if scaler_y_path else None

        self.model = None
        self.scaler_X = None
        self.scaler_y = None

        self._load()

    def _load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        print(f"  -> Model type: {type(self.model).__name__}")

        if self.scaler_X_path and self.scaler_X_path.exists():
            self.scaler_X = joblib.load(self.scaler_X_path)
            print(f"  -> Loaded scaler_X")

        if self.scaler_y_path and self.scaler_y_path.exists():
            self.scaler_y = joblib.load(self.scaler_y_path)
            print(f"  -> Loaded scaler_y")

    def _build_features(self, raw_input):
        ts = raw_input.get("timestamp")
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        elif ts is None:
            ts = datetime.utcnow()

        idx = pd.Timestamp(ts)

        feature_values = {
            "hour": idx.hour,
            "dayofweek": idx.dayofweek,
            "month": idx.month,
            "quarter": idx.quarter,
            "dayofyear": idx.dayofyear,
            "is_weekend": int(idx.dayofweek >= 5),
            "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
            "month_sin": np.sin(2 * np.pi * idx.month / 12),
            "month_cos": np.cos(2 * np.pi * idx.month / 12),
        }

        required_continuous = [
            "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h",
            "rolling_24h_mean", "rolling_24h_std",
            "rolling_168h_mean", "rolling_168h_std",
        ]
        for key in required_continuous:
            if key not in raw_input:
                raise ValueError(f"Missing required field: {key}")
            feature_values[key] = float(raw_input[key])

        feature_values["temperature"] = float(raw_input.get("temperature", 15.0))
        feature_values["humidity"] = float(raw_input.get("humidity", 60.0))
        feature_values["wind_speed"] = float(raw_input.get("wind_speed", 5.0))

        X = pd.DataFrame([feature_values])[FEATURE_COLUMNS]
        return X

    def predict(self, raw_input):
        X = self._build_features(raw_input)
        pred = float(self.model.predict(X)[0])

        return {
            "predicted_power_kw": round(pred, 4),
            "model_used": self.model_path.stem,
            "timestamp": str(raw_input.get("timestamp", datetime.utcnow())),
        }

    def get_feature_list(self):
        return FEATURE_COLUMNS
