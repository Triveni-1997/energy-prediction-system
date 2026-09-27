"""
Test the prediction service end-to-end.

Run: python -m src.models.test_prediction_service
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.prediction_service import PredictionService


def main():
    print("=" * 60)
    print("PREDICTION SERVICE TEST")
    print("=" * 60)

    service = PredictionService(
        PROJECT_ROOT / "models_saved" / "lightgbm_tuned.pkl",
        PROJECT_ROOT / "models_saved" / "scaler_X.pkl",
        PROJECT_ROOT / "models_saved" / "scaler_y.pkl",
    )

    print("\nFeature list:")
    print(service.get_feature_list())

    sample = {
        "timestamp": "2010-06-15 14:00:00",
        "lag_1h": 1.25,
        "lag_2h": 1.10,
        "lag_3h": 0.95,
        "lag_24h": 1.40,
        "lag_48h": 1.35,
        "lag_168h": 1.20,
        "rolling_24h_mean": 1.15,
        "rolling_24h_std": 0.45,
        "rolling_168h_mean": 1.10,
        "rolling_168h_std": 0.50,
        "temperature": 22.0,
        "humidity": 55.0,
        "wind_speed": 8.0,
    }

    print("\nInput:")
    for k, v in sample.items():
        print(f"  {k}: {v}")

    result = service.predict(sample)
    print("\nPrediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("PREDICTION SERVICE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
