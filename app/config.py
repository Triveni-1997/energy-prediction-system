"""
Application configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    # Flask
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-me")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{PROJECT_ROOT / 'database' / 'energy.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT
    JWT_SECRET = os.getenv("JWT_SECRET", "jwt-secret-change-me")
    JWT_EXPIRY_HOURS = 24

    # Paths
        # Define the root of the project properly for both local and cloud
    MODELS_DIR = Path(__file__).resolve().parents[1] / "models_saved"
    PRODUCTION_MODEL = MODELS_DIR / "lightgbm_tuned.pkl"
    SCALER_X = MODELS_DIR / "scaler_X.pkl"
    SCALER_Y = MODELS_DIR / "scaler_y.pkl"

    # Weather API (optional)
    WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
