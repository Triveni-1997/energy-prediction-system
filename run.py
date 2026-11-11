"""
Entry point for the Flask development server.
"""
import webbrowser
from threading import Timer
from pathlib import Path
from app import create_app
from app.services.prediction_service import PredictionService

app = create_app()

# OVERRIDE THE PATH HERE FOR THE CLOUD
# This forces the app to look in the actual models_saved folder at the root
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models_saved" / "lightgbm_tuned.pkl"
SCALER_X = PROJECT_ROOT / "models_saved" / "scaler_X.pkl"
SCALER_Y = PROJECT_ROOT / "models_saved" / "scaler_y.pkl"

# Inject the correct model path into the app config
app.config["PRODUCTION_MODEL"] = MODEL_PATH
app.config["SCALER_X"] = SCALER_X
app.config["SCALER_Y"] = SCALER_Y

def open_browser():
    webbrowser.open_new("http://localhost:5000/")

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Energy Prediction API...")
    print("Opening: http://localhost:5000/")
    print("=" * 60)

    Timer(1.2, open_browser).start()

    app.run(host="0.0.0.0", port=5000, debug=True)