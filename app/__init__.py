"""
Flask application factory.
"""

from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS


def create_app(config_object="app.config.Config"):
    """Create and configure the Flask app."""
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Enable CORS (frontend will call this from a different port)
    CORS(app)

    # Ensure database folder exists
    project_root = Path(app.root_path).parent
    (project_root / "database").mkdir(parents=True, exist_ok=True)

    # --- Health check endpoint ---
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "service": "energy-prediction-api",
            "version": "0.1.0"
        }), 200

    # --- Root endpoint ---
    @app.route("/", methods=["GET"])
    def root():
        return jsonify({
            "message": "Energy Consumption Prediction API",
            "endpoints": ["/health", "/api/predict", "/api/history", "/api/alerts"]
        }), 200

    # --- Error handlers ---
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app
