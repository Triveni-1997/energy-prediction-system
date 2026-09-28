"""
Flask application factory.
"""

from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS

from app.models import init_db


def create_app(config_object="app.config.Config"):
    app = Flask(__name__)
    app.config.from_object(config_object)
    CORS(app)

    project_root = Path(app.root_path).parent
    (project_root / "database").mkdir(parents=True, exist_ok=True)

    init_db(app)

    from app.routes import api_bp
    app.register_blueprint(api_bp)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "service": "energy-prediction-api",
            "version": "0.3.0"
        }), 200

    @app.route("/", methods=["GET"])
    def root():
        return jsonify({
            "message": "Energy Consumption Prediction API",
            "endpoints": [
                "/health",
                "/api/predict (POST)",
                "/api/predictions (GET)",
                "/api/history (GET)",
                "/api/alerts (GET)",
                "/api/alerts/<id>/acknowledge (POST)",
                "/api/stats (GET)"
            ]
        }), 200

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app
