"""
API routes for the Energy Prediction service.
"""

from flask import Blueprint, request, jsonify, current_app

from app.models import db, Prediction, Alert, EnergyReading, User
from app.services.prediction_service import PredictionService
from app.services.auth_service import jwt_required
from app.services.weather_service import fetch_current_weather


api_bp = Blueprint("api", __name__, url_prefix="/api")

_prediction_service = None


def get_prediction_service():
    global _prediction_service
    if _prediction_service is None:
        cfg = current_app.config
        _prediction_service = PredictionService(
            cfg["PRODUCTION_MODEL"],
            cfg["SCALER_X"],
            cfg["SCALER_Y"],
        )
    return _prediction_service


@api_bp.route("/predict", methods=["POST"])
@jwt_required
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    try:
        service = get_prediction_service()
        result = service.predict(data)

        pred_row = Prediction(
            predicted_power_kw=result["predicted_power_kw"],
            model_used=result["model_used"],
            notes="via /api/predict",
        )
        db.session.add(pred_row)

        if result["predicted_power_kw"] > 3.0:
            alert = Alert(
                severity="warning",
                message=f"High predicted consumption: {result['predicted_power_kw']:.3f} kW",
            )
            db.session.add(alert)

        db.session.commit()
        result["prediction_id"] = pred_row.id
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@api_bp.route("/predictions", methods=["GET"])
def list_predictions():
    limit = min(int(request.args.get("limit", 20)), 100)
    rows = Prediction.query.order_by(Prediction.timestamp.desc()).limit(limit).all()
    return jsonify({
        "count": len(rows),
        "predictions": [r.to_dict() for r in rows],
    }), 200


@api_bp.route("/history", methods=["GET"])
def history():
    limit = min(int(request.args.get("limit", 100)), 1000)
    rows = EnergyReading.query.order_by(EnergyReading.timestamp.desc()).limit(limit).all()
    return jsonify({
        "count": len(rows),
        "readings": [r.to_dict() for r in rows],
    }), 200


@api_bp.route("/alerts", methods=["GET"])
def list_alerts():
    include_ack = request.args.get("include_acknowledged", "false").lower() == "true"
    q = Alert.query
    if not include_ack:
        q = q.filter_by(acknowledged=False)
    rows = q.order_by(Alert.timestamp.desc()).limit(100).all()
    return jsonify({
        "count": len(rows),
        "alerts": [r.to_dict() for r in rows],
    }), 200


@api_bp.route("/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id):
    alert = Alert.query.get(alert_id)
    if not alert:
        return jsonify({"error": "Alert not found"}), 404
    alert.acknowledged = True
    db.session.commit()
    return jsonify({"message": "Alert acknowledged", "alert": alert.to_dict()}), 200


@api_bp.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "total_predictions": Prediction.query.count(),
        "total_alerts": Alert.query.count(),
        "unacknowledged_alerts": Alert.query.filter_by(acknowledged=False).count(),
        "total_users": User.query.count(),
    }), 200



@api_bp.route("/weather", methods=["GET"])
def weather():
    """Fetch current weather for a city (default: Paris)."""
    city = request.args.get("city", "Paris")
    data = fetch_current_weather(city=city)
    return jsonify(data), 200
