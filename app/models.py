"""
SQLAlchemy database models.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120))
    role = db.Column(db.String(20), default="user")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    predicted_power_kw = db.Column(db.Float, nullable=False)
    model_used = db.Column(db.String(50), default="lightgbm_tuned")
    input_features = db.Column(db.Text)
    notes = db.Column(db.String(255))

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "predicted_power_kw": self.predicted_power_kw,
            "model_used": self.model_used,
            "notes": self.notes,
        }


class Alert(db.Model):
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    severity = db.Column(db.String(20), default="info")
    message = db.Column(db.String(500), nullable=False)
    acknowledged = db.Column(db.Boolean, default=False)
    acknowledged_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "severity": self.severity,
            "message": self.message,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
        }


class EnergyReading(db.Model):
    __tablename__ = "energy_readings"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    global_active_power = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    source = db.Column(db.String(50), default="historical")

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "global_active_power": self.global_active_power,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "source": self.source,
        }


def init_db(app):
    """Initialize database with app context."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return db
