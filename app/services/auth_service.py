"""
Authentication service - password hashing and JWT tokens.
"""

import jwt
from datetime import datetime, timedelta
from functools import wraps

from flask import request, jsonify, current_app, g
from werkzeug.security import generate_password_hash, check_password_hash

from app.models import db, User


def hash_password(password):
    return generate_password_hash(password)


def verify_password(password, password_hash):
    return check_password_hash(password_hash, password)


def create_token(user_id, expiry_hours=24):
    secret = current_app.config.get("JWT_SECRET", "jwt-secret")
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=expiry_hours),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token):
    secret = current_app.config.get("JWT_SECRET", "jwt-secret")
    return jwt.decode(token, secret, algorithms=["HS256"])


def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.replace("Bearer ", "").strip()

        try:
            payload = decode_token(token)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        user = User.query.get(payload.get("user_id"))
        if not user:
            return jsonify({"error": "User not found"}), 401

        g.current_user = user
        return f(*args, **kwargs)

    return decorated
