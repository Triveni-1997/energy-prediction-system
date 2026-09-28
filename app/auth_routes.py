"""
Authentication routes: register, login, me.
"""

from flask import Blueprint, request, jsonify, g

from app.models import db, User
from app.services.auth_service import (
    hash_password, verify_password, create_token, jwt_required
)


auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.route("/register", methods=["POST"])
def register():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    full_name = data.get("full_name", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    user = User(
        email=email,
        password_hash=hash_password(password),
        full_name=full_name,
        role="user",
    )
    db.session.add(user)
    db.session.commit()

    token = create_token(user.id)
    return jsonify({
        "message": "User registered",
        "user": user.to_dict(),
        "token": token,
    }), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not verify_password(password, user.password_hash):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_token(user.id)
    return jsonify({
        "message": "Login successful",
        "user": user.to_dict(),
        "token": token,
    }), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required
def me():
    return jsonify({"user": g.current_user.to_dict()}), 200
