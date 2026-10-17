"""
HTML page routes (separate from the JSON API).
"""

from flask import Blueprint, render_template, redirect, url_for


pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def home():
    """Default entry point - send user to login."""
    return redirect(url_for("pages.login_page"))


@pages_bp.route("/login")
def login_page():
    return render_template("login.html")


@pages_bp.route("/register")
def register_page():
    return render_template("register.html")


@pages_bp.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")