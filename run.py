"""
Entry point for the Flask development server.

Run: python run.py
"""

import webbrowser
from threading import Timer

from app import create_app

app = create_app()


def open_browser():
    webbrowser.open_new("http://localhost:5000/")


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Energy Prediction API...")
    print("Opening: http://localhost:5000/")
    print("=" * 60)

    # Open browser ~1.2 seconds after server boot
    Timer(1.2, open_browser).start()

    app.run(host="0.0.0.0", port=5000, debug=True)