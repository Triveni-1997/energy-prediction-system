"""
Entry point for the Flask development server.

Run: python run.py
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Energy Prediction API...")
    print("Visit: http://localhost:5000/health")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
