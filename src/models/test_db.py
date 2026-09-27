"""
Test database creation and basic operations.

Run: python -m src.models.test_db
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app
from app.models import db, User, Prediction, Alert, EnergyReading


def main():
    print("=" * 60)
    print("DATABASE TEST")
    print("=" * 60)

    app = create_app()
    with app.app_context():
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"Tables created: {tables}")

        existing = User.query.filter_by(email="test@example.com").first()
        if not existing:
            user = User(
                email="test@example.com",
                password_hash="dummy_hash",
                full_name="Test User",
                role="admin",
            )
            db.session.add(user)
            db.session.commit()
            print(f"Created user: {user.email} (id={user.id})")
        else:
            print(f"User already exists: {existing.email}")

        print(f"\nRecord counts:")
        print(f"  Users: {User.query.count()}")
        print(f"  Predictions: {Prediction.query.count()}")
        print(f"  Alerts: {Alert.query.count()}")
        print(f"  Energy readings: {EnergyReading.query.count()}")

    print("\n" + "=" * 60)
    print("DATABASE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
