"""
End-to-end API smoke test.

Requires the Flask server running on http://localhost:5000.

Run: python -m src.models.test_api
"""

import sys
import requests

BASE = "http://localhost:5000"


def check(label, ok, detail=""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label} {detail}")
    return ok


def main():
    print("=" * 60)
    print("API SMOKE TEST")
    print("=" * 60)

    results = []

    r = requests.get(f"{BASE}/health", timeout=5)
    results.append(check("GET /health", r.status_code == 200))

    r = requests.get(f"{BASE}/", timeout=5)
    results.append(check("GET /", r.status_code == 200))

    r = requests.get(f"{BASE}/api/stats", timeout=5)
    results.append(check("GET /api/stats", r.status_code == 200))

    r = requests.post(f"{BASE}/api/auth/register", json={
        "email": "smoketest@test.com",
        "password": "test1234",
        "full_name": "Smoke Test"
    }, timeout=5)
    registered = r.status_code in (201, 409)
    results.append(check("POST /api/auth/register", registered, f"({r.status_code})"))

    r = requests.post(f"{BASE}/api/auth/login", json={
        "email": "smoketest@test.com",
        "password": "test1234"
    }, timeout=5)
    ok = r.status_code == 200
    results.append(check("POST /api/auth/login", ok))
    token = r.json().get("token") if ok else None
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    r = requests.post(f"{BASE}/api/predict", json={}, timeout=5)
    results.append(check("POST /api/predict (no token -> 401)", r.status_code == 401))

    if token:
        body = {
            "timestamp": "2010-06-15 14:00:00",
            "lag_1h": 1.25, "lag_2h": 1.10, "lag_3h": 0.95,
            "lag_24h": 1.40, "lag_48h": 1.35, "lag_168h": 1.20,
            "rolling_24h_mean": 1.15, "rolling_24h_std": 0.45,
            "rolling_168h_mean": 1.10, "rolling_168h_std": 0.50,
            "temperature": 22.0, "humidity": 55.0, "wind_speed": 8.0
        }
        r = requests.post(f"{BASE}/api/predict", json=body, headers=headers, timeout=10)
        detail = f"pred={r.json().get('predicted_power_kw')}" if r.status_code == 200 else ""
        results.append(check("POST /api/predict (with token)", r.status_code == 200, detail))
    else:
        results.append(check("POST /api/predict (with token)", False, "no token"))

    if token:
        r = requests.get(f"{BASE}/api/auth/me", headers=headers, timeout=5)
        results.append(check("GET /api/auth/me", r.status_code == 200))

    r = requests.get(f"{BASE}/api/predictions", timeout=5)
    results.append(check("GET /api/predictions", r.status_code == 200))

    r = requests.get(f"{BASE}/api/alerts", timeout=5)
    results.append(check("GET /api/alerts", r.status_code == 200))

    r = requests.get(f"{BASE}/api/weather", timeout=10)
    src = r.json().get("source", "?") if r.status_code == 200 else ""
    results.append(check("GET /api/weather", r.status_code == 200, f"({src})"))

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULT: {passed}/{total} passed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
