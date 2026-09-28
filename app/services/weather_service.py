"""
Weather service - fetches current weather from OpenWeatherMap.

Falls back to realistic defaults if API key is missing or request fails.
"""

import os
import requests
from datetime import datetime


DEFAULT_WEATHER = {
    "temperature": 20.0,
    "humidity": 60.0,
    "wind_speed": 5.0,
    "description": "fallback (no API key)",
}


def fetch_current_weather(city="Paris", api_key=None):
    """Fetch current weather for a city."""
    api_key = api_key or os.getenv("WEATHER_API_KEY", "").strip()

    if not api_key:
        return {
            **DEFAULT_WEATHER,
            "city": city,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "fallback",
        }

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
        r = requests.get(url, params=params, timeout=8)

        if r.status_code != 200:
            return {
                **DEFAULT_WEATHER,
                "city": city,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "fallback",
                "error": f"API returned status {r.status_code}",
            }

        data = r.json()
        return {
            "temperature": float(data["main"]["temp"]),
            "humidity": float(data["main"]["humidity"]),
            "wind_speed": float(data["wind"]["speed"]),
            "description": data["weather"][0]["description"],
            "city": data.get("name", city),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "openweathermap",
        }

    except Exception as e:
        return {
            **DEFAULT_WEATHER,
            "city": city,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "fallback",
            "error": str(e),
        }
