"""
Fetch hourly weather data using Meteostat (v1.7+).
No API key needed - uses public weather station data.
Location: Paris, France (UCI dataset origin)
"""

from datetime import datetime
import pandas as pd
import meteostat as ms

LATITUDE = 48.8566
LONGITUDE = 2.3522


def fetch_weather(start_date, end_date, output_path="data/external/weather.csv"):
    print(f"Fetching weather from {start_date} to {end_date}...")
    point = ms.Point(LATITUDE, LONGITUDE)
    
    # v1.7+ API: use Hourly class
    ts = ms.Hourly(point, start_date, end_date)
    df = ts.fetch()
    
    if df is None or df.empty:
        raise ValueError("No weather data returned.")
    
    df = df.reset_index().rename(columns={"time": "timestamp"})
    
    keep_cols = ["timestamp"]
    rename_map = {}
    for col, new_name in [("temp", "temperature"),
                          ("rhum", "humidity"),
                          ("prcp", "precipitation"),
                          ("wspd", "wind_speed")]:
        if col in df.columns:
            keep_cols.append(col)
            rename_map[col] = new_name
    
    df = df[keep_cols].rename(columns=rename_map)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} weather records to {output_path}")
    print(df.head())
    return df


if __name__ == "__main__":
    start = datetime(2006, 12, 16)
    end = datetime(2010, 11, 26)
    fetch_weather(start, end)
