"""
OpticCrop: Multi-Modal Weather & Spatial Caching Service
Integrates Open-Meteo API, Geohash Level-6 spatial caching (~1.2 km²),
and graceful fallback to 30-year historical climate averages.
"""

import time
import os
import sys
import requests
from typing import Dict, Any, Optional

# Ensure workspace root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.db import get_historical_climate_fallback, get_weather_cache, set_weather_cache

# Geohash Base32 Character Map
GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(lat: float, lon: float, precision: int = 6) -> str:
    """Encodes latitude and longitude into a standard Base32 Geohash."""
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    is_even = True

    while len(geohash) < precision:
        if is_even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if lon > mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if lat > mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid

        is_even = not is_even
        if bit < 4:
            bit += 1
        else:
            geohash.append(GEOHASH_BASE32[ch])
            bit = 0
            ch = 0

    return "".join(geohash)





def fetch_weather_stream(
    latitude: float,
    longitude: float,
    forecast_window_days: int = 14,
    timeout_seconds: float = 0.8,
) -> Dict[str, Any]:
    """
    Fetches real-time weather & forecast vector with Geohash level-6 caching
    and graceful fallback to 30-year historical baseline if latency > 800ms.
    """
    t0 = time.perf_counter()
    geohash6 = encode_geohash(latitude, longitude, precision=6)

    # 1. Check Spatial Cache (Geohash Level-6, 1hr TTL)
    cached_payload = get_weather_cache(geohash6)
    if cached_payload is not None:
        fetch_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "source": "cache",
            "geohash6": geohash6,
            "weather_fetch_ms": round(fetch_ms, 2),
            **cached_payload,
        }

    # 2. Live Meteorological API Ingestion (Open-Meteo)
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}&"
            f"current=temperature_2m,relative_humidity_2m,precipitation&"
            f"daily=temperature_2m_max,temperature_2m_min,precipitation_sum&"
            f"forecast_days={min(max(forecast_window_days, 1), 16)}&"
            f"timezone=auto"
        )
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        cur_temp = current.get("temperature_2m", 25.0)
        cur_humidity = current.get("relative_humidity_2m", 70.0)

        # Calculate seasonal/forecast rainfall sum
        precip_sums = daily.get("precipitation_sum", [3.0])
        total_precip_forecast = sum(precip_sums)
        # Scaled representative seasonal rainfall parameter
        daily_mean = total_precip_forecast / max(len(precip_sums), 1)
        estimated_seasonal_rainfall = max(25.0, round(daily_mean * 25.0 + 45.0, 1))

        weather_data = {
            "temperature_avg": round(float(cur_temp), 1),
            "humidity_avg": round(float(cur_humidity), 1),
            "rainfall_forecast_sum": round(float(total_precip_forecast), 1),
            "rainfall_equivalent": estimated_seasonal_rainfall,
            "forecast_window_days": forecast_window_days,
            "is_fallback": False,
        }

        # Store in Geohash Cache
        set_weather_cache(
            geohash6=geohash6, 
            lat=latitude, 
            lon=longitude, 
            forecast_days=forecast_window_days, 
            temp=weather_data["temperature_avg"], 
            hum=weather_data["humidity_avg"], 
            rain=weather_data["rainfall_equivalent"]
        )

        fetch_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "source": "live_api",
            "geohash6": geohash6,
            "weather_fetch_ms": round(fetch_ms, 2),
            **weather_data,
        }

    except Exception as e:
        # 3. Graceful Fallback to 30-Year Historical Climate Baseline in SQLite
        fallback = get_historical_climate_fallback(latitude, longitude, geohash6)
        fetch_ms = (time.perf_counter() - t0) * 1000.0

        fallback_data = {
            "temperature_avg": round(fallback["temperature"], 1),
            "humidity_avg": round(fallback["humidity"], 1),
            "rainfall_forecast_sum": 0.0,
            "rainfall_equivalent": round(fallback["rainfall"], 1),
            "forecast_window_days": forecast_window_days,
            "is_fallback": True,
            "fallback_reason": str(e),
            "fallback_region": fallback.get("fallback_region", "Regional Normal"),
        }

        return {
            "source": "historical_fallback",
            "geohash6": geohash6,
            "weather_fetch_ms": round(fetch_ms, 2),
            **fallback_data,
        }


def geocode_location(city_name: str) -> Dict[str, Any]:
    """Geocodes city name to latitude and longitude."""
    if not city_name or not city_name.strip():
        return {"success": False, "error": "Location query cannot be empty"}

    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name.strip()}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url, timeout=3.0)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        results = geo_data.get("results")
        if not results or len(results) == 0:
            return {"success": False, "error": f"Location '{city_name}' not found."}

        res = results[0]
        return {
            "success": True,
            "name": res.get("name"),
            "admin1": res.get("admin1", ""),
            "country": res.get("country", ""),
            "latitude": res.get("latitude"),
            "longitude": res.get("longitude"),
            "geohash6": encode_geohash(res.get("latitude"), res.get("longitude"), 6),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    lat, lon = 13.0827, 80.2707  # Chennai / Tamil Nadu
    print("Geohash Level-6:", encode_geohash(lat, lon, 6))
    w = fetch_weather_stream(lat, lon, forecast_window_days=14)
    print("Weather Stream:", w)
