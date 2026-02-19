"""
Weather tool with deterministic, short output for predictable TTS.
Uses Open-Meteo (no API key required).
"""
import os
import re
from typing import Any, Dict, Tuple

import requests


DEFAULT_LOCATION = os.getenv("WEATHER_DEFAULT_LOCATION", "Charlotte, NC")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("WEATHER_HTTP_TIMEOUT_SECONDS", "4.0"))

# Open-Meteo weather code mapping:
# https://open-meteo.com/en/docs
WEATHER_CODE_LABELS: Dict[int, str] = {
    0: "clear skies",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "snow showers",
    95: "thunderstorms",
    96: "thunderstorms with hail",
    99: "heavy thunderstorms with hail",
}


def _round_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _format_location_label(geo_result: Dict[str, Any]) -> str:
    name = (geo_result.get("name") or "").strip()
    admin1 = (geo_result.get("admin1") or "").strip()
    country = (geo_result.get("country") or "").strip()

    if name and admin1:
        return f"{name}, {admin1}"
    if name and country:
        return f"{name}, {country}"
    if name:
        return name
    return "your area"


def _resolve_location(location: str) -> Tuple[str, float, float]:
    query = (location or DEFAULT_LOCATION).strip() or DEFAULT_LOCATION
    normalized = re.sub(r"\s+", " ", query)

    candidates = [normalized]

    no_comma = normalized.replace(",", " ")
    no_comma = re.sub(r"\s+", " ", no_comma).strip()
    if no_comma and no_comma not in candidates:
        candidates.append(no_comma)

    if "," in normalized:
        city_only = normalized.split(",", 1)[0].strip()
        if city_only and city_only not in candidates:
            candidates.append(city_only)

    parts = no_comma.split()
    if len(parts) >= 2 and len(parts[-1]) == 2 and parts[-1].isalpha():
        without_state_code = " ".join(parts[:-1]).strip()
        if without_state_code and without_state_code not in candidates:
            candidates.append(without_state_code)

    for candidate in candidates:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": candidate,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if results:
            best = results[0]
            label = _format_location_label(best)
            lat = float(best["latitude"])
            lon = float(best["longitude"])
            return label, lat, lon

    raise ValueError("location_not_found")


def get_weather(location: str = "") -> str:
    """
    Return weather in one short fixed template for fast, predictable TTS:
    "Weather {location}: {condition}, {temp} degrees now, high {high}, low {low}, wind {wind} miles per hour."
    """
    requested_location = (location or DEFAULT_LOCATION).strip() or DEFAULT_LOCATION

    try:
        resolved_location, latitude, longitude = _resolve_location(requested_location)
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()

        current = payload.get("current") or {}
        daily = payload.get("daily") or {}
        daily_high = (daily.get("temperature_2m_max") or [None])[0]
        daily_low = (daily.get("temperature_2m_min") or [None])[0]

        temp_now = _round_int(current.get("temperature_2m"))
        high = _round_int(daily_high, default=temp_now)
        low = _round_int(daily_low, default=temp_now)
        wind_mph = _round_int(current.get("wind_speed_10m"))
        weather_code = _round_int(current.get("weather_code"), default=-1)
        condition = WEATHER_CODE_LABELS.get(weather_code, "unknown conditions")

        return (
            f"Weather {resolved_location}: {condition}, {temp_now} degrees now, "
            f"high {high}, low {low}, wind {wind_mph} miles per hour."
        )
    except ValueError:
        return f"Weather {requested_location}: location not found."
    except Exception:
        return f"Weather {requested_location}: unavailable right now."
