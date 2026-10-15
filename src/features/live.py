"""Build feature rows for the next 24 hours from fresh data, reusing the TRAINING feature code."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.features.build import (WEATHER_COLS, add_calendar_features,
                                add_demand_lags, add_weather_features)
from src.ingest.eia import build_session, clean as clean_eia, fetch_all
from src.ingest.weather import HOURLY_VARS, LATITUDE, LONGITUDE
from src.processing.clean_demand import clean_one

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORY_DAYS = 10   # need >= 8 days: lag_168 plus the 168-hour average, both ending 24h before the target
HORIZON = 24


def fetch_recent_demand() -> pd.DataFrame:
    """Last ~10 days of actual demand from EIA (same fetch code as the backfill)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=HISTORY_DAYS)
    fmt = "%Y-%m-%dT%H"
    df = clean_eia(fetch_all(start.strftime(fmt), end.strftime(fmt)))
    return df[df["type"] == "D"]


def fetch_weather_forecast() -> pd.DataFrame:
    """Past 5 days + next 3 days of hourly weather, in UTC so it matches the demand clock."""
    params = {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": ",".join(HOURLY_VARS),
              "timezone": "GMT", "past_days": 5, "forecast_days": 3}
    resp = build_session().get(FORECAST_URL, params=params, timeout=60)
    resp.raise_for_status()
    w = pd.DataFrame(resp.json()["hourly"])
    w["period"] = pd.to_datetime(w.pop("time"), utc=True)
    return w


def build_live_features(utility: str, raw_demand: pd.DataFrame, weather: pd.DataFrame):
    """Return (feature rows for the next 24 hours, timestamp of the newest actual hour)."""
    s = raw_demand[raw_demand["respondent"] == utility].set_index("period")["value"].sort_index()
    clean = clean_one(s)          # the same cleaning the training data got
    last = clean.index[clean["usable"]].max()      # newest hour with a trustworthy value

    # Extend the hourly grid 24 hours past the last actual. Those future demand values stay blank;
    # the lag clues for them point back to hours that DO exist.
    grid = pd.date_range(clean.index.min(), last + timedelta(hours=HORIZON), freq="h")
    g = clean[["demand_mw"]].reindex(grid).rename_axis("period").reset_index()

    g = add_demand_lags(g)                                              # same function the tests cover
    g = g.merge(add_weather_features(weather), on="period", how="left")
    g = add_calendar_features(g)

    target = g[g["period"] > last].copy()
    if target[WEATHER_COLS].isna().any().any():
        raise ValueError("Weather forecast does not cover the forecast window")
    return target, last


if __name__ == "__main__":  # quick manual check, no API needed
    rows, last = build_live_features("AZPS", fetch_recent_demand(), fetch_weather_forecast())
    print("Newest actual hour (UTC):", last)
    print(rows[["period", "hour", "lag_24", "lag_168", "roll_mean_168", "temperature_2m", "cdh"]].to_string())