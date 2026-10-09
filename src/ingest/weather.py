"""Fetch hourly historical weather for Phoenix from the Open-Meteo archive API."""

import argparse
from pathlib import Path

import pandas as pd

from src.ingest.eia import build_session  # reuse the retrying session instead of rewriting it

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
RAW_DIR = Path("data/raw")

# Phoenix, AZ. One point is a simplification: both utilities cover the metro area.
LATITUDE, LONGITUDE = 33.45, -112.07

# Verify names at open-meteo.com/en/docs/historical-weather-api
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "shortwave_radiation",  # sunlight: matters for rooftop solar and AC load
    "cloud_cover",
    "wind_speed_10m",
]


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """One request covers the whole range, since weather is a small payload."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start,  # format: YYYY-MM-DD
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "GMT",  # must match EIA's UTC timestamps so the two datasets join cleanly
    }
    resp = build_session().get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]  # dict of equal-length lists: time + one list per variable
    return pd.DataFrame(hourly)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # rename to "period" and make it timezone-aware UTC so it matches the EIA column exactly
    df["period"] = pd.to_datetime(df.pop("time"), utc=True)
    # the archive lags a few days, so the newest hours come back null: drop them, don't keep NaN rows
    df = df.dropna(subset=["temperature_2m"])
    return df.drop_duplicates(subset=["period"]).sort_values("period").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Open-Meteo hourly weather")
    parser.add_argument("--start", required=True, help="e.g. 2022-01-01")
    parser.add_argument("--end", required=True, help="e.g. 2026-10-08")
    args = parser.parse_args()

    df = clean(fetch_weather(args.start, args.end))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"weather_{args.start}_{args.end}.parquet"
    df.to_parquet(out, index=False)

    print(df.head())
    print(f"{len(df):,} rows, {df['period'].min()} to {df['period'].max()}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()