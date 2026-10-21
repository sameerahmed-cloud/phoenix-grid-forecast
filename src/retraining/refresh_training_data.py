"""Keep the training table in Neon growing: refetch the last few weeks, rebuild the features, upsert.

Run from the project folder (DATABASE_URL and EIA_API_KEY must be set):
    python -m src.retraining.refresh_training_data [--days 30] [--dry-run]
It uses the same cleaning and feature code as the original pipeline, and the same input checks as the
API, so a broken delivery from EIA or Open-Meteo stops the job instead of reaching the training table.
Rewriting the last few weeks each time is safe: EIA revises recent values, and the upsert just refreshes
them. Only hours that already have weather are stored, because the weather archive trails real time by a
few days.
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import psycopg
from dotenv import load_dotenv

from src.features.build import FEATURES, build_features
from src.features.validate_live import validate_demand, validate_weather
from src.ingest.eia import clean as clean_eia
from src.ingest.eia import fetch_all
from src.ingest.weather import clean as clean_weather
from src.ingest.weather import fetch_weather
from src.processing.clean_demand import clean_one
from src.retraining.training_data import upsert_rows

DEFAULT_DAYS = 30        # 8 days go to context for the lag clues, leaving about 3 weeks of rows
WEATHER_WARMUP_DAYS = 4  # the 3-day average temperature needs 72 hours of weather first
MIN_NEW_ROWS = 48        # fewer than two days of rows means something went wrong upstream
EIA_FORMAT = "%Y-%m-%dT%H"


def clean_demand_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """The same per-utility cleaning that clean_demand.py applies to the full history."""
    parts = []
    for utility, g in raw[raw["type"] == "D"].groupby("respondent"):
        cleaned = clean_one(g.set_index("period")["value"].sort_index())
        cleaned["respondent"] = utility
        parts.append(cleaned.rename_axis("period").reset_index())
    return pd.concat(parts, ignore_index=True)


def new_training_rows(raw_demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Check both fetches, then build the rows that are safe to store. Raises on a junk delivery."""
    demand_only = raw_demand[raw_demand["type"] == "D"]
    for utility in sorted(demand_only["respondent"].unique()):
        validate_demand(demand_only, utility)
    validate_weather(weather)
    table = build_features(clean_demand_frame(raw_demand), weather)
    return table.dropna(subset=FEATURES).reset_index(drop=True)  # drops hours that have no weather yet


def fetch_inputs(now: datetime, days: int):
    """Raw EIA demand and archive weather for the trailing window."""
    start = now - timedelta(days=days)
    raw = clean_eia(fetch_all(start.strftime(EIA_FORMAT), now.strftime(EIA_FORMAT)))
    weather_start = (start - timedelta(days=WEATHER_WARMUP_DAYS)).strftime("%Y-%m-%d")
    return raw, clean_weather(fetch_weather(weather_start, now.strftime("%Y-%m-%d")))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Refresh the training table in Neon.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="how far back to refetch")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the rows and report, but write nothing")
    args = parser.parse_args(argv)

    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Put it in .env or in the environment.")

    raw, weather = fetch_inputs(datetime.now(timezone.utc), args.days)
    rows = new_training_rows(raw, weather)
    if len(rows) < MIN_NEW_ROWS:
        raise SystemExit(f"Only {len(rows)} rows were built, far fewer than expected. Nothing was written.")

    for utility, g in rows.groupby("respondent"):
        first, last = g["period"].min(), g["period"].max()
        print(f"{utility}: {len(g):,} rows, {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC")
    if args.dry_run:
        print("Dry run: nothing was written.")
        return
    with psycopg.connect(url, connect_timeout=10) as conn:
        print(f"Upserted {upsert_rows(conn, rows):,} rows into the training table.")


if __name__ == "__main__":
    main()