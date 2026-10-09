"""Fetch hourly electricity data from the EIA API v2 (electricity/rto/region-data)."""

import argparse
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()  

BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
PAGE_SIZE = 5000  # EIA's max rows per request
RAW_DIR = Path("data/raw")

# Verify these against the EIA docs (facets section of the route)
RESPONDENTS = ["AZPS", "SRP"]  # APS and SRP balancing authorities
TYPES = ["D", "DF"]            # D = actual demand, DF = day-ahead forecast


def get_api_key() -> str:
    key = os.getenv("EIA_API_KEY")
    if not key:
        raise RuntimeError("EIA_API_KEY not found. Check your .env file.")
    return key


def build_session() -> requests.Session:
    """A session that automatically retries on rate limits and server errors."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,  # waits 1s, 2s, 4s, ...
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_page(session: requests.Session, start: str, end: str, offset: int) -> dict:
    """Fetch one page (up to 5,000 rows) from the API."""
    params = {
        "api_key": get_api_key(),
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": RESPONDENTS,  # requests repeats the key per list item
        "facets[type][]": TYPES,
        "start": start,  # format: YYYY-MM-DDTHH
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": offset,
        "length": PAGE_SIZE,
    }
    resp = session.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all(start: str, end: str) -> pd.DataFrame:
    """Loop through pages until every row in the date range is collected."""
    session = build_session()
    rows: list[dict] = []
    offset = 0

    while True:
        payload = fetch_page(session, start, end, offset)
        data = payload["response"]["data"]
        total = int(payload["response"]["total"])  # note: API returns this as a string

        rows.extend(data)
        print(f"Fetched {len(rows):,} / {total:,} rows")

        offset += PAGE_SIZE
        if offset >= total or not data:
            break

    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning only. Heavier validation comes later in the pipeline."""
    if df.empty:
        return df
    df = df.copy()
    df["period"] = pd.to_datetime(df["period"], utc=True)  # confirm the timezone in the docs
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.drop_duplicates(subset=["period", "respondent", "type"])
    return df.sort_values(["respondent", "type", "period"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download EIA hourly data")
    parser.add_argument("--start", required=True, help="e.g. 2024-01-01T00")
    parser.add_argument("--end", required=True, help="e.g. 2024-01-07T00")
    args = parser.parse_args()

    df = clean(fetch_all(args.start, args.end))
    if df.empty:
        print("No data returned. Check dates and facet codes.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"eia_{args.start[:10]}_{args.end[:10]}.parquet"
    df.to_parquet(out, index=False)

    print(df.head())
    print(df.groupby(["respondent", "type"]).size())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()