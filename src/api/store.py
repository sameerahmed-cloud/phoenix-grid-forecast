"""Where forecasts are saved: Postgres when DATABASE_URL is set, otherwise a local CSV file."""

import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["made_at_utc", "utility", "period_utc", "predicted_mw", "model_trained_through_utc"]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS forecast_log (
    made_at_utc               TIMESTAMPTZ      NOT NULL,
    utility                   TEXT             NOT NULL,
    period_utc                TIMESTAMPTZ      NOT NULL,
    predicted_mw              DOUBLE PRECISION NOT NULL,
    model_trained_through_utc TIMESTAMPTZ      NOT NULL,
    PRIMARY KEY (made_at_utc, utility, period_utc)
)
"""

INSERT = """
INSERT INTO forecast_log (made_at_utc, utility, period_utc, predicted_mw, model_trained_through_utc)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT DO NOTHING
"""


def _save_postgres(df: pd.DataFrame, url: str) -> None:
    import psycopg  # imported here so the CSV path and the tests don't need the database driver

    rows = [
        (r.made_at_utc.to_pydatetime(), r.utility, r.period_utc.to_pydatetime(),
         float(r.predicted_mw), r.model_trained_through_utc.to_pydatetime())
        for r in df.itertuples(index=False)
    ]
    with psycopg.connect(url, connect_timeout=10) as conn:  # commits automatically on success
        conn.execute(CREATE_TABLE)
        with conn.cursor() as cur:
            cur.executemany(INSERT, rows)


def _save_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def save_forecasts(df: pd.DataFrame, csv_path: Path) -> str:
    """Save to Postgres if configured. If that fails, keep the forecast in a CSV instead of losing it."""
    url = os.getenv("DATABASE_URL")
    if url:
        try:
            _save_postgres(df, url)
            return "postgres"
        except Exception:
            log.exception("Postgres write failed; falling back to CSV")
    _save_csv(df, csv_path)
    return "csv"