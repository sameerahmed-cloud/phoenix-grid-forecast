"""Score the live forecast log against what EIA later published.

Run from the project folder:  python -m src.models.run_live_scoring [--save]
Needs DATABASE_URL and EIA_API_KEY in .env (or in the environment).
With --save, the graded rows are also written to the forecast_scores table in Neon.
The grading rules live in score_live.py; this file only fetches the inputs and prints the report.
"""

import argparse
import os

import pandas as pd
import psycopg
from dotenv import load_dotenv

from src.ingest.eia import clean as clean_eia
from src.ingest.eia import fetch_all
from src.models.score_live import MATURITY_HOURS, score_all, to_score_records
from src.processing.clean_demand import clean_one

HOUR_FMT = "%Y-%m-%dT%H"
PAD_DAYS = 3  # extra history before the first target hour: copy-yesterday and cleaning need context
LOG_SQL = ("SELECT made_at_utc, utility, period_utc, predicted_mw, model_trained_through_utc "
           "FROM forecast_log")

CREATE_SCORES = """
CREATE TABLE IF NOT EXISTS forecast_scores (
    made_at_utc               TIMESTAMPTZ      NOT NULL,
    utility                   TEXT             NOT NULL,
    period_utc                TIMESTAMPTZ      NOT NULL,
    model_trained_through_utc TIMESTAMPTZ      NOT NULL,
    lead_h                    DOUBLE PRECISION NOT NULL,
    bucket                    TEXT             NOT NULL,
    predicted_mw              DOUBLE PRECISION NOT NULL,
    actual_mw                 DOUBLE PRECISION NOT NULL,
    naive_24h                 DOUBLE PRECISION NOT NULL,
    scored_at_utc             TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (made_at_utc, utility, period_utc)
)
"""

# Rerunning is safe: a row that already exists is refreshed (EIA may have revised the actual), not duplicated.
UPSERT_SCORE = """
INSERT INTO forecast_scores (made_at_utc, utility, period_utc, model_trained_through_utc,
                             lead_h, bucket, predicted_mw, actual_mw, naive_24h)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (made_at_utc, utility, period_utc) DO UPDATE SET
    lead_h = EXCLUDED.lead_h, bucket = EXCLUDED.bucket, predicted_mw = EXCLUDED.predicted_mw,
    actual_mw = EXCLUDED.actual_mw, naive_24h = EXCLUDED.naive_24h, scored_at_utc = now()
"""


def load_log(url: str) -> pd.DataFrame:
    """Every forecast we ever saved, straight from Neon."""
    with psycopg.connect(url, connect_timeout=10) as conn, conn.cursor() as cur:
        cur.execute(LOG_SQL)
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def load_actuals(first_target: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    """Actual demand from EIA, cleaned with the SAME rules the training data got."""
    start = first_target - pd.Timedelta(days=PAD_DAYS)
    raw = clean_eia(fetch_all(start.strftime(HOUR_FMT), now.strftime(HOUR_FMT)))
    raw = raw[raw["type"] == "D"]
    frames = []
    for utility, g in raw.groupby("respondent"):
        series = g.set_index("period")["value"].sort_index()
        cleaned = clean_one(series).rename_axis("period_utc").reset_index()
        frames.append(cleaned.assign(utility=utility)[["utility", "period_utc", "demand_mw", "usable"]])
    return pd.concat(frames, ignore_index=True)


def save_scores(url: str, scored: pd.DataFrame) -> int:
    """Write the graded rows to Neon, creating the table the first time. Returns how many were sent."""
    records = to_score_records(scored)
    with psycopg.connect(url, connect_timeout=10) as conn:  # commits automatically on success
        conn.execute(CREATE_SCORES)
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SCORE, records)
    return len(records)


def main(save: bool = False) -> None:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Put it in .env or in the environment.")

    now = pd.Timestamp.now(tz="UTC")
    log = load_log(url)
    targets = pd.to_datetime(log["period_utc"], utc=True)
    print(f"Loaded {len(log)} logged predictions (24 per run per utility).")

    mature = targets[targets <= now - pd.Timedelta(hours=MATURITY_HOURS)]
    if mature.empty:
        print(f"Nothing is {MATURITY_HOURS} hours old yet, so there is nothing to score. Try again later.")
        return

    actuals = load_actuals(mature.min(), now)
    scored, funnel, table = score_all(log, actuals, now)

    print("\nWhat happened to the rows (each step can only remove rows):")
    for step, count in funnel.items():
        print(f"  {step:>20}: {count}")
    if table.empty:
        print("\nNo hours could be scored yet.")
        return
    print("\nScoreboard (lower is better; 'naive' = copy yesterday):")
    print(table.round(2).to_string(index=False))

    if save:
        print(f"\nSaved {save_scores(url, scored)} graded rows to forecast_scores.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score logged forecasts against actual demand.")
    parser.add_argument("--save", action="store_true",
                        help="also write the graded rows to the forecast_scores table in Neon")
    main(save=parser.parse_args().save)