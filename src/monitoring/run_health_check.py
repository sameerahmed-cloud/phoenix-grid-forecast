"""Run the weekly health check against the live scores.

Run from the project folder:  python -m src.monitoring.run_health_check
Needs DATABASE_URL in .env (or in the environment). Exits with code 1 when a check fails, so the
GitHub run turns red and emails you. On GitHub the report also appears on the run's summary page.
The checks themselves live in health.py; this file only fetches the scores and prints the report.
"""

import os

import pandas as pd
import psycopg
from dotenv import load_dotenv

from src.monitoring.health import evaluate, has_failures, to_markdown

SCORES_SQL = ("SELECT utility, bucket, period_utc, predicted_mw, actual_mw, naive_24h "
              "FROM forecast_scores WHERE period_utc > now() - interval '40 days'")


def load_scores(url: str) -> pd.DataFrame:
    """The graded forecasts from the last 40 days, straight from Neon."""
    with psycopg.connect(url, connect_timeout=10) as conn, conn.cursor() as cur:
        cur.execute(SCORES_SQL)
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def write_summary(markdown: str, path: str | None = None) -> None:
    """Add the report to the GitHub run's summary page. Does nothing when run on your own computer."""
    path = path or os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")


def main() -> None:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Put it in .env or in the environment.")

    now = pd.Timestamp.now(tz="UTC")
    results = evaluate(load_scores(url), now)
    report = to_markdown(results, now)
    print(report)
    write_summary(report)
    if has_failures(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()