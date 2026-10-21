"""The training table in Neon: the same rows as data/processed/features.parquet, reachable from GitHub.

Run from the project folder in the normal .venv (DATABASE_URL must be in .env):
    python -m src.retraining.training_data load  [--parquet data/processed/features.parquet]
    python -m src.retraining.training_data check [--parquet data/processed/features.parquet]
`load` creates the table if needed and upserts every row, so running it twice changes nothing.
`check` reads the table back and compares it with the parquet file, and says exactly what differs.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv

from src.features.build import FEATURES

TABLE = "training_features"
KEY = ["period", "respondent"]
COLUMNS = ["period", "respondent", "demand_mw", "was_repaired", *FEATURES]
INT_COLUMNS = {"hour", "dayofweek", "month", "is_weekend", "is_holiday"}
BATCH_SIZE = 5000


def column_sql(name: str) -> str:
    fixed = {"period": "period TIMESTAMPTZ NOT NULL", "respondent": "respondent TEXT NOT NULL",
             "demand_mw": "demand_mw DOUBLE PRECISION NOT NULL",
             "was_repaired": "was_repaired BOOLEAN NOT NULL"}
    if name in fixed:
        return fixed[name]
    return f"{name} SMALLINT" if name in INT_COLUMNS else f"{name} DOUBLE PRECISION"


def create_sql() -> str:
    columns = ",\n    ".join(column_sql(c) for c in COLUMNS)
    return f"CREATE TABLE IF NOT EXISTS {TABLE} (\n    {columns},\n    PRIMARY KEY ({', '.join(KEY)})\n)"


def upsert_sql() -> str:
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS if c not in KEY)
    return (f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) VALUES ({', '.join(['%s'] * len(COLUMNS))}) "
            f"ON CONFLICT ({', '.join(KEY)}) DO UPDATE SET {updates}")


def to_records(df: pd.DataFrame) -> list:
    """Plain Python values in table order. Missing feature values become NULL."""
    if df["period"].dt.tz is None:
        raise ValueError("period must be timezone-aware (UTC)")
    if df["demand_mw"].isna().any():
        raise ValueError("demand_mw has missing values; the training table only holds rows with a target")
    records = []
    for row in df[COLUMNS].itertuples(index=False, name=None):
        record = []
        for column, value in zip(COLUMNS, row, strict=True):
            if column == "period":
                record.append(pd.Timestamp(value).to_pydatetime())
            elif column == "respondent":
                record.append(str(value))
            elif column == "was_repaired":
                record.append(bool(value))
            elif pd.isna(value):
                record.append(None)
            else:
                record.append(int(value) if column in INT_COLUMNS else float(value))
        records.append(tuple(record))
    return records


def upsert_rows(conn, df: pd.DataFrame, batch_size: int = BATCH_SIZE) -> int:
    """Create the table if needed, then upsert every row in batches. Returns how many rows were sent."""
    records = to_records(df)
    conn.execute(create_sql())
    with conn.cursor() as cur:
        for start in range(0, len(records), batch_size):
            cur.executemany(upsert_sql(), records[start:start + batch_size])
    return len(records)


def load_table(url: str, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """The training table from Neon, optionally only rows at or after `since`."""
    sql, params = f"SELECT {', '.join(COLUMNS)} FROM {TABLE}", ()
    if since is not None:
        sql, params = sql + " WHERE period >= %s", (since.to_pydatetime(),)
    with psycopg.connect(url, connect_timeout=10) as conn, conn.cursor() as cur:
        cur.execute(sql + " ORDER BY respondent, period", params)
        df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    df["period"] = pd.to_datetime(df["period"], utc=True)
    return df


def verify(neon: pd.DataFrame, parquet: pd.DataFrame) -> list:
    """Compare the two tables. Returns a plain-words line per difference; an empty list means identical."""
    problems = []
    for utility in sorted(set(neon["respondent"]) | set(parquet["respondent"])):
        a, b = neon[neon["respondent"] == utility], parquet[parquet["respondent"] == utility]
        if len(a) != len(b):
            problems.append(f"{utility}: {len(a)} rows in Neon but {len(b)} in the parquet file")
            continue
        if len(a) and (a["period"].min() != b["period"].min() or a["period"].max() != b["period"].max()):
            problems.append(f"{utility}: the time range differs")
        for column in COLUMNS[2:]:
            if not np.isclose(a[column].astype(float).sum(), b[column].astype(float).sum(), rtol=1e-9):
                problems.append(f"{utility}: the total of {column} differs")
    return problems


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Load or check the training table in Neon.")
    parser.add_argument("command", choices=["load", "check"])
    parser.add_argument("--parquet", default="data/processed/features.parquet")
    args = parser.parse_args(argv)

    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Put it in .env or in the environment.")
    parquet = pd.read_parquet(Path(args.parquet))[COLUMNS]

    if args.command == "load":
        with psycopg.connect(url, connect_timeout=10) as conn:
            sent = upsert_rows(conn, parquet)
        print(f"Sent {sent:,} rows to {TABLE}. Now run the same command with `check` to confirm.")
        return

    problems = verify(load_table(url), parquet)
    if problems:
        print("Neon does NOT match the parquet file:")
        for line in problems:
            print(f"  {line}")
        raise SystemExit(1)
    print(f"Neon matches the parquet file: {len(parquet):,} rows, every column total identical.")


if __name__ == "__main__":
    main()