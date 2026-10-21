"""The model registry in Neon: a ledger of models, what happened to them, and which one is the champion.

Run from the project folder in the normal .venv (DATABASE_URL must be in .env):
    python -m src.retraining.registry init             create the two tables
    python -m src.retraining.registry seed [--dir models] [--version v1]
                                                       record the models in models/ as the first champion
    python -m src.retraining.registry status           show what the registry holds (never the model files)
Two tables: model_registry (one row per model, with its file, checksum and role) and retrain_runs (one row
per comparison, with the verdict). A database rule guarantees at most one champion per utility.
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

from src.api.model_store import UTILITIES, sha256_of

ROLES = ("champion", "candidate", "retired")

CREATE_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS model_registry (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    utility             TEXT NOT NULL,
    version             TEXT NOT NULL,
    role                TEXT NOT NULL CHECK (role IN ('champion', 'candidate', 'retired')),
    trained_through_utc TIMESTAMPTZ NOT NULL,
    training_rows       INTEGER NOT NULL,
    sha256              TEXT NOT NULL,
    model_file          BYTEA NOT NULL,
    metrics             JSONB NOT NULL DEFAULT '{}',
    UNIQUE (utility, version)
)""",
    ("CREATE UNIQUE INDEX IF NOT EXISTS one_champion_per_utility ON model_registry (utility) "
     "WHERE role = 'champion'"),
    """CREATE TABLE IF NOT EXISTS retrain_runs (
    id                BIGSERIAL PRIMARY KEY,
    run_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    utility           TEXT NOT NULL,
    champion_version  TEXT NOT NULL,
    holdout_start     TIMESTAMPTZ,
    holdout_end       TIMESTAMPTZ,
    holdout_hours     INTEGER,
    new_data_days     DOUBLE PRECISION,
    champion_mape     DOUBLE PRECISION,
    challenger_mape   DOUBLE PRECISION,
    baseline_mape     DOUBLE PRECISION,
    verdict           TEXT NOT NULL CHECK (verdict IN ('promote', 'keep', 'skipped')),
    reasons           JSONB NOT NULL,
    candidate_version TEXT
)""",
]

INSERT_MODEL = (
    "INSERT INTO model_registry (utility, version, role, trained_through_utc, training_rows, sha256, "
    "model_file, metrics) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)")
SELECT_MODEL = (
    "SELECT version, trained_through_utc, training_rows, sha256, model_file, metrics FROM model_registry "
    "WHERE utility = %s AND role = %s ORDER BY created_at DESC LIMIT 1")
INSERT_RUN = (
    "INSERT INTO retrain_runs (utility, champion_version, holdout_start, holdout_end, holdout_hours, "
    "new_data_days, champion_mape, challenger_mape, baseline_mape, verdict, reasons, candidate_version) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)")
RUN_FIELDS = [
    "utility", "champion_version", "holdout_start", "holdout_end", "holdout_hours", "new_data_days",
    "champion_mape", "challenger_mape", "baseline_mape", "verdict", "reasons", "candidate_version",
]


def create_tables(conn) -> None:
    for statement in CREATE_STATEMENTS:
        conn.execute(statement)


def add_model(conn, utility: str, version: str, role: str, trained_through, training_rows: int,
              model_file: bytes, metrics: dict | None = None) -> str:
    """Store one model with its checksum. Returns the checksum."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, not {role!r}")
    digest = sha256_of(model_file)
    conn.execute(INSERT_MODEL, (utility, version, role, trained_through, int(training_rows), digest,
                                model_file, json.dumps(metrics or {})))
    return digest


def get_model(conn, utility: str, role: str = "champion") -> dict | None:
    """The newest model with this role, checked against its stored checksum. None if there isn't one."""
    row = conn.execute(SELECT_MODEL, (utility, role)).fetchone()
    if row is None:
        return None
    version, trained_through, training_rows, digest, model_file, metrics = row
    model_file = bytes(model_file)
    if sha256_of(model_file) != digest:
        raise ValueError(f"{utility} {version}: the stored model file does not match its checksum")
    return {"version": version, "trained_through_utc": trained_through, "training_rows": training_rows,
            "sha256": digest, "model_file": model_file, "metrics": metrics}


def next_version(existing: list) -> str:
    """v1, v2, v3 ...: one more than the highest vN already used."""
    numbers = [int(m.group(1)) for v in existing if (m := re.fullmatch(r"v(\d+)", v))]
    return f"v{max(numbers, default=0) + 1}"


def versions_of(conn, utility: str) -> list:
    return [r[0] for r in conn.execute("SELECT version FROM model_registry WHERE utility = %s",
                                       (utility,)).fetchall()]


def promote_candidate(conn, utility: str, version: str) -> None:
    """Retire the current champion and make the candidate the champion, all or nothing."""
    with conn.transaction():
        conn.execute("UPDATE model_registry SET role = 'retired' WHERE utility = %s AND role = 'champion'",
                     (utility,))
        result = conn.execute(
            "UPDATE model_registry SET role = 'champion' "
            "WHERE utility = %s AND version = %s AND role = 'candidate'", (utility, version))
        if result.rowcount != 1:
            raise ValueError(f"{utility} {version} is not a candidate, so nothing was changed")


def log_run(conn, run: dict) -> None:
    """Record one comparison, whatever its verdict."""
    values = [json.dumps(run[f]) if f == "reasons" else run.get(f) for f in RUN_FIELDS]
    conn.execute(INSERT_RUN, values)


def seed(conn, model_dir: Path, version: str) -> list:
    """Record the models in model_dir as champion unless one already exists. Returns messages."""
    info = json.loads((model_dir / "model_info.json").read_text())
    messages = []
    for utility in UTILITIES:
        if get_model(conn, utility) is not None:
            messages.append(f"{utility}: a champion already exists, left alone")
            continue
        data = (model_dir / f"{utility}.joblib").read_bytes()
        trained_through = pd.Timestamp(info[utility]["trained_through_utc"])
        rows = info[utility]["training_rows"]
        digest = add_model(conn, utility, version, "champion", trained_through, rows, data,
                           {"source": f"seeded from {model_dir}/"})
        messages.append(f"{utility}: recorded {version} as champion (sha256 {digest[:12]})")
    return messages


def status_tables(conn):
    models = conn.execute(
        "SELECT utility, version, role, trained_through_utc, training_rows, left(sha256, 12) AS sha256, "
        "created_at FROM model_registry ORDER BY utility, id")
    runs = conn.execute(
        "SELECT run_at, utility, champion_version, holdout_hours, champion_mape, challenger_mape, verdict, "
        "candidate_version FROM retrain_runs ORDER BY id DESC LIMIT 10")
    return [pd.DataFrame(c.fetchall(), columns=[d.name for d in c.description]) for c in (models, runs)]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Create, seed or inspect the model registry in Neon.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    seed_parser = sub.add_parser("seed")
    seed_parser.add_argument("--dir", default="models")
    seed_parser.add_argument("--version", default="v1")
    sub.add_parser("status")
    args = parser.parse_args(argv)

    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Put it in .env or in the environment.")

    with psycopg.connect(url, connect_timeout=10) as conn:
        if args.command == "init":
            create_tables(conn)
            print("Tables model_registry and retrain_runs are ready.")
        elif args.command == "seed":
            create_tables(conn)
            for line in seed(conn, Path(args.dir), args.version):
                print(line)
        else:
            models, runs = status_tables(conn)
            print("Models:\n" + (models.to_string(index=False) if len(models) else "  (none yet)"))
            recent = runs.to_string(index=False) if len(runs) else "  (none yet)"
            print("\nRecent comparisons:\n" + recent)


if __name__ == "__main__":
    main()