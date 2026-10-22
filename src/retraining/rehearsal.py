"""Helpers for rehearsing the retraining loop on a throwaway copy of the system.

    python -m src.retraining.rehearsal make-weak [--before 2026-09-01] [--dir rehearsal_models]
    python -m src.retraining.rehearsal reset
The rehearsal needs its own Neon branch and its own S3 bucket. `make-weak` trains a deliberately poor model
to play the champion, so the real comparison has something to beat. `reset` empties the registry, so it
refuses to run unless DATABASE_URL and MODEL_BUCKET were set in THIS terminal session to values that differ
from the ones in your .env (your live system), and the bucket name contains 'rehearsal'.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import psycopg
from dotenv import dotenv_values, load_dotenv

from src.api.model_store import UTILITIES
from src.features.build import FEATURES
from src.retraining.registry import create_tables
from src.retraining.training_data import load_table

# Few, shallow trees: this model is meant to be bad.
WEAK_PARAMS = {"n_estimators": 10, "num_leaves": 4, "learning_rate": 0.05, "min_child_samples": 50,
               "random_state": 42, "verbose": -1}


def check_rehearsal_targets(shell: dict, live: dict) -> None:
    """Raise unless the shell points at a rehearsal database and bucket, never at the live ones."""
    url, bucket = shell.get("DATABASE_URL"), shell.get("MODEL_BUCKET")
    if not live.get("DATABASE_URL"):
        raise ValueError("cannot find your live DATABASE_URL in .env, so the rehearsal database "
                         "cannot be told apart from it")
    if not url or url == live["DATABASE_URL"]:
        raise ValueError("DATABASE_URL must be set in THIS terminal to the rehearsal branch's connection "
                         "string. It is missing, or it is the live database from your .env")
    if not bucket or bucket == live.get("MODEL_BUCKET") or "rehearsal" not in bucket:
        raise ValueError("MODEL_BUCKET must be set in THIS terminal to a separate bucket whose name "
                         "contains 'rehearsal'. It is missing, or it is the live bucket")


def train_weak(rows: pd.DataFrame, target: pd.Series):
    import lightgbm as lgb  # imported here so the rest of this file needs no LightGBM

    model = lgb.LGBMRegressor(**WEAK_PARAMS)
    model.fit(rows, target)
    return model


def make_weak_champions(table: pd.DataFrame, before, features, train_fn=train_weak):
    """One deliberately weak model per utility, trained only on data before `before`."""
    models, info = {}, {}
    for utility in UTILITIES:
        g = table[table["respondent"] == utility].dropna(subset=[*features, "demand_mw"])
        train = g[g["period"] < pd.Timestamp(before)]
        if train.empty:
            raise ValueError(f"{utility}: there is no training data before {before}")
        models[utility] = train_fn(train[features], train["demand_mw"])
        info[utility] = {"trained_through_utc": str(train["period"].max()), "training_rows": len(train),
                         "target": "demand_mw (level)"}
    return models, info


def write_model_dir(folder: Path, models: dict, info: dict, features) -> None:
    """Write the same files train_final.py writes, so the normal seed and upload commands accept them."""
    folder.mkdir(parents=True, exist_ok=True)
    for utility, model in models.items():
        joblib.dump(model, folder / f"{utility}.joblib")
    (folder / "model_info.json").write_text(json.dumps({**info, "features": list(features)}, indent=2))


def reset_registry(conn) -> None:
    """Empty the registry tables, creating them first if the database has never had them."""
    create_tables(conn)
    conn.execute("DELETE FROM retrain_runs")
    conn.execute("DELETE FROM model_registry")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Helpers for the rehearsal.")
    sub = parser.add_subparsers(dest="command", required=True)
    weak = sub.add_parser("make-weak")
    weak.add_argument("--before", default="2026-09-01", help="train only on data before this date")
    weak.add_argument("--dir", default="rehearsal_models")
    sub.add_parser("reset")
    args = parser.parse_args(argv)

    if args.command == "reset":
        # read the shell's values BEFORE anything loads .env
        shell = {name: os.environ.get(name) for name in ("DATABASE_URL", "MODEL_BUCKET")}
        try:
            check_rehearsal_targets(shell, dotenv_values(".env"))
        except ValueError as e:
            raise SystemExit(f"Refusing to reset: {e}") from e
        with psycopg.connect(shell["DATABASE_URL"], connect_timeout=10) as conn:
            reset_registry(conn)
        print("Rehearsal registry is empty. (Bucket in use: " + shell["MODEL_BUCKET"] + ")")
        return

    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set.")
    table = load_table(url)
    models, info = make_weak_champions(table, pd.Timestamp(args.before, tz="UTC"), FEATURES)
    write_model_dir(Path(args.dir), models, info, FEATURES)
    for utility in UTILITIES:
        print(f"{utility}: weak model trained through {info[utility]['trained_through_utc']} "
              f"on {info[utility]['training_rows']} rows, saved in {args.dir}/")


if __name__ == "__main__":
    main()