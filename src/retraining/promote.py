"""Make a candidate model live, or roll back to an earlier one. Run from your laptop in the normal .venv.

    python -m src.retraining.promote --utility SRP --version v2         put candidate v2 live
    python -m src.retraining.promote --utility SRP --rollback-to v1     go back to the retired version v1
Needs DATABASE_URL and MODEL_BUCKET in .env, and a current `aws login`. You are shown what will change and
must type the version name to confirm (or pass --yes). The order is: copy the model to S3, move the live
pointer, then update the registry. If something fails midway, run the same command again: every step can
be repeated.
"""

import argparse
import io
import json
import os
import pickle

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv

from src.api.model_store import UTILITIES, sha256_of
from src.features.build import FEATURES
from src.retraining.publish_model import (
    S3Store,
    activate,
    describe,
    meta_key,
    model_key,
    read_pointer,
)
from src.retraining.registry import get_model, promote_candidate

LOAD_ERRORS = (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError, KeyError,
               TypeError, ValueError)
SELECT_VERSION = (
    "SELECT version, role, trained_through_utc, training_rows, sha256, model_file, metrics "
    "FROM model_registry WHERE utility = %s AND version = %s")


def get_version(conn, utility: str, version: str) -> dict | None:
    """One specific model from the registry, whatever its role, checked against its checksum."""
    row = conn.execute(SELECT_VERSION, (utility, version)).fetchone()
    if row is None:
        return None
    name, role, trained_through, training_rows, digest, model_file, metrics = row
    model_file = bytes(model_file)
    if sha256_of(model_file) != digest:
        raise ValueError(f"{utility} {name}: the stored model file does not match its checksum")
    return {"version": name, "role": role, "trained_through_utc": trained_through,
            "training_rows": training_rows, "sha256": digest, "model_file": model_file,
            "metrics": metrics or {}}


def restore_version(conn, utility: str, version: str) -> None:
    """Retire the current champion and make a previously retired model the champion, all or nothing."""
    with conn.transaction():
        conn.execute("UPDATE model_registry SET role = 'retired' WHERE utility = %s AND role = 'champion'",
                     (utility,))
        result = conn.execute(
            "UPDATE model_registry SET role = 'champion' "
            "WHERE utility = %s AND version = %s AND role = 'retired'", (utility, version))
        if result.rowcount != 1:
            raise ValueError(f"{utility} {version} is not a retired model, so nothing was changed")


def upload_model_file(store, utility: str, version: str, data: bytes, details: dict, features) -> dict:
    """Copy one model and its details to S3. A version is never overwritten with different content."""
    digest, key = sha256_of(data), model_key(utility, version)
    if not store.exists(key):
        store.put(key, data)
    elif sha256_of(store.get(key)) != digest:
        raise ValueError(f"{utility} {version} already exists in S3 with different content")
    meta = {**details, "version": version, "key": key, "sha256": digest, "features": list(features)}
    store.put(meta_key(utility, version), json.dumps(meta, indent=2).encode())
    return meta


def check_loadable(model_file: bytes, features) -> None:
    """Refuse a file the API could not use: it must unpickle and expect the features the API builds."""
    try:
        model = joblib.load(io.BytesIO(model_file))
    except LOAD_ERRORS as e:
        raise ValueError(f"the candidate file cannot be loaded: {e}") from e
    width = getattr(model, "n_features_in_", None)
    if width is not None and width != len(features):
        raise ValueError(f"the candidate expects {width} features but the API builds {len(features)}")


def confirm_by_typing(lines: list, expected: str, input_fn=input) -> bool:
    """Show what will change, then require the version name to be typed exactly."""
    print("\n".join(lines))
    return input_fn(f"Type {expected} to confirm, anything else cancels: ").strip() == expected


def auto_confirm(lines: list, expected: str) -> bool:
    print("\n".join(lines))
    return True


def describe_promotion(utility: str, candidate: dict, champion: dict) -> list:
    m = candidate["metrics"]
    lines = [f"{utility}: put {candidate['version']} live in place of {champion['version']}.",
             (f"  candidate trained through {candidate['trained_through_utc']} on "
              f"{candidate['training_rows']} rows")]
    if "challenger_mape" in m:
        lines.append(f"  holdout {m.get('holdout_start', '?')} to {m.get('holdout_end', '?')}: candidate "
                     f"{m['challenger_mape']:.2f}% vs champion {m['champion_mape']:.2f}% "
                     f"vs copy-yesterday {m['baseline_mape']:.2f}%")
    lines.append(f"  to undo later: python -m src.retraining.promote --utility {utility} "
                 f"--rollback-to {champion['version']}")
    return lines


def promote(conn, store, utility: str, version: str, features, confirm_fn, force: bool = False) -> list:
    """Make a registry candidate the live model. Returns the lines to show afterwards."""
    candidate = get_version(conn, utility, version)
    if candidate is None:
        raise ValueError(f"{utility} {version} is not in the registry")
    if candidate["role"] != "candidate":
        raise ValueError(f"{utility} {version} is a {candidate['role']}, not a candidate")
    champion = get_model(conn, utility, "champion")
    if champion is None:
        raise ValueError(f"{utility} has no champion in the registry; run the registry seed command first")

    # What S3 serves must be what the registry calls the champion (or the candidate, if an earlier run
    # stopped halfway). Names can differ, so compare the files by checksum.
    live = read_pointer(store).get(utility, {}).get("sha256")
    if live not in (champion["sha256"], candidate["sha256"]) and not force:
        raise ValueError(f"S3 is serving a model the registry does not call the champion for {utility}. "
                         "Find out why, or use --force if you are sure")
    check_loadable(candidate["model_file"], features)

    if not confirm_fn(describe_promotion(utility, candidate, champion), version):
        return ["Cancelled. Nothing was changed."]
    details = {"trained_through_utc": str(pd.Timestamp(candidate["trained_through_utc"])),
               "training_rows": candidate["training_rows"], "target": "demand_mw (level)"}
    meta = upload_model_file(store, utility, version, candidate["model_file"], details, features)
    pointer = activate(store, version, only=[utility])
    promote_candidate(conn, utility, version)
    return [f"Uploaded {utility} {version} to S3 (sha256 {meta['sha256'][:12]}).",
            *(f"Live in S3 now: {line}" for line in describe(pointer)),
            f"Registry updated: {utility} {version} is the champion, {champion['version']} is retired.",
            ("The API switches on its next cold start: after about 15 minutes without requests, or at the "
             "next scheduled collector run. Check /model-info afterwards.")]


def rollback(conn, store, utility: str, version: str, confirm_fn) -> list:
    """Go back to a model that was live before. Returns the lines to show afterwards."""
    target = get_version(conn, utility, version)
    if target is None:
        raise ValueError(f"{utility} {version} is not in the registry")
    if target["role"] != "retired":
        raise ValueError(f"{utility} {version} is a {target['role']}; only a retired model can be restored")
    champion = get_model(conn, utility, "champion")
    lines = [f"{utility}: roll back from {champion['version'] if champion else 'nothing'} to {version}."]
    if not confirm_fn(lines, version):
        return ["Cancelled. Nothing was changed."]
    pointer = activate(store, version, only=[utility])
    restore_version(conn, utility, version)
    return [*(f"Live in S3 now: {line}" for line in describe(pointer)),
            f"Registry updated: {utility} {version} is the champion again.",
            "The API switches on its next cold start."]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Make a candidate model live, or roll back.")
    parser.add_argument("--utility", required=True, choices=list(UTILITIES))
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--version", help="the candidate to put live")
    which.add_argument("--rollback-to", help="a retired version to restore")
    parser.add_argument("--yes", action="store_true", help="skip the typed confirmation")
    parser.add_argument("--force", action="store_true", help="promote even if S3 and the registry disagree")
    parser.add_argument("--bucket", help="bucket name; defaults to MODEL_BUCKET")
    args = parser.parse_args(argv)

    load_dotenv()
    url, bucket = os.getenv("DATABASE_URL"), args.bucket or os.getenv("MODEL_BUCKET")
    if not url or not bucket:
        raise SystemExit("DATABASE_URL and MODEL_BUCKET must both be set (in .env or the environment).")
    confirm = auto_confirm if args.yes else confirm_by_typing
    store = S3Store(bucket)
    with psycopg.connect(url, connect_timeout=10) as conn:
        if args.version:
            lines = promote(conn, store, args.utility, args.version, FEATURES, confirm, args.force)
        else:
            lines = rollback(conn, store, args.utility, args.rollback_to, confirm)
    print("\n".join(lines))


if __name__ == "__main__":
    main()