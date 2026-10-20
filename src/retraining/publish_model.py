"""Publish models to S3 and choose which version is live.

Run from your laptop in the normal .venv, after `aws login`. The bucket comes from --bucket or MODEL_BUCKET.
    python -m src.retraining.publish_model upload --version v1 [--dir models]    copy the model files to S3
    python -m src.retraining.publish_model activate --version v1 [--utility SRP]  make that version live
    python -m src.retraining.publish_model status                                 show what is live now
A version is never overwritten, and the pointer is written last, so the API never sees a half-finished
upload. Rolling back means activating an older version.
"""

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src.api.model_store import POINTER_KEY, UTILITIES, sha256_of

VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def model_key(utility: str, version: str) -> str:
    return f"models/{utility}/{version}.joblib"


def meta_key(utility: str, version: str) -> str:
    return f"models/{utility}/{version}.json"


def check_version(version: str) -> None:
    if not VERSION_PATTERN.match(version):
        raise ValueError(
            f"version names may only use letters, digits, dots, dashes and underscores: {version!r}")


class S3Store:
    """The few bucket operations this tool needs. boto3 is imported here so the tests don't need it."""

    def __init__(self, bucket: str):
        import boto3

        self.bucket = bucket
        self.client = boto3.client("s3")

    def get(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True


def read_pointer(store) -> dict:
    return json.loads(store.get(POINTER_KEY)) if store.exists(POINTER_KEY) else {}


def upload_version(store, model_dir: Path, version: str) -> dict:
    """Copy each utility's model file and its details to the bucket. Does NOT make the version live."""
    check_version(version)
    info = json.loads((model_dir / "model_info.json").read_text())
    published = {}
    for utility in UTILITIES:
        data = (model_dir / f"{utility}.joblib").read_bytes()
        digest, key = sha256_of(data), model_key(utility, version)
        if not store.exists(key):
            store.put(key, data)
        elif sha256_of(store.get(key)) != digest:
            raise ValueError(f"{utility} version {version} already exists with different content; "
                             "choose a new version name")
        meta = {**info[utility], "version": version, "key": key, "sha256": digest,
                "features": info["features"]}
        store.put(meta_key(utility, version), json.dumps(meta, indent=2).encode())
        published[utility] = meta
    return published


def activate(store, version: str, only: list | None = None) -> dict:
    """Make `version` live for all utilities, or only those listed. The pointer is written last."""
    check_version(version)
    chosen = list(only) if only else list(UTILITIES)
    unknown = set(chosen) - set(UTILITIES)
    if unknown:
        raise ValueError(f"unknown utility: {sorted(unknown)}")

    current = read_pointer(store)
    entries = {}
    for utility in UTILITIES:
        if utility in chosen:
            if not store.exists(meta_key(utility, version)):
                raise ValueError(f"{utility} version {version} was never uploaded")
            entries[utility] = json.loads(store.get(meta_key(utility, version)))
        elif utility in current:
            entries[utility] = current[utility]
        else:
            raise ValueError(f"{utility} has no live version yet, so activate it too")

    features = entries[chosen[0]]["features"]
    if any(entries[u]["features"] != features for u in chosen):
        raise ValueError("the chosen models were trained with different feature lists")
    if len(chosen) < len(UTILITIES) and current.get("features") != features:
        raise ValueError(
            "a different feature list would break the utility you are not changing; activate both")

    pointer = {u: {k: v for k, v in entries[u].items() if k != "features"} for u in UTILITIES}
    pointer["features"] = features
    store.put(POINTER_KEY, json.dumps(pointer, indent=2).encode())
    return pointer


def describe(pointer: dict) -> list:
    if not pointer:
        return ["Nothing is live in S3 yet; the API uses the models baked into its image."]
    return [f"{u}: version {pointer[u]['version']}, trained through {pointer[u]['trained_through_utc']}, "
            f"{pointer[u]['training_rows']} rows, sha256 {pointer[u]['sha256'][:12]}"
            for u in UTILITIES]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Publish models to S3 and choose which version is live.")
    parser.add_argument("--bucket", help="bucket name; defaults to the MODEL_BUCKET setting")
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upload", help="copy the model files to the bucket")
    up.add_argument("--version", required=True)
    up.add_argument("--dir", default="models", help="folder with the .joblib files and model_info.json")
    act = sub.add_parser("activate", help="make a version the live one")
    act.add_argument("--version", required=True)
    act.add_argument("--utility", action="append", choices=list(UTILITIES), help="only this utility")
    sub.add_parser("status", help="show what is live")
    args = parser.parse_args(argv)

    load_dotenv()
    bucket = args.bucket or os.getenv("MODEL_BUCKET")
    if not bucket:
        raise SystemExit("Set MODEL_BUCKET in .env or the environment, or pass --bucket.")
    store = S3Store(bucket)

    if args.command == "upload":
        for utility, meta in upload_version(store, Path(args.dir), args.version).items():
            print(f"Uploaded {utility} {args.version}: {meta['key']} (sha256 {meta['sha256'][:12]})")
        print(f"Not live yet. Run: python -m src.retraining.publish_model activate --version {args.version}")
    elif args.command == "activate":
        pointer = activate(store, args.version, args.utility)
        print("Live now:")
    else:
        pointer = read_pointer(store)
        print("Live now:")
    if args.command != "upload":
        for line in describe(pointer):
            print(f"  {line}")


if __name__ == "__main__":
    main()