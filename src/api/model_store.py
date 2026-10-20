"""Where the API gets its models.

The active models live in a private S3 bucket. A small pointer file says which version is live for
each utility, and every model file is checked against the checksum recorded in the pointer. If MODEL_BUCKET
is not set, or anything about S3 goes wrong, the API falls back to the copy baked into the image, so a bad
upload can never take the service down.

Bucket layout:
    active.json                         the pointer: live version, checksum and details for each utility
    models/<utility>/<version>.joblib   one model file per utility and version
    models/<utility>/<version>.json     the details recorded when that version was uploaded
"""

import hashlib
import io
import json
import logging
import pickle
from pathlib import Path

import joblib

log = logging.getLogger(__name__)

POINTER_KEY = "active.json"
UTILITIES = ("AZPS", "SRP")


class ModelLoadError(Exception):
    """The active models in S3 could not be loaded safely."""


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def s3_fetcher(bucket: str):
    """Return fetch(key) -> bytes for a private bucket. boto3 is imported here so tests don't need it."""
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as e:
        raise ModelLoadError(f"boto3 is not installed: {e}") from e

    # Short timeouts: a slow S3 must not hold up a cold start when the image already has a working copy.
    config = Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 2})
    client = boto3.client("s3", config=config)

    def fetch(key: str) -> bytes:
        try:
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except (BotoCoreError, ClientError) as e:
            raise ModelLoadError(f"could not read s3://{bucket}/{key}: {e}") from e

    return fetch


def load_from_store(fetch, expected_features) -> tuple[dict, dict]:
    """Read the pointer, then each model it names. Returns (models, info). Raises ModelLoadError."""
    try:
        info = json.loads(fetch(POINTER_KEY))
    except ValueError as e:
        raise ModelLoadError(f"{POINTER_KEY} is not valid JSON: {e}") from e
    if not isinstance(info, dict) or info.get("features") != list(expected_features):
        raise ModelLoadError("the feature list in the pointer does not match the one this API builds")

    models = {}
    for utility in UTILITIES:
        entry = info.get(utility)
        if not isinstance(entry, dict) or not {"key", "sha256", "version"} <= entry.keys():
            raise ModelLoadError(f"the pointer has no usable entry for {utility}")
        data = fetch(entry["key"])
        if sha256_of(data) != entry["sha256"]:
            raise ModelLoadError(
                f"checksum mismatch for {utility} {entry['version']}: the file is damaged or was replaced")
        try:
            models[utility] = joblib.load(io.BytesIO(data))
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError, KeyError,
                TypeError, ValueError) as e:
            raise ModelLoadError(f"could not read the {utility} {entry['version']} model file: {e}") from e
    return models, info


def load_models(model_dir: Path, features, bucket: str | None = None, make_fetch=s3_fetcher):
    """Active models from S3 when a bucket is set; otherwise, or on any failure, the copy in the image."""
    if bucket:
        try:
            models, info = load_from_store(make_fetch(bucket), features)
        except ModelLoadError as e:
            log.warning("Falling back to the models baked into the image: %s", e)
        except Exception:  # last resort: whatever goes wrong, the copy in the image must still start
            log.exception("Unexpected error loading models from S3; falling back to the image copy")
        else:
            info["source"] = f"s3://{bucket}"
            live = ", ".join(f"{u} {info[u]['version']}" for u in UTILITIES)
            print(f"Loaded models from s3://{bucket}: {live}", flush=True)
            return models, info

    models = {u: joblib.load(model_dir / f"{u}.joblib") for u in UTILITIES}
    info = json.loads((model_dir / "model_info.json").read_text())
    info["source"] = "image"
    return models, info