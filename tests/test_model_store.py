import io
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.api.model_store import (
    POINTER_KEY,
    ModelLoadError,
    load_from_store,
    load_models,
    sha256_of,
)

FEATURES = ["lag_24", "temperature_2m"]


class Level:
    """A stand-in model that always predicts one number. It must be importable so joblib can reload it."""

    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        return np.full(len(rows), self.value)


def dump(model) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def make_bucket(version="v2", level=5000.0):
    """A fake bucket (a dict) holding a valid pointer and both model files."""
    files = {}
    pointer = {"features": FEATURES}
    for utility in ["AZPS", "SRP"]:
        data = dump(Level(level))
        key = f"models/{utility}/{version}.joblib"
        files[key] = data
        pointer[utility] = {"version": version, "key": key, "sha256": sha256_of(data),
                            "trained_through_utc": "2026-10-08 00:00:00+00:00", "training_rows": 1000}
    files[POINTER_KEY] = json.dumps(pointer).encode()
    return files


def fetcher(files):
    def fetch(key):
        if key not in files:
            raise ModelLoadError(f"could not read {key}: not found")
        return files[key]
    return fetch


def image_dir(level=1.0):
    """A model folder like the one baked into the image."""
    folder = Path(tempfile.mkdtemp())
    for utility in ["AZPS", "SRP"]:
        joblib.dump(Level(level), folder / f"{utility}.joblib")
    (folder / "model_info.json").write_text(json.dumps({"AZPS": {"version": "image"}, "features": FEATURES}))
    return folder


def test_the_active_models_load_and_predict():
    models, info = load_from_store(fetcher(make_bucket(level=5000.0)), FEATURES)
    assert list(models["AZPS"].predict([[0, 0]])) == [5000.0]
    assert info["AZPS"]["version"] == "v2" and info["SRP"]["version"] == "v2"


def test_a_damaged_model_file_is_refused():
    files = make_bucket()
    files["models/SRP/v2.joblib"] = files["models/SRP/v2.joblib"] + b"x"  # one stray byte
    with pytest.raises(ModelLoadError, match="checksum mismatch for SRP v2"):
        load_from_store(fetcher(files), FEATURES)


def test_a_pointer_with_the_wrong_feature_list_is_refused():
    with pytest.raises(ModelLoadError, match="feature list"):
        load_from_store(fetcher(make_bucket()), ["lag_24", "something_else"])


def test_a_pointer_missing_a_utility_is_refused():
    files = make_bucket()
    pointer = json.loads(files[POINTER_KEY])
    del pointer["SRP"]
    files[POINTER_KEY] = json.dumps(pointer).encode()
    with pytest.raises(ModelLoadError, match="no usable entry for SRP"):
        load_from_store(fetcher(files), FEATURES)


def test_a_pointer_that_is_not_json_is_refused():
    files = make_bucket()
    files[POINTER_KEY] = b"this is not json"
    with pytest.raises(ModelLoadError, match="not valid JSON"):
        load_from_store(fetcher(files), FEATURES)


def test_a_model_file_that_is_not_a_model_is_refused_even_with_a_matching_checksum():
    files = make_bucket()
    garbage = b"definitely not a pickle"
    files["models/AZPS/v2.joblib"] = garbage
    pointer = json.loads(files[POINTER_KEY])
    pointer["AZPS"]["sha256"] = sha256_of(garbage)
    files[POINTER_KEY] = json.dumps(pointer).encode()
    with pytest.raises(ModelLoadError, match="could not read the AZPS v2 model file"):
        load_from_store(fetcher(files), FEATURES)


def test_a_missing_file_in_the_bucket_is_refused():
    files = make_bucket()
    del files["models/SRP/v2.joblib"]
    with pytest.raises(ModelLoadError, match="not found"):
        load_from_store(fetcher(files), FEATURES)


def test_s3_is_used_when_a_bucket_is_set_and_the_source_is_labelled():
    models, info = load_models(image_dir(level=1.0), FEATURES, "my-bucket",
                               make_fetch=lambda bucket: fetcher(make_bucket(level=5000.0)))
    assert list(models["SRP"].predict([[0, 0]])) == [5000.0]  # from the bucket, not the image
    assert info["source"] == "s3://my-bucket"


def test_the_image_copy_is_used_when_s3_fails():
    def broken(bucket):
        raise ModelLoadError("S3 is down")

    models, info = load_models(image_dir(level=1.0), FEATURES, "my-bucket", make_fetch=broken)
    assert list(models["AZPS"].predict([[0, 0]])) == [1.0]
    assert info["source"] == "image"


def test_the_image_copy_is_used_when_the_bucket_content_is_bad():
    files = make_bucket()
    files["models/AZPS/v2.joblib"] += b"x"
    models, info = load_models(image_dir(level=1.0), FEATURES, "my-bucket", make_fetch=lambda b: fetcher(files))
    assert info["source"] == "image" and list(models["SRP"].predict([[0, 0]])) == [1.0]


def test_an_unexpected_error_still_falls_back_to_the_image():
    def surprising(bucket):
        raise RuntimeError("something nobody planned for")

    models, info = load_models(image_dir(level=1.0), FEATURES, "my-bucket", make_fetch=surprising)
    assert info["source"] == "image" and list(models["AZPS"].predict([[0, 0]])) == [1.0]


def test_without_a_bucket_s3_is_never_touched():
    def must_not_be_called(bucket):
        raise AssertionError("S3 should not be contacted when MODEL_BUCKET is not set")

    _, info = load_models(image_dir(), FEATURES, None, make_fetch=must_not_be_called)
    assert info["source"] == "image"