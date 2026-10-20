import io
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.api.model_store import POINTER_KEY, ModelLoadError, load_from_store
from src.retraining.publish_model import (
    activate,
    describe,
    meta_key,
    model_key,
    read_pointer,
    upload_version,
)

FEATURES = ["lag_24", "temperature_2m"]


class Level:
    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        return np.full(len(rows), self.value)


class MemoryStore:
    """A fake bucket with the same three operations as the real one."""

    def __init__(self):
        self.files = {}

    def get(self, key):
        return self.files[key]

    def put(self, key, data):
        self.files[key] = data

    def exists(self, key):
        return key in self.files


def model_dir(level=5000.0, features=None):
    folder = Path(tempfile.mkdtemp())
    info = {"features": features or FEATURES}
    for utility in ["AZPS", "SRP"]:
        joblib.dump(Level(level), folder / f"{utility}.joblib")
        info[utility] = {"trained_through_utc": "2026-10-08 00:00:00+00:00", "training_rows": 1000,
                         "target": "demand_mw (level)"}
    (folder / "model_info.json").write_text(json.dumps(info))
    return folder


def fetch_from(store):
    def fetch(key):
        try:
            return store.get(key)
        except KeyError as e:
            raise ModelLoadError(f"not found: {key}") from e
    return fetch


def live_level(store):
    models, _ = load_from_store(fetch_from(store), FEATURES)
    return float(models["SRP"].predict([[0, 0]])[0])


def test_upload_stores_each_model_with_its_details_but_does_not_make_it_live():
    store = MemoryStore()
    published = upload_version(store, model_dir(), "v1")
    assert model_key("AZPS", "v1") in store.files and meta_key("SRP", "v1") in store.files
    assert published["AZPS"]["version"] == "v1" and len(published["AZPS"]["sha256"]) == 64
    assert POINTER_KEY not in store.files  # nothing is live until activate is run


def test_uploading_the_same_content_twice_is_fine_but_changed_content_under_the_same_name_is_refused():
    store = MemoryStore()
    upload_version(store, model_dir(level=5000.0), "v1")
    upload_version(store, model_dir(level=5000.0), "v1")
    with pytest.raises(ValueError, match="already exists with different content"):
        upload_version(store, model_dir(level=6000.0), "v1")


def test_version_names_are_restricted_to_safe_characters():
    with pytest.raises(ValueError, match="version names"):
        upload_version(MemoryStore(), model_dir(), "v1/../../escape")


def test_activate_makes_the_version_live_and_the_api_can_load_it():
    store = MemoryStore()
    upload_version(store, model_dir(level=5000.0), "v1")
    pointer = activate(store, "v1")
    assert pointer["SRP"]["version"] == "v1" and pointer["features"] == FEATURES
    assert live_level(store) == 5000.0  # published by this tool, loaded the way the API does


def test_rolling_back_means_activating_an_older_version():
    store = MemoryStore()
    upload_version(store, model_dir(level=5000.0), "v1")
    upload_version(store, model_dir(level=6000.0), "v2")
    activate(store, "v2")
    assert live_level(store) == 6000.0
    activate(store, "v1")
    assert live_level(store) == 5000.0
    assert model_key("SRP", "v2") in store.files  # the newer version is kept, just not live


def test_activating_one_utility_leaves_the_other_untouched():
    store = MemoryStore()
    upload_version(store, model_dir(level=5000.0), "v1")
    upload_version(store, model_dir(level=6000.0), "v2")
    activate(store, "v1")
    pointer = activate(store, "v2", only=["SRP"])
    assert pointer["SRP"]["version"] == "v2" and pointer["AZPS"]["version"] == "v1"


def test_activating_one_utility_needs_the_other_to_be_live_already():
    store = MemoryStore()
    upload_version(store, model_dir(), "v1")
    with pytest.raises(ValueError, match="AZPS has no live version yet"):
        activate(store, "v1", only=["SRP"])


def test_a_version_that_was_never_uploaded_cannot_be_activated():
    with pytest.raises(ValueError, match="was never uploaded"):
        activate(MemoryStore(), "v9")


def test_a_changed_feature_list_cannot_be_activated_for_just_one_utility():
    store = MemoryStore()
    upload_version(store, model_dir(), "v1")
    upload_version(store, model_dir(features=["lag_24", "something_new"]), "v2")
    activate(store, "v1")
    with pytest.raises(ValueError, match="different feature list"):
        activate(store, "v2", only=["SRP"])


def test_status_reads_what_is_live_and_says_so_when_nothing_is():
    store = MemoryStore()
    assert "baked into its image" in describe(read_pointer(store))[0]
    upload_version(store, model_dir(), "v1")
    activate(store, "v1")
    assert describe(read_pointer(store))[0].startswith("AZPS: version v1, trained through 2026-10-08")
    assert isinstance(io.BytesIO(), io.BytesIO)