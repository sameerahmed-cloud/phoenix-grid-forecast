import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.retraining import rehearsal as rh
from src.retraining.publish_model import upload_version

FEATS = ["t", "lag_24"]
LIVE = {"DATABASE_URL": "postgresql://live-host/db", "MODEL_BUCKET": "phoenix-grid-forecast-123"}
SAFE = {"DATABASE_URL": "postgresql://rehearsal-branch-host/db",
        "MODEL_BUCKET": "phoenix-grid-forecast-rehearsal-123"}


def test_a_proper_rehearsal_setup_passes_the_guard():
    rh.check_rehearsal_targets(SAFE, LIVE)


def test_the_live_database_is_refused():
    with pytest.raises(ValueError, match="live database"):
        rh.check_rehearsal_targets({**SAFE, "DATABASE_URL": LIVE["DATABASE_URL"]}, LIVE)


def test_a_missing_database_setting_is_refused():
    with pytest.raises(ValueError, match="DATABASE_URL must be set"):
        rh.check_rehearsal_targets({"MODEL_BUCKET": SAFE["MODEL_BUCKET"]}, LIVE)


def test_the_live_bucket_is_refused():
    with pytest.raises(ValueError, match="separate bucket"):
        rh.check_rehearsal_targets({**SAFE, "MODEL_BUCKET": LIVE["MODEL_BUCKET"]}, LIVE)


def test_a_bucket_without_rehearsal_in_its_name_is_refused():
    with pytest.raises(ValueError, match="'rehearsal'"):
        rh.check_rehearsal_targets({**SAFE, "MODEL_BUCKET": "some-other-bucket"}, LIVE)


def test_the_guard_refuses_when_it_has_no_live_setting_to_compare_against():
    with pytest.raises(ValueError, match="cannot find your live DATABASE_URL"):
        rh.check_rehearsal_targets(SAFE, {})


class SpyModel:
    def __init__(self, level):
        self.level = level

    def predict(self, rows):
        return np.full(len(rows), self.level)


def make_table():
    period = pd.date_range("2026-01-01", periods=200 * 24, freq="h", tz="UTC")
    frames = [pd.DataFrame({"period": period, "respondent": u, "demand_mw": 5000.0, "lag_24": 5000.0,
                            "t": np.arange(len(period))}) for u in ["AZPS", "SRP"]]
    return pd.concat(frames, ignore_index=True)


def test_weak_models_train_only_on_data_before_the_cutoff_and_record_it():
    seen = []

    def spy(rows, target):
        seen.append(int(rows["t"].max()))
        return SpyModel(float(target.mean()))

    cutoff = pd.Timestamp("2026-04-01", tz="UTC")
    models, info = rh.make_weak_champions(make_table(), cutoff, FEATS, train_fn=spy)
    cutoff_t = int((cutoff - pd.Timestamp("2026-01-01", tz="UTC")) / pd.Timedelta(hours=1))
    assert set(models) == {"AZPS", "SRP"} and all(t < cutoff_t for t in seen)
    assert info["SRP"]["trained_through_utc"] == "2026-03-31 23:00:00+00:00"
    assert info["SRP"]["training_rows"] == 90 * 24


def test_a_cutoff_before_any_data_is_refused():
    with pytest.raises(ValueError, match="no training data before"):
        rh.make_weak_champions(make_table(), pd.Timestamp("2025-01-01", tz="UTC"), FEATS,
                               train_fn=lambda rows, target: SpyModel(1.0))


class MemoryStore:
    def __init__(self):
        self.files = {}

    def get(self, key):
        return self.files[key]

    def put(self, key, data):
        self.files[key] = data

    def exists(self, key):
        return key in self.files


def test_the_written_folder_is_accepted_by_the_normal_upload_command():
    models = {"AZPS": SpyModel(1.0), "SRP": SpyModel(2.0)}
    info = {u: {"trained_through_utc": "2026-08-31 23:00:00+00:00", "training_rows": 100,
                "target": "demand_mw (level)"} for u in models}
    folder = Path(tempfile.mkdtemp()) / "rehearsal_models"
    rh.write_model_dir(folder, models, info, FEATS)
    assert json.loads((folder / "model_info.json").read_text())["features"] == FEATS
    store = MemoryStore()
    uploaded = upload_version(store, folder, "v1")
    assert set(uploaded) == {"AZPS", "SRP"} and "models/SRP/v1.joblib" in store.files


class RecordingConn:
    def __init__(self):
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append(sql)


def test_reset_creates_the_tables_if_needed_then_empties_both():
    conn = RecordingConn()
    rh.reset_registry(conn)
    assert any("CREATE TABLE IF NOT EXISTS model_registry" in s for s in conn.statements)
    assert conn.statements[-2:] == ["DELETE FROM retrain_runs", "DELETE FROM model_registry"]


def test_reset_refuses_to_touch_the_live_database(monkeypatch):
    monkeypatch.setattr(rh, "dotenv_values", lambda path: dict(LIVE))
    monkeypatch.setenv("DATABASE_URL", LIVE["DATABASE_URL"])
    monkeypatch.setenv("MODEL_BUCKET", SAFE["MODEL_BUCKET"])

    def must_not_connect(url, connect_timeout=None):
        raise AssertionError("the guard should have stopped the reset before connecting")

    monkeypatch.setattr(rh.psycopg, "connect", must_not_connect)
    try:
        rh.main(["reset"])
    except SystemExit as e:
        assert "Refusing to reset" in str(e.code) and "live database" in str(e.code)
    else:
        raise AssertionError("the reset should have been refused")


def test_reset_runs_against_a_proper_rehearsal_setup(monkeypatch):
    monkeypatch.setattr(rh, "dotenv_values", lambda path: dict(LIVE))
    monkeypatch.setenv("DATABASE_URL", SAFE["DATABASE_URL"])
    monkeypatch.setenv("MODEL_BUCKET", SAFE["MODEL_BUCKET"])
    conn = RecordingConn()

    class Connection:
        def __enter__(self):
            return conn

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(rh.psycopg, "connect", lambda url, connect_timeout=None: Connection())
    rh.main(["reset"])
    assert conn.statements[-1] == "DELETE FROM model_registry"