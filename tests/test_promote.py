import io
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.api.model_store import ModelLoadError, load_from_store, sha256_of
from src.retraining import promote as p
from src.retraining.publish_model import activate, read_pointer, upload_version

FEATURES = ["lag_24", "temperature_2m"]


class Level:
    """A stand-in model that always predicts one number."""

    def __init__(self, value):
        self.value = value

    def predict(self, rows):
        return np.full(len(rows), self.value)


class WrongWidth(Level):
    n_features_in_ = 5  # a model trained on a different number of features


class MemoryStore:
    def __init__(self):
        self.files = {}

    def get(self, key):
        return self.files[key]

    def put(self, key, data):
        self.files[key] = data

    def exists(self, key):
        return key in self.files


def dump(model) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def live_store(level=5000.0, version="v1"):
    """A bucket where `version` is live for both utilities, like your bucket after the first publish."""
    folder = Path(tempfile.mkdtemp())
    info = {"features": FEATURES}
    for utility in ["AZPS", "SRP"]:
        joblib.dump(Level(level), folder / f"{utility}.joblib")
        info[utility] = {"trained_through_utc": "2026-10-08 00:00:00+00:00", "training_rows": 1000}
    (folder / "model_info.json").write_text(json.dumps(info))
    store = MemoryStore()
    upload_version(store, folder, version)
    activate(store, version)
    return store


def row(version, role, model=None, metrics=None):
    data = dump(model or Level(5000.0))
    return {"version": version, "role": role, "trained_through_utc": pd.Timestamp("2026-10-27", tz="UTC"),
            "training_rows": 41000, "sha256": sha256_of(data), "model_file": data, "metrics": metrics or {}}


def registry(monkeypatch, candidate, champion, extra=()):
    """Replace the registry calls with recorders. `extra` are more rows get_version may be asked for."""
    rows = {(r["version"]): r for r in [candidate, champion, *extra]}
    calls = {"promoted": [], "restored": []}
    monkeypatch.setattr(p, "get_version", lambda conn, utility, version: rows.get(version))
    monkeypatch.setattr(p, "get_model", lambda conn, utility, role="champion": champion)
    monkeypatch.setattr(p, "promote_candidate",
                        lambda conn, utility, version: calls["promoted"].append(version))
    monkeypatch.setattr(p, "restore_version",
                        lambda conn, utility, version: calls["restored"].append(version))
    return calls


def yes(lines, expected):
    return True


def no(lines, expected):
    return False


def live_level(store, utility):
    """The number the live model for `utility` would predict, loaded the way the API loads it."""
    def fetch(key):
        if key not in store.files:
            raise ModelLoadError(f"not found: {key}")
        return store.get(key)

    models, _ = load_from_store(fetch, FEATURES)
    return float(models[utility].predict([[0, 0]])[0])


def test_promotion_uploads_the_model_moves_the_pointer_and_crowns_it_in_the_registry(monkeypatch):
    store = live_store(level=5000.0)
    champion = row("v1", "champion", Level(5000.0))
    calls = registry(monkeypatch, row("v2", "candidate", Level(6000.0)), champion)
    lines = p.promote(object(), store, "SRP", "v2", FEATURES, yes)
    assert read_pointer(store)["SRP"]["version"] == "v2" and read_pointer(store)["AZPS"]["version"] == "v1"
    # what the API would load after its next cold start
    assert live_level(store, "SRP") == 6000.0 and live_level(store, "AZPS") == 5000.0
    assert calls["promoted"] == ["v2"]
    assert any("Registry updated" in line for line in lines) and any("cold start" in line for line in lines)


def test_declining_the_confirmation_changes_nothing(monkeypatch):
    store = live_store()
    before = dict(store.files)
    calls = registry(monkeypatch, row("v2", "candidate", Level(6000.0)), row("v1", "champion"))
    assert p.promote(object(), store, "SRP", "v2", FEATURES, no) == ["Cancelled. Nothing was changed."]
    assert store.files == before and calls["promoted"] == []


def test_something_that_is_not_a_candidate_is_refused(monkeypatch):
    registry(monkeypatch, row("v2", "retired"), row("v1", "champion"))
    with pytest.raises(ValueError, match="is a retired, not a candidate"):
        p.promote(object(), live_store(), "SRP", "v2", FEATURES, yes)


def test_an_unknown_version_is_refused(monkeypatch):
    registry(monkeypatch, row("v2", "candidate"), row("v1", "champion"))
    with pytest.raises(ValueError, match="v9 is not in the registry"):
        p.promote(object(), live_store(), "SRP", "v9", FEATURES, yes)


def test_s3_and_the_registry_disagreeing_blocks_promotion_unless_forced(monkeypatch):
    store = live_store(level=5000.0)
    champion = row("v1", "champion", Level(1234.0))  # not the file S3 is actually serving
    calls = registry(monkeypatch, row("v2", "candidate", Level(6000.0)), champion)
    with pytest.raises(ValueError, match="does not call the champion"):
        p.promote(object(), store, "SRP", "v2", FEATURES, yes)
    p.promote(object(), store, "SRP", "v2", FEATURES, yes, force=True)
    assert calls["promoted"] == ["v2"]


def test_a_different_version_name_for_the_same_file_is_not_a_disagreement(monkeypatch):
    store = live_store(level=5000.0, version="v1b")  # live under another name, same content as the champion
    champion = row("v1", "champion", Level(5000.0))
    calls = registry(monkeypatch, row("v2", "candidate", Level(6000.0)), champion)
    p.promote(object(), store, "SRP", "v2", FEATURES, yes)
    assert calls["promoted"] == ["v2"]


def test_a_candidate_that_expects_the_wrong_number_of_features_is_refused(monkeypatch):
    registry(monkeypatch, row("v2", "candidate", WrongWidth(6000.0)), row("v1", "champion"))
    with pytest.raises(ValueError, match="expects 5 features but the API builds 2"):
        p.promote(object(), live_store(), "SRP", "v2", FEATURES, yes)


def test_a_file_that_cannot_be_loaded_is_refused():
    with pytest.raises(ValueError, match="cannot be loaded"):
        p.check_loadable(b"not a model", FEATURES)


def test_running_again_after_a_halfway_failure_finishes_the_job(monkeypatch):
    store = live_store()
    candidate, champion = row("v2", "candidate", Level(6000.0)), row("v1", "champion")
    calls = registry(monkeypatch, candidate, champion)
    attempts = []

    def flaky_promote(conn, utility, version):
        attempts.append(version)
        if len(attempts) == 1:
            raise RuntimeError("the database dropped the connection")
        calls["promoted"].append(version)

    monkeypatch.setattr(p, "promote_candidate", flaky_promote)
    with pytest.raises(RuntimeError):
        p.promote(object(), store, "SRP", "v2", FEATURES, yes)
    assert read_pointer(store)["SRP"]["version"] == "v2"          # S3 got ahead of the registry
    p.promote(object(), store, "SRP", "v2", FEATURES, yes)         # the same command finishes the job
    assert calls["promoted"] == ["v2"] and read_pointer(store)["SRP"]["version"] == "v2"


def test_a_version_cannot_be_overwritten_in_s3_with_different_content():
    store = live_store(level=5000.0)
    with pytest.raises(ValueError, match="already exists in S3 with different content"):
        p.upload_model_file(store, "SRP", "v1", dump(Level(9999.0)), {}, FEATURES)


def test_rollback_points_s3_back_and_restores_the_old_champion(monkeypatch):
    store = live_store(level=5000.0)
    champion_v2 = row("v2", "champion", Level(6000.0))
    # put v2 in S3 and make it live, the way a promotion would have
    p.upload_model_file(store, "SRP", "v2", champion_v2["model_file"],
                        {"trained_through_utc": "2026-10-27", "training_rows": 41000}, FEATURES)
    activate(store, "v2", only=["SRP"])
    assert live_level(store, "SRP") == 6000.0
    calls = registry(monkeypatch, row("v1", "retired", Level(5000.0)), champion_v2)
    lines = p.rollback(object(), store, "SRP", "v1", yes)
    assert live_level(store, "SRP") == 5000.0 and calls["restored"] == ["v1"]
    assert any("champion again" in line for line in lines)


def test_rollback_only_restores_a_model_that_was_once_live(monkeypatch):
    registry(monkeypatch, row("v3", "candidate"), row("v1", "champion"))
    with pytest.raises(ValueError, match="only a retired model can be restored"):
        p.rollback(object(), live_store(), "SRP", "v3", yes)


def test_the_exact_version_name_must_be_typed_to_confirm():
    assert p.confirm_by_typing(["change"], "v2", input_fn=lambda prompt: "v2") is True
    assert p.confirm_by_typing(["change"], "v2", input_fn=lambda prompt: " v2 ") is True
    assert p.confirm_by_typing(["change"], "v2", input_fn=lambda prompt: "yes") is False
    assert p.confirm_by_typing(["change"], "v2", input_fn=lambda prompt: "") is False


def test_the_summary_shows_the_holdout_result_and_how_to_undo(monkeypatch):
    metrics = {"challenger_mape": 6.1, "champion_mape": 7.4, "baseline_mape": 10.0,
               "holdout_start": "2026-10-27", "holdout_end": "2026-11-24"}
    lines = p.describe_promotion("SRP", row("v2", "candidate", metrics=metrics), row("v1", "champion"))
    text = "\n".join(lines)
    assert "candidate 6.10% vs champion 7.40% vs copy-yesterday 10.00%" in text
    assert "--rollback-to v1" in text


class ScriptedConn:
    """Answers with queued results, and records statements, like the registry tests."""

    def __init__(self, results=()):
        self.calls, self.results, self.rolled_back = [], list(results), False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return self.results.pop(0)

    def transaction(self):
        conn = self

        class Transaction:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                conn.rolled_back = exc_type is not None
                return False

        return Transaction()


class Answer:
    def __init__(self, rows=None, rowcount=1):
        self.rows, self.rowcount = rows or [], rowcount

    def fetchone(self):
        return self.rows[0] if self.rows else None


def test_get_version_returns_any_role_and_verifies_the_checksum():
    data = b"model-bytes"
    good = ("v2", "candidate", pd.Timestamp("2026-10-27", tz="UTC"), 41000, sha256_of(data), data, {"a": 1})
    found = p.get_version(ScriptedConn([Answer([good])]), "SRP", "v2")
    assert found["role"] == "candidate" and found["metrics"] == {"a": 1}
    assert p.get_version(ScriptedConn([Answer([])]), "SRP", "v9") is None
    bad = (*good[:4], "0" * 64, data, {})
    with pytest.raises(ValueError, match="does not match its checksum"):
        p.get_version(ScriptedConn([Answer([bad])]), "SRP", "v2")


def test_restore_retires_the_champion_then_restores_and_undoes_itself_on_failure():
    conn = ScriptedConn([Answer(), Answer(rowcount=1)])
    p.restore_version(conn, "SRP", "v1")
    assert "role = 'retired'" in conn.calls[0][0] and "role = 'champion'" in conn.calls[1][0]
    failing = ScriptedConn([Answer(), Answer(rowcount=0)])
    with pytest.raises(ValueError, match="is not a retired model"):
        p.restore_version(failing, "SRP", "v7")
    assert failing.rolled_back is True


def test_main_needs_both_the_database_and_the_bucket(monkeypatch):
    monkeypatch.setattr(p, "load_dotenv", lambda: None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MODEL_BUCKET", raising=False)
    try:
        p.main(["--utility", "SRP", "--version", "v2"])
    except SystemExit as e:
        assert "MODEL_BUCKET" in str(e.code)
    else:
        raise AssertionError("main() should have stopped")


def test_main_promotes_end_to_end_with_the_yes_flag(monkeypatch):
    store = live_store(level=5000.0)
    champion = row("v1", "champion", Level(5000.0))
    calls = registry(monkeypatch, row("v2", "candidate", Level(6000.0)), champion)

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    monkeypatch.setenv("MODEL_BUCKET", "my-bucket")
    monkeypatch.setattr(p, "load_dotenv", lambda: None)
    monkeypatch.setattr(p, "S3Store", lambda bucket: store)
    monkeypatch.setattr(p, "FEATURES", FEATURES)
    monkeypatch.setattr(p.psycopg, "connect", lambda url, connect_timeout=None: FakeConn())
    p.main(["--utility", "SRP", "--version", "v2", "--yes"])
    assert calls["promoted"] == ["v2"] and read_pointer(store)["SRP"]["version"] == "v2"