import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.retraining import registry as reg


class Result:
    """What a database call hands back: some rows, a count of changed rows, and column names."""

    def __init__(self, rows=None, rowcount=1, names=()):
        self.rows, self.rowcount = list(rows or []), rowcount
        self.description = [type("Col", (), {"name": n}) for n in names]

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class Transaction:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.conn.rolled_back = exc_type is not None  # a real transaction undoes its changes on an error
        return False


class FakeConn:
    """Records every statement, and answers with the results you queued up, in order."""

    def __init__(self, results=()):
        self.calls, self.results, self.rolled_back = [], list(results), False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return self.results.pop(0) if self.results else Result()

    def transaction(self):
        return Transaction(self)


def stored_row(data=b"model-bytes", version="v1", digest=None):
    """A row shaped like the database's answer to the champion lookup."""
    digest = digest or hashlib.sha256(data).hexdigest()
    return (version, pd.Timestamp("2026-10-08", tz="UTC"), 41569, digest, data, {"source": "test"})


def test_the_tables_are_created_with_a_one_champion_rule_and_safe_value_lists():
    conn = FakeConn()
    reg.create_tables(conn)
    sql = "\n".join(call[0] for call in conn.calls)
    assert len(conn.calls) == 3
    assert "CREATE UNIQUE INDEX IF NOT EXISTS one_champion_per_utility" in sql
    assert "WHERE role = 'champion'" in sql
    assert "CHECK (role IN ('champion', 'candidate', 'retired'))" in sql
    assert "CHECK (verdict IN ('promote', 'keep', 'skipped'))" in sql


def test_a_model_is_stored_with_its_checksum_and_only_known_roles_are_allowed():
    conn = FakeConn()
    digest = reg.add_model(conn, "SRP", "v2", "candidate", pd.Timestamp("2026-10-27", tz="UTC"), 41000,
                           b"abc", {"challenger_mape": 6.1})
    assert digest == hashlib.sha256(b"abc").hexdigest()
    params = conn.calls[0][1]
    assert params[:3] == ("SRP", "v2", "candidate") and params[6] == b"abc"
    assert json.loads(params[7]) == {"challenger_mape": 6.1}
    with pytest.raises(ValueError, match="role must be one of"):
        reg.add_model(conn, "SRP", "v3", "winner", pd.Timestamp("2026-10-27", tz="UTC"), 1, b"x")


def test_the_champion_is_read_back_after_its_checksum_is_verified():
    model = reg.get_model(FakeConn([Result([stored_row()])]), "SRP")
    assert model["version"] == "v1" and model["model_file"] == b"model-bytes"
    assert model["training_rows"] == 41569


def test_a_stored_file_that_no_longer_matches_its_checksum_is_refused():
    conn = FakeConn([Result([stored_row(digest="0" * 64)])])
    with pytest.raises(ValueError, match="does not match its checksum"):
        reg.get_model(conn, "SRP")


def test_asking_for_a_role_nobody_holds_gives_none():
    assert reg.get_model(FakeConn([Result([])]), "SRP", role="candidate") is None


def test_the_next_version_is_one_more_than_the_highest_numbered_one():
    assert reg.next_version(["v1", "v2", "v10", "v1b"]) == "v11"
    assert reg.next_version([]) == "v1"
    assert reg.next_version(["v1b", "test"]) == "v1"  # names that are not vN are ignored


def test_versions_of_lists_what_exists_for_a_utility():
    assert reg.versions_of(FakeConn([Result([("v1",), ("v2",)])]), "SRP") == ["v1", "v2"]


def test_promotion_retires_the_old_champion_first_then_crowns_the_candidate():
    conn = FakeConn([Result(), Result(rowcount=1)])
    reg.promote_candidate(conn, "SRP", "v2")
    assert "role = 'retired'" in conn.calls[0][0] and "role = 'champion'" in conn.calls[1][0]
    assert conn.calls[1][1] == ("SRP", "v2") and conn.rolled_back is False


def test_promoting_something_that_is_not_a_candidate_changes_nothing():
    conn = FakeConn([Result(), Result(rowcount=0)])
    with pytest.raises(ValueError, match="is not a candidate"):
        reg.promote_candidate(conn, "SRP", "v9")
    assert conn.rolled_back is True  # the retirement of the old champion is undone too


def test_a_comparison_is_logged_with_its_reasons_as_json():
    conn = FakeConn()
    reg.log_run(conn, {"utility": "SRP", "champion_version": "v1", "verdict": "keep",
                       "reasons": ["PASS: holdout has 2000 hours", "FAIL: not 5% better"]})
    params = conn.calls[0][1]
    assert params[0] == "SRP" and params[9] == "keep" and params[11] is None  # no candidate was produced
    assert json.loads(params[10]) == ["PASS: holdout has 2000 hours", "FAIL: not 5% better"]


def model_dir():
    folder = Path(tempfile.mkdtemp())
    info = {"features": ["lag_24"]}
    for utility in ["AZPS", "SRP"]:
        (folder / f"{utility}.joblib").write_bytes(f"{utility}-model".encode())
        info[utility] = {"trained_through_utc": "2026-10-08 00:00:00+00:00", "training_rows": 1000}
    (folder / "model_info.json").write_text(json.dumps(info))
    return folder


def test_seeding_records_both_models_as_the_first_champion():
    conn = FakeConn()  # every champion lookup finds nothing
    messages = reg.seed(conn, model_dir(), "v1")
    inserts = [c for c in conn.calls if c[0].startswith("INSERT INTO model_registry")]
    assert len(inserts) == 2 and all(i[1][1:3] == ("v1", "champion") for i in inserts)
    assert messages[0].startswith("AZPS: recorded v1 as champion") and "SRP: recorded" in messages[1]


def test_seeding_never_replaces_a_champion_that_already_exists():
    conn = FakeConn([Result([stored_row()]), Result([])])  # AZPS already has one, SRP does not
    messages = reg.seed(conn, model_dir(), "v1")
    inserts = [c for c in conn.calls if c[0].startswith("INSERT INTO model_registry")]
    assert len(inserts) == 1 and inserts[0][1][0] == "SRP"
    assert messages[0] == "AZPS: a champion already exists, left alone"


def test_the_status_view_never_reads_the_model_files():
    names = ["a"]
    conn = FakeConn([Result([(1,)], names=names), Result([(2,)], names=names)])
    models, runs = reg.status_tables(conn)
    assert all("model_file" not in call[0] for call in conn.calls)
    assert len(models) == 1 and len(runs) == 1