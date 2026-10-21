import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURES
from src.retraining import training_data as td


def make_df(n=3, utility="SRP", start="2026-01-01"):
    """A small table with the same columns, order and kinds of values as features.parquet."""
    df = pd.DataFrame({"period": pd.date_range(start, periods=n, freq="h", tz="UTC"), "respondent": utility,
                       "demand_mw": np.arange(n) + 4000.0, "was_repaired": [False] * n})
    for column in FEATURES:
        df[column] = (np.arange(n) % 7).astype(int) if column in td.INT_COLUMNS else np.arange(n) * 1.5
    return df


class FakeCursor:
    def __init__(self, calls):
        self.calls = calls

    def executemany(self, sql, rows):
        self.calls.append(("executemany", sql, list(rows)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self):
        self.calls = []

    def execute(self, sql):
        self.calls.append(("execute", sql))

    def cursor(self):
        return FakeCursor(self.calls)


def test_the_table_definition_has_every_column_once_and_a_primary_key():
    ddl = td.create_sql()
    assert "PRIMARY KEY (period, respondent)" in ddl
    for column in td.COLUMNS:
        assert ddl.count(f"\n    {column} ") == 1
    assert "hour SMALLINT" in ddl and "lag_24 DOUBLE PRECISION" in ddl
    assert "was_repaired BOOLEAN NOT NULL" in ddl


def test_records_have_plain_python_values_in_table_order_and_blanks_become_null():
    df = make_df()
    df.loc[1, "roll_mean_168"] = np.nan
    records = td.to_records(df)
    assert len(records) == 3 and len(records[0]) == len(td.COLUMNS)
    first, second = records[0], records[1]
    assert first[0].tzinfo is not None and first[1] == "SRP" and first[2] == 4000.0 and first[3] is False
    assert second[td.COLUMNS.index("roll_mean_168")] is None
    assert type(first[td.COLUMNS.index("hour")]) is int and type(first[td.COLUMNS.index("lag_24")]) is float


def test_a_period_without_a_timezone_is_refused():
    df = make_df()
    df["period"] = df["period"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        td.to_records(df)


def test_a_missing_target_is_refused():
    df = make_df()
    df.loc[0, "demand_mw"] = np.nan
    with pytest.raises(ValueError, match="demand_mw"):
        td.to_records(df)


def test_upsert_creates_the_table_first_then_sends_rows_in_batches():
    conn = FakeConn()
    assert td.upsert_rows(conn, make_df(n=12), batch_size=5) == 12
    kinds = [c[0] for c in conn.calls]
    assert kinds == ["execute", "executemany", "executemany", "executemany"]
    assert "CREATE TABLE IF NOT EXISTS training_features" in conn.calls[0][1]
    assert [len(c[2]) for c in conn.calls[1:]] == [5, 5, 2]  # 12 rows in batches of 5
    sql = conn.calls[1][1]
    assert sql.count("%s") == len(td.COLUMNS) and "ON CONFLICT (period, respondent) DO UPDATE" in sql
    assert "period = EXCLUDED.period" not in sql  # the key is never rewritten


def test_identical_tables_verify_clean():
    assert td.verify(make_df(), make_df()) == []


def test_verify_names_the_utility_and_what_differs():
    base = pd.concat([make_df(utility="AZPS"), make_df(utility="SRP")], ignore_index=True)
    assert "SRP: 2 rows in Neon but 3 in the parquet" in td.verify(base.drop(base.index[-1]), base)[0]

    shifted = base.copy()
    shifted.loc[shifted["respondent"] == "SRP", "period"] += pd.Timedelta(hours=1)
    assert td.verify(shifted, base) == ["SRP: the time range differs"]

    changed = base.copy()
    changed.loc[changed["respondent"] == "AZPS", "temperature_2m"] += 1.0
    assert td.verify(changed, base) == ["AZPS: the total of temperature_2m differs"]


def test_a_utility_missing_from_neon_is_reported():
    base = pd.concat([make_df(utility="AZPS"), make_df(utility="SRP")], ignore_index=True)
    problems = td.verify(base[base["respondent"] == "AZPS"], base)
    assert problems == ["SRP: 0 rows in Neon but 3 in the parquet file"]


def test_load_table_filters_by_time_and_returns_utc(monkeypatch):
    captured = {}

    class Cursor:
        def __init__(self):
            self.description = [type("Col", (), {"name": c}) for c in td.COLUMNS]

        def execute(self, sql, params):
            captured["sql"], captured["params"] = sql, params

        def fetchall(self):
            return td.to_records(make_df(n=2))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Conn:
        def cursor(self):
            return Cursor()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(td.psycopg, "connect", lambda url, connect_timeout=None: Conn())
    df = td.load_table("postgresql://fake", since=pd.Timestamp("2026-01-01", tz="UTC"))
    assert "WHERE period >= %s" in captured["sql"]
    assert captured["sql"].endswith("ORDER BY respondent, period")
    assert len(captured["params"]) == 1 and len(df) == 2 and str(df["period"].dt.tz) == "UTC"