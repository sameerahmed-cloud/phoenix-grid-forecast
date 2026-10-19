import os
import tempfile

import numpy as np
import pandas as pd

from src.monitoring import run_health_check as r


class FakeCursor:
    def __init__(self, rows, names):
        self.rows, self.description = rows, [type("Col", (), {"name": n}) for n in names]

    def execute(self, sql):
        self.sql = sql

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def recent_scores(model_err, naive_err):
    """Ten days of graded hours ending two days before the real 'now', the way live data looks."""
    end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(days=2)
    period = pd.date_range(end=end, periods=240, freq="h")
    sign = np.where(np.arange(240) % 2 == 0, 1.0, -1.0)
    return pd.DataFrame({"utility": "AZPS", "bucket": "6_12h", "period_utc": period, "actual_mw": 5000.0,
                         "predicted_mw": 5000.0 * (1 + sign * model_err),
                         "naive_24h": 5000.0 * (1 + sign * naive_err)})


def run_main(monkeypatch, scores):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", os.path.join(tempfile.mkdtemp(), "summary.md"))
    monkeypatch.setattr(r, "load_dotenv", lambda: None)
    monkeypatch.setattr(r, "load_scores", lambda url: scores)
    try:
        r.main()
    except SystemExit as e:
        return e.code, os.environ["GITHUB_STEP_SUMMARY"]
    return 0, os.environ["GITHUB_STEP_SUMMARY"]


def test_load_scores_returns_a_table_with_the_column_names(monkeypatch):
    names = ["utility", "bucket", "period_utc", "predicted_mw", "actual_mw", "naive_24h"]
    cursor = FakeCursor([("AZPS", "6_12h", "2026-10-20", 5000.0, 5100.0, 5050.0)], names)
    monkeypatch.setattr(r.psycopg, "connect", lambda url, connect_timeout=None: FakeConn(cursor))
    df = r.load_scores("postgresql://fake")
    assert list(df.columns) == names and len(df) == 1
    assert "forecast_scores" in cursor.sql and "40 days" in cursor.sql


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_write_summary_appends_to_the_given_file():
    path = os.path.join(tempfile.mkdtemp(), "summary.md")
    r.write_summary("first", path)
    r.write_summary("second", path)
    assert read(path) == "first\nsecond\n"


def test_a_healthy_run_exits_cleanly_and_writes_the_summary(monkeypatch):
    code, summary_path = run_main(monkeypatch, recent_scores(model_err=0.04, naive_err=0.06))
    assert code == 0
    assert "### AZPS: OK" in read(summary_path)


def test_a_failing_check_exits_with_code_1_so_github_turns_red(monkeypatch):
    code, summary_path = run_main(monkeypatch, recent_scores(model_err=0.08, naive_err=0.05))
    assert code == 1
    assert "FAIL" in read(summary_path)


def test_missing_database_url_stops_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(r, "load_dotenv", lambda: None)
    try:
        r.main()
    except SystemExit as e:
        assert "DATABASE_URL" in str(e.code)
    else:
        raise AssertionError("main() should have stopped")