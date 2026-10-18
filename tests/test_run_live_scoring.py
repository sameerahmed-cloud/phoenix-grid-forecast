import pandas as pd

from src.models import run_live_scoring as r
from src.models.score_live import score_all


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
    def __init__(self, calls):
        self.calls = calls

    def execute(self, sql):
        self.calls.append(("execute", sql))

    def cursor(self):
        return FakeCursor(self.calls)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def graded_rows():
    log = pd.DataFrame({"made_at_utc": "2026-10-16 02:00", "utility": "AZPS",
                        "period_utc": ["2026-10-16 22:00"], "predicted_mw": [5000.0],
                        "model_trained_through_utc": "2026-10-08 00:00"})
    actuals = pd.DataFrame({"utility": "AZPS",
                            "period_utc": pd.to_datetime(["2026-10-15 22:00", "2026-10-16 22:00"], utc=True),
                            "demand_mw": [5100.0, 5400.0], "usable": True})
    scored, _, _ = score_all(log, actuals, pd.Timestamp("2027-01-01", tz="UTC"))
    return scored


def test_save_scores_creates_the_table_then_upserts_every_row(monkeypatch):
    calls = []
    monkeypatch.setattr(r.psycopg, "connect", lambda url, connect_timeout=None: FakeConn(calls))
    sent = r.save_scores("postgresql://fake", graded_rows())
    assert sent == 1
    assert [c[0] for c in calls] == ["execute", "executemany"]  # table first, then the rows
    assert "CREATE TABLE IF NOT EXISTS forecast_scores" in calls[0][1]
    _, sql, rows = calls[1]
    assert "ON CONFLICT (made_at_utc, utility, period_utc) DO UPDATE" in sql  # rerunning never duplicates
    assert sql.count("%s") == len(rows[0]) == 9                              # one placeholder per column