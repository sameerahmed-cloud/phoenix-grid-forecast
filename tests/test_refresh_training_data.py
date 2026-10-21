import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURES, WEATHER_COLS
from src.features.validate_live import DataValidationError
from src.retraining import refresh_training_data as r

END = pd.Timestamp("2026-10-20", tz="UTC")


def raw_demand(days=20, extra_type_rows=True):
    """EIA-style rows: actual demand ("D") for two utilities, plus the utility's own forecast ("DF")."""
    period = pd.date_range(end=END, periods=days * 24, freq="h")
    frames = []
    for utility, base in [("AZPS", 4000.0), ("SRP", 5000.0)]:
        wave = 300 * np.sin(np.arange(len(period)) / 24 * 2 * np.pi)
        actual = {"period": period, "respondent": utility, "type": "D", "value": base + wave}
        frames.append(pd.DataFrame(actual))
        if extra_type_rows:
            forecast = {"period": period, "respondent": utility, "type": "DF", "value": 99999.0}
            frames.append(pd.DataFrame(forecast))
    return pd.concat(frames, ignore_index=True)


def weather(days=24, end=END):
    period = pd.date_range(end=end, periods=days * 24, freq="h")
    out = pd.DataFrame({"period": period})
    for i, column in enumerate(WEATHER_COLS):
        out[column] = 20.0 + i + 5 * np.sin(np.arange(len(period)) / 24 * 2 * np.pi)
    return out


class FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_only_actual_demand_is_cleaned_and_each_utility_stays_separate():
    cleaned = r.clean_demand_frame(raw_demand(days=10))
    assert set(cleaned["respondent"]) == {"AZPS", "SRP"} and len(cleaned) == 2 * 10 * 24
    assert {"demand_mw", "was_repaired", "usable"} <= set(cleaned.columns)
    assert cleaned["demand_mw"].max() < 10_000  # the 99,999 forecast rows never get in


def test_new_rows_are_complete_and_start_after_the_context_week():
    rows = r.new_training_rows(raw_demand(days=20), weather())
    assert rows[FEATURES].notna().all().all() and len(rows) > 0
    first_demand_hour = END - pd.Timedelta(days=20) + pd.Timedelta(hours=1)
    # lag_168 needs a week of history before the first stored row
    assert rows["period"].min() >= first_demand_hour + pd.Timedelta(days=7)


def test_hours_without_weather_yet_are_left_out():
    rows = r.new_training_rows(raw_demand(days=20), weather(end=END - pd.Timedelta(days=3)))
    assert rows["period"].max() <= END - pd.Timedelta(days=3)  # the archive trails real time


def test_a_short_demand_delivery_is_refused():
    with pytest.raises(DataValidationError, match="hourly demand rows"):
        r.new_training_rows(raw_demand(days=3), weather())


def test_impossible_weather_is_refused():
    bad = weather()
    bad.loc[5, "temperature_2m"] = 400.0
    with pytest.raises(DataValidationError, match="temperature_2m"):
        r.new_training_rows(raw_demand(days=20), bad)


def prepare_main(monkeypatch, rows_or_none=None):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    monkeypatch.setattr(r, "load_dotenv", lambda: None)
    monkeypatch.setattr(r, "fetch_inputs", lambda now, days: (raw_demand(days=20), weather()))
    if rows_or_none is not None:
        monkeypatch.setattr(r, "new_training_rows", lambda raw, w: rows_or_none)


def test_a_dry_run_reports_but_never_connects_to_the_database(monkeypatch):
    prepare_main(monkeypatch)

    def must_not_connect(url, connect_timeout=None):
        raise AssertionError("a dry run must not touch the database")

    monkeypatch.setattr(r.psycopg, "connect", must_not_connect)
    r.main(["--dry-run"])


def test_a_normal_run_sends_every_new_row_to_the_table(monkeypatch):
    prepare_main(monkeypatch)
    sent = []
    monkeypatch.setattr(r.psycopg, "connect", lambda url, connect_timeout=None: FakeConn())
    monkeypatch.setattr(r, "upsert_rows", lambda conn, df: sent.append(len(df)) or len(df))
    r.main([])
    expected = len(r.new_training_rows(raw_demand(days=20), weather()))
    assert sent == [expected] and expected > 48


def test_too_few_rows_stops_before_anything_is_written(monkeypatch):
    tiny = r.new_training_rows(raw_demand(days=20), weather()).head(10)
    prepare_main(monkeypatch, rows_or_none=tiny)

    def must_not_connect(url, connect_timeout=None):
        raise AssertionError("nothing should be written when too few rows were built")

    monkeypatch.setattr(r.psycopg, "connect", must_not_connect)
    try:
        r.main([])
    except SystemExit as e:
        assert "Only 10 rows" in str(e.code)
    else:
        raise AssertionError("the job should have stopped")


def test_a_missing_database_url_stops_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(r, "load_dotenv", lambda: None)
    try:
        r.main([])
    except SystemExit as e:
        assert "DATABASE_URL" in str(e.code)
    else:
        raise AssertionError("the job should have stopped")