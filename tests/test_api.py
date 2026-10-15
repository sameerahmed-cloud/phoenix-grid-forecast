import numpy as np
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

from src.api import main
from src.features.build import FEATURES

# Not used as a "with" block on purpose: that would run startup and need real model files.
client = TestClient(main.app)


class FakeModel:
    def predict(self, X):
        return np.full(len(X), 5000.0)


def fake_rows():
    period = pd.date_range("2026-10-15 01:00", periods=24, freq="h", tz="UTC")
    rows = pd.DataFrame({f: 0.0 for f in FEATURES}, index=range(24))
    rows["period"] = period
    rows["hour"] = period.tz_convert("America/Phoenix").hour
    return rows


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Swap the model, the live data, and the log file for fakes."""
    info = {u: {"trained_through_utc": "2026-10-08 00:00:00+00:00"} for u in ["AZPS", "SRP"]}
    monkeypatch.setitem(main.state, "models", {"AZPS": FakeModel(), "SRP": FakeModel()})
    monkeypatch.setitem(main.state, "info", info)
    monkeypatch.setattr(main, "get_live_inputs", lambda: {"demand": None, "weather": None})
    monkeypatch.setattr(main, "build_live_features",
                        lambda u, d, w: (fake_rows(), pd.Timestamp("2026-10-15 00:00", tz="UTC")))
    log = tmp_path / "forecast_log.csv"
    monkeypatch.setattr(main, "LOG_PATH", log)
    return log


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_forecast_returns_24_hours(wired):
    r = client.get("/forecast", params={"utility": "AZPS"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecasts"]) == 24
    assert body["forecasts"][0]["predicted_mw"] == 5000.0
    assert body["latest_actual_utc"].startswith("2026-10-15T00:00")


def test_forecast_is_logged(wired):
    client.get("/forecast", params={"utility": "SRP"})
    log = pd.read_csv(wired)
    assert len(log) == 24
    assert set(log["utility"]) == {"SRP"}
    assert {"made_at_utc", "period_utc", "predicted_mw", "model_trained_through_utc"} <= set(log.columns)


def test_unknown_utility_is_rejected(wired):
    assert client.get("/forecast", params={"utility": "XYZ"}).status_code == 422


def test_upstream_failure_returns_502(wired, monkeypatch):
    def down():
        raise requests.ConnectionError("EIA unreachable")
    monkeypatch.setattr(main, "get_live_inputs", down)
    assert client.get("/forecast", params={"utility": "AZPS"}).status_code == 502


def test_weather_gap_returns_503(wired, monkeypatch):
    def gap(u, d, w):
        raise ValueError("Weather forecast does not cover the forecast window")
    monkeypatch.setattr(main, "build_live_features", gap)
    assert client.get("/forecast", params={"utility": "AZPS"}).status_code == 503