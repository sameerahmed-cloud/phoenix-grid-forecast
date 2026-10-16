import pandas as pd

from src.api import store


def sample():
    ts = pd.Timestamp("2026-10-15 01:00", tz="UTC")
    return pd.DataFrame({"made_at_utc": [ts], "utility": ["AZPS"], "period_utc": [ts],
                         "predicted_mw": [5000.0], "model_trained_through_utc": [ts]})[store.COLUMNS]


def test_csv_used_when_no_database_url(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    path = tmp_path / "log.csv"
    assert store.save_forecasts(sample(), path) == "csv"
    assert len(pd.read_csv(path)) == 1


def test_postgres_used_when_database_url_set(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    calls = []
    monkeypatch.setattr(store, "_save_postgres", lambda df, url: calls.append(len(df)))
    path = tmp_path / "log.csv"
    assert store.save_forecasts(sample(), path) == "postgres"
    assert calls == [1]
    assert not path.exists()


def test_falls_back_to_csv_when_postgres_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")

    def boom(df, url):
        raise RuntimeError("database down")

    monkeypatch.setattr(store, "_save_postgres", boom)
    path = tmp_path / "log.csv"
    assert store.save_forecasts(sample(), path) == "csv"
    assert len(pd.read_csv(path)) == 1