import numpy as np
import pandas as pd

from src.features import build as build_module
from src.features.build import FEATURES, WEATHER_COLS, build, build_features


def inputs(hours=400):
    """Cleaned demand for two utilities, and raw weather that covers the same hours."""
    period = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    frames = []
    for utility, base in [("AZPS", 4000.0), ("SRP", 5000.0)]:
        wave = 300 * np.sin(np.arange(hours) / 24 * 2 * np.pi)
        frames.append(pd.DataFrame({"period": period, "respondent": utility, "demand_mw": base + wave,
                                    "was_repaired": False, "usable": True, "flag_reason": ""}))
    weather = pd.DataFrame({"period": period})
    for i, column in enumerate(WEATHER_COLS):
        weather[column] = 20.0 + i + 5 * np.sin(np.arange(hours) / 24 * 2 * np.pi)
    return pd.concat(frames, ignore_index=True), weather


def test_the_table_has_the_expected_columns_and_only_complete_rows():
    demand, weather = inputs()
    table = build_features(demand, weather)
    assert list(table.columns) == ["period", "respondent", "demand_mw", "was_repaired", *FEATURES]
    for column in ["demand_mw", "lag_24", "lag_168"]:
        assert table[column].notna().all()
    # the first week has no lag_168 yet
    assert table["period"].min() == pd.Timestamp("2026-01-08", tz="UTC")


def test_build_features_does_not_change_the_data_it_is_given():
    demand, weather = inputs()
    demand_before, weather_before = demand.copy(), weather.copy()
    build_features(demand, weather)
    pd.testing.assert_frame_equal(demand, demand_before)
    pd.testing.assert_frame_equal(weather, weather_before)


def test_build_reads_its_two_files_and_gives_the_same_table(monkeypatch):
    demand, weather = inputs()
    files = {build_module.DEMAND: demand, build_module.WEATHER: weather}
    monkeypatch.setattr(pd, "read_parquet", lambda path: files[path].copy())
    pd.testing.assert_frame_equal(build(), build_features(demand, weather))