import numpy as np
import pandas as pd
import pytest

from src.features.build import WEATHER_COLS
from src.features.live import HORIZON, build_live_features

START = pd.Timestamp("2026-01-01", tz="UTC")
N = 240  # 10 days of hourly data, like the real fetch


def make_raw(n=N, last_value=None, utility="AZPS", base=3000.0):
    """Fake EIA demand: base + hour number, so each value reveals its hour."""
    period = pd.date_range(START, periods=n, freq="h")
    value = base + np.arange(n, dtype=float)
    if last_value is not None:
        value[-1] = last_value
    return pd.DataFrame({"period": period, "respondent": utility, "type": "D", "value": value})


def make_weather(hours=N + 96):
    """Fake weather that extends well past the demand data, like a real forecast does."""
    period = pd.date_range(START, periods=hours, freq="h")
    return pd.DataFrame({"period": period, **{c: 25.0 for c in WEATHER_COLS}})


def test_window_is_24_consecutive_hours_after_last_actual():
    rows, last = build_live_features("AZPS", make_raw(), make_weather())
    assert len(rows) == HORIZON
    assert rows["period"].min() == last + pd.Timedelta(hours=1)
    assert (rows["period"].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_lags_point_to_the_right_hours():
    rows, _ = build_live_features("AZPS", make_raw(), make_weather())
    for lag, col in [(24, "lag_24"), (168, "lag_168")]:
        hours_since_start = (rows["period"] - pd.Timedelta(hours=lag) - START) / pd.Timedelta(hours=1)
        assert (rows[col].values == 3000.0 + hours_since_start.values).all()


def test_partial_newest_hour_is_not_used():
    # The real failure we hit: EIA's newest hour was a partial reading (1,310 vs ~4,300).
    raw = make_raw(last_value=1000.0)
    rows, last = build_live_features("AZPS", raw, make_weather())
    assert last == raw["period"].max() - pd.Timedelta(hours=2)  # bad hour and its neighbor are skipped
    assert rows["lag_24"].notna().all()                          # no blank clues, so no prediction cliff


def test_other_utilities_do_not_leak_in():
    raw = pd.concat([make_raw(utility="AZPS"), make_raw(utility="SRP", base=90000.0)])
    rows, _ = build_live_features("AZPS", raw, make_weather())
    assert rows["lag_24"].max() < 4000


def test_missing_weather_forecast_fails_loudly():
    short_weather = make_weather(hours=N)  # ends exactly where demand ends, so the future has no weather
    with pytest.raises(ValueError):
        build_live_features("AZPS", make_raw(), short_weather)

def test_no_usable_demand_fails_with_a_clear_message():
    raw = make_raw()
    raw["value"] = 0.0  # every hour is non-positive, so none is usable
    with pytest.raises(ValueError, match="usable"):
        build_live_features("AZPS", raw, make_weather())