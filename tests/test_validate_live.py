import numpy as np
import pandas as pd
import pytest

from src.features.validate_live import (
    MIN_HOURLY_ROWS,
    WEATHER_LIMITS,
    DataValidationError,
    check_fresh,
    validate_demand,
    validate_live_inputs,
    validate_weather,
)

START = pd.Timestamp("2026-10-01", tz="UTC")


def demand(n=240, base=4000.0, utility="AZPS"):
    """A calm, realistic-looking demand fetch: a daily wave around `base`."""
    period = pd.date_range(START, periods=n, freq="h")
    value = base + 500 * np.sin(np.arange(n) / 24 * 2 * np.pi)
    return pd.DataFrame({"period": period, "respondent": utility, "type": "D", "value": value})


def weather(hours=100):
    period = pd.date_range(START, periods=hours, freq="h")
    mid = {col: (low + high) / 2 for col, (low, high) in WEATHER_LIMITS.items()}
    return pd.DataFrame({"period": period, **mid})


def test_good_demand_passes_with_no_warnings():
    assert validate_demand(demand(), "AZPS") == []


def test_missing_column_is_named_in_the_error():
    with pytest.raises(DataValidationError, match="type"):
        validate_demand(demand().drop(columns="type"), "AZPS")


def test_too_few_rows_names_the_utility_and_the_count():
    with pytest.raises(DataValidationError, match="only 100 .* AZPS"):
        validate_demand(demand(n=100), "AZPS")


def test_another_utilitys_rows_do_not_hide_a_shortage():
    both = pd.concat([demand(n=10, utility="AZPS"), demand(n=240, utility="SRP")])
    with pytest.raises(DataValidationError, match="AZPS"):
        validate_demand(both, "AZPS")
    assert validate_demand(both, "SRP") == []  # SRP is fine even though AZPS is short


def test_duplicate_hours_are_rejected():
    raw = demand()
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_demand(pd.concat([raw, raw.iloc[[5]]]), "AZPS")


def test_wrong_units_are_rejected():
    with pytest.raises(DataValidationError, match="units"):
        validate_demand(demand(base=4_000_000.0), "AZPS")  # kilowatts instead of megawatts


def test_one_extreme_reading_does_not_fail_the_run():
    raw = demand()
    raw.loc[100, "value"] = 90_000.0  # a single bad hour is cleaning's job, not a reason to stop
    assert validate_demand(raw, "AZPS") == []


def test_many_flagged_hours_give_a_warning_not_an_error():
    raw = demand()
    raw.loc[raw.index[::3], "value"] = 40_000.0  # every third hour is a spike
    warnings = validate_demand(raw, "AZPS")
    assert len(warnings) == 1 and "flagged" in warnings[0]


def test_good_weather_passes():
    assert validate_weather(weather()) == []


def test_weather_blanks_are_left_for_later_and_do_not_fail():
    w = weather()
    w.loc[3, "temperature_2m"] = np.nan
    assert validate_weather(w) == []


def test_impossible_temperature_is_rejected():
    w = weather()
    w.loc[0, "temperature_2m"] = 400.0
    with pytest.raises(DataValidationError, match="temperature_2m"):
        validate_weather(w)


def test_empty_or_incomplete_weather_is_rejected():
    with pytest.raises(DataValidationError, match="empty"):
        validate_weather(weather().iloc[0:0])
    with pytest.raises(DataValidationError, match="cloud_cover"):
        validate_weather(weather().drop(columns="cloud_cover"))


def test_validate_live_inputs_runs_both_checks():
    assert validate_live_inputs(demand(), weather(), "AZPS") == []
    with pytest.raises(DataValidationError, match="Weather"):
        validate_live_inputs(demand(), weather().drop(columns="cloud_cover"), "AZPS")
    with pytest.raises(DataValidationError, match="demand"):
        validate_live_inputs(demand(n=MIN_HOURLY_ROWS - 1), weather(), "AZPS")


NOW = pd.Timestamp("2026-10-20 12:00", tz="UTC")


def test_recent_data_passes_quietly():
    assert check_fresh(NOW - pd.Timedelta(hours=3), NOW) == []


def test_somewhat_old_data_warns_but_still_works():
    # A partial newest hour already pushes the newest usable hour back about 2 hours, so this must not fail.
    warnings = check_fresh(NOW - pd.Timedelta(hours=8), NOW)
    assert len(warnings) == 1 and "8.0 hours" in warnings[0]
    assert len(check_fresh(NOW - pd.Timedelta(hours=12), NOW)) == 1  # exactly at the limit still passes


def test_stale_data_is_rejected():
    with pytest.raises(DataValidationError, match="12.5 hours old"):
        check_fresh(NOW - pd.Timedelta(hours=12.5), NOW)


def test_no_usable_hour_is_rejected():
    with pytest.raises(DataValidationError, match="No usable"):
        check_fresh(pd.NaT, NOW)