"""Checks on freshly fetched data, before the model is allowed to trust it.

`clean_one` already repairs single bad hours. These checks look at the whole delivery instead:
did enough data arrive, does it look sane, and is it recent. A hard problem raises
DataValidationError, which the API turns into a 503, so no bad forecast is made or logged.
A softer problem is returned as a warning message and also written to the log.
"""

import logging

import pandas as pd

from src.processing.clean_demand import clean_one

log = logging.getLogger(__name__)

DEMAND_COLUMNS = {"period", "respondent", "type", "value"}
MIN_HOURLY_ROWS = 192                  # 8 days: the 168-hour lag plus a day of context
DEMAND_MEDIAN_RANGE = (1_000, 15_000)  # MW. Catches unit mix-ups; single bad hours are cleaning's job
MAX_FLAGGED_SHARE = 0.10               # cleaning normally flags about 0.4% of hours
STALE_WARN_HOURS = 6                   # newest usable hour is normally 1 to 4 hours old
STALE_FAIL_HOURS = 12                  # a partial newest hour already pushes it back about 2 hours

WEATHER_LIMITS = {                     # generous physical limits, in the units Open-Meteo returns
    "temperature_2m": (-30, 60),
    "relative_humidity_2m": (0, 100),
    "shortwave_radiation": (0, 1500),
    "cloud_cover": (0, 100),
    "wind_speed_10m": (0, 250),
}


class DataValidationError(ValueError):
    """The fetched data failed a check, so no forecast should be made from it."""


def _warn(messages: list[str]) -> list[str]:
    for message in messages:
        log.warning(message)
    return messages


def validate_demand(raw: pd.DataFrame, utility: str) -> list[str]:
    """Check one utility's slice of the EIA demand fetch. Returns warnings; raises on a hard failure."""
    missing = DEMAND_COLUMNS - set(raw.columns)
    if missing:
        raise DataValidationError(f"EIA demand data is missing columns: {sorted(missing)}")

    g = raw[raw["respondent"] == utility]
    if len(g) < MIN_HOURLY_ROWS:
        raise DataValidationError(
            f"EIA returned only {len(g)} hourly demand rows for {utility}; at least {MIN_HOURLY_ROWS} are needed")
    if g["period"].duplicated().any():
        raise DataValidationError(f"EIA demand for {utility} contains duplicate hours")

    values = pd.to_numeric(g["value"], errors="coerce")
    median = values.median()
    low, high = DEMAND_MEDIAN_RANGE
    if not low <= median <= high:
        raise DataValidationError(
            f"Typical {utility} demand is {median:,.0f} MW, outside {low:,} to {high:,}; "
            "the units or the feed may have changed")

    series = values.set_axis(pd.DatetimeIndex(g["period"])).sort_index()
    flagged = (clean_one(series)["flag_reason"] != "").mean()
    warnings = []
    if flagged > MAX_FLAGGED_SHARE:
        warnings.append(
            f"{flagged:.0%} of recent {utility} demand hours were flagged by cleaning (normally about 0.4%)")
    return _warn(warnings)


def validate_weather(weather: pd.DataFrame) -> list[str]:
    """Check the weather forecast fetch. Returns warnings (none today); raises on a hard failure."""
    missing = ({"period"} | set(WEATHER_LIMITS)) - set(weather.columns)
    if missing:
        raise DataValidationError(f"Weather data is missing columns: {sorted(missing)}")
    if weather.empty:
        raise DataValidationError("The weather forecast came back empty")
    for column, (low, high) in WEATHER_LIMITS.items():
        values = weather[column].dropna()  # blanks are handled later, where the forecast window is known
        bad = int(((values < low) | (values > high)).sum())
        if bad:
            raise DataValidationError(f"Weather column {column} has {bad} values outside {low} to {high}")
    return []


def validate_live_inputs(demand: pd.DataFrame, weather: pd.DataFrame, utility: str) -> list[str]:
    """Run every input check for one utility's forecast."""
    return validate_demand(demand, utility) + validate_weather(weather)


def check_fresh(last: pd.Timestamp, now) -> list[str]:
    """The forecast starts right after `last`, so `last` must be recent. Raises if it is not."""
    if pd.isna(last):
        raise DataValidationError("No usable demand hours in the latest EIA data")
    age = (pd.Timestamp(now) - pd.Timestamp(last)) / pd.Timedelta(hours=1)
    if age > STALE_FAIL_HOURS:
        raise DataValidationError(
            f"The newest usable demand hour is {age:.1f} hours old (limit {STALE_FAIL_HOURS}); "
            "EIA may have stopped updating")
    if age > STALE_WARN_HOURS:
        return _warn([f"The newest usable demand hour is {age:.1f} hours old (normally under 4)"])
    return []