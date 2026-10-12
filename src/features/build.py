"""Build the model-ready feature table: demand + weather + calendar, with leakage-safe lags."""

from pathlib import Path

import holidays
import pandas as pd

DEMAND = Path("data/processed/demand_clean.parquet")
WEATHER = Path("data/raw/weather_2022-01-01_2026-10-08.parquet")
OUT = Path("data/processed/features.parquet")

LOCAL_TZ = "America/Phoenix"
CDH_BASE_C = 20.0  # where the EDA hockey stick bends; we'll tune it later
MIN_LAG = 24       # day-ahead rule: every demand-based feature is at least 24h old

WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation",
                "cloud_cover", "wind_speed_10m"]
FEATURES = (
    ["lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_mean_168"]
    + WEATHER_COLS
    + ["cdh", "temp_roll_72h"]
    + ["hour", "dayofweek", "month", "is_weekend", "is_holiday"]
)


def add_demand_lags(g: pd.DataFrame) -> pd.DataFrame:
    """Lag and rolling features for ONE utility. Input must be an unbroken hourly series."""
    g = g.sort_values("period").reset_index(drop=True)
    # shift(24) only means "24 hours" if no hours are missing, so verify it.
    assert (g["period"].diff().dropna() == pd.Timedelta(hours=1)).all(), "hourly grid has gaps"

    s = g["demand_mw"]
    g["lag_24"] = s.shift(24)    # same hour yesterday
    g["lag_48"] = s.shift(48)
    g["lag_168"] = s.shift(168)  # same hour last week
    # Trailing averages that END 24h before the target hour, so they never see newer data.
    g["roll_mean_24"] = s.shift(MIN_LAG).rolling(24, min_periods=18).mean()
    g["roll_mean_168"] = s.shift(MIN_LAG).rolling(168, min_periods=120).mean()
    return g


def add_weather_features(w: pd.DataFrame) -> pd.DataFrame:
    w = w.sort_values("period").reset_index(drop=True)
    # Cooling degree hours: how far above the AC threshold it is (0 when mild).
    w["cdh"] = (w["temperature_2m"] - CDH_BASE_C).clip(lower=0)
    # Heat builds up over days, so a 3-day average captures heat waves.
    w["temp_roll_72h"] = w["temperature_2m"].rolling(72, min_periods=48).mean()
    return w[["period"] + WEATHER_COLS + ["cdh", "temp_roll_72h"]]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    local = df["period"].dt.tz_convert(LOCAL_TZ)  # local time, because "hour of day" is a local concept
    us_holidays = holidays.US(years=range(2022, 2028))
    df["hour"] = local.dt.hour
    df["dayofweek"] = local.dt.dayofweek
    df["month"] = local.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = local.dt.date.map(lambda d: d in us_holidays).astype(int)
    return df


def build() -> pd.DataFrame:
    demand = pd.read_parquet(DEMAND)
    weather = add_weather_features(pd.read_parquet(WEATHER))
    df = demand.merge(weather, on="period", how="left")
    df = pd.concat([add_demand_lags(g) for _, g in df.groupby("respondent")], ignore_index=True)
    df = add_calendar_features(df)
    # Keep only rows where the target and the two key lags exist, so every model
    # (baselines included) is scored on exactly the same hours.
    keep = df["demand_mw"].notna() & df["lag_24"].notna() & df["lag_168"].notna()
    return df.loc[keep, ["period", "respondent", "demand_mw", "was_repaired"] + FEATURES]


def main() -> None:
    out = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(out.groupby("respondent")["period"].agg(["min", "max", "count"]))
    print(out[FEATURES].isna().sum()[lambda s: s > 0].to_string() or "no NaNs in features")
    print(f"Saved {len(out):,} rows to {OUT}")


if __name__ == "__main__":
    main()