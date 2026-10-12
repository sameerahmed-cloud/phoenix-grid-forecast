"""Score the benchmarks the ML model must beat: yesterday, last week, and the utility's own forecast."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = Path("data/processed/features.parquet")
RAW = Path("data/raw/eia_2022-01-01_2026-10-08.parquet")
LOCAL_TZ = "America/Phoenix"
EVAL_START = pd.Timestamp("2023-01-01", tz="UTC")  # 2022 stays free as training history
MODELS = ["naive_24h", "naive_168h", "utility_df"]


def mae(y, p) -> float:
    return float(np.abs(y - p).mean())


def mape(y, p) -> float:
    return float((np.abs(y - p) / y).mean() * 100)


def peak_mask(g: pd.DataFrame) -> pd.Series:
    """Summer evening peak window (Jun-Sep, 4-8 PM local), chosen by clock, not by outcome."""
    local = g["period"].dt.tz_convert(LOCAL_TZ)
    return local.dt.month.isin([6, 7, 8, 9]) & local.dt.hour.between(16, 19)


def load_eval_table() -> pd.DataFrame:
    f = pd.read_parquet(FEATURES)
    f = f[f["period"] >= EVAL_START]
    f = f[["period", "respondent", "demand_mw", "lag_24", "lag_168"]].rename(
        columns={"lag_24": "naive_24h", "lag_168": "naive_168h"})
    raw = pd.read_parquet(RAW)
    fc = raw.loc[raw["type"] == "DF", ["period", "respondent", "value"]].rename(
        columns={"value": "utility_df"})
    return f.merge(fc, on=["period", "respondent"], how="left")


def main() -> None:
    ev = load_eval_table()
    print(ev.groupby("respondent").agg(eval_hours=("demand_mw", "size"),
                                       with_forecast=("utility_df", "count")), "\n")

    # Fair comparison: score every model on the same hours, and only where the utility forecast exists.
    common = ev.dropna(subset=["utility_df"]).copy()
    common["year"] = common["period"].dt.tz_convert(LOCAL_TZ).dt.year

    rows = []
    for resp, g in common.groupby("respondent"):
        pk = peak_mask(g)
        for m in MODELS:
            rows.append({
                "respondent": resp, "model": m, "hours": len(g),
                "MAE_MW": mae(g["demand_mw"], g[m]),
                "MAPE_%": mape(g["demand_mw"], g[m]),
                "peak_MAPE_%": mape(g.loc[pk, "demand_mw"], g.loc[pk, m]),
            })
    print(pd.DataFrame(rows).round(2).to_string(index=False), "\n")

    by_year = [{"respondent": r, "year": y, "model": m, "MAPE_%": mape(g["demand_mw"], g[m])}
               for (r, y), g in common.groupby(["respondent", "year"]) for m in MODELS]
    print(pd.DataFrame(by_year).pivot_table(index=["respondent", "year"], columns="model",
                                            values="MAPE_%").round(2))


if __name__ == "__main__":
    main()