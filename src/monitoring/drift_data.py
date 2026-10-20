"""Prepare the 'before' and 'after' tables for the drift report.

Pure pandas, so it runs (and is tested) without Evidently. Both windows are the same three calendar
months, one year apart, so that normal seasonal change is not mistaken for drift.
"""

import numpy as np
import pandas as pd

from src.models.baselines import mape

LOCAL_TZ = "America/Phoenix"
REFERENCE_MONTHS = ["2025-04", "2025-05", "2025-06"]
CURRENT_MONTHS = ["2026-04", "2026-05", "2026-06"]
MODEL = "lgbm_level"  # the model the API serves
MODELS = ["naive_24h", "utility_df", "lgbm_level", "lgbm_relative"]
FEATURE_COLUMNS = ["temperature_2m", "relative_humidity_2m", "shortwave_radiation", "cloud_cover",
                   "wind_speed_10m", "cdh", "lag_24", "roll_mean_168"]
DRIFT_COLUMNS = ["demand_mw", MODEL, "abs_pct_error", *FEATURE_COLUMNS]


def build_table(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """One row per utility-hour: demand, every model's prediction, the model inputs, and the error."""
    needed = ["period", "respondent", *FEATURE_COLUMNS]
    missing = set(needed) - set(features.columns)
    if missing:
        raise ValueError(f"features table is missing columns: {sorted(missing)}")
    table = predictions[["period", "respondent", "demand_mw", *MODELS]].merge(
        features[needed], on=["period", "respondent"], how="inner")
    table["abs_pct_error"] = (table["demand_mw"] - table[MODEL]).abs() / table["demand_mw"] * 100
    return table


def local_month(period: pd.Series) -> pd.Series:
    """Calendar month in Phoenix time, e.g. '2026-04'. The data is stored in UTC."""
    return period.dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m")


def make_windows(table: pd.DataFrame, utility: str):
    """The (reference, current) tables for one utility."""
    t = table[table["respondent"] == utility]
    month = local_month(t["period"])
    return (t[month.isin(REFERENCE_MONTHS)].reset_index(drop=True),
            t[month.isin(CURRENT_MONTHS)].reset_index(drop=True))


def shift_table(reference: pd.DataFrame, current: pd.DataFrame, columns=DRIFT_COLUMNS) -> pd.DataFrame:
    """How far each column moved: percent change in its average, and the move in standard deviations."""
    rows = []
    for c in columns:
        ref_mean, cur_mean, ref_std = reference[c].mean(), current[c].mean(), reference[c].std()
        rows.append({"column": c, "reference_mean": ref_mean, "current_mean": cur_mean,
                     "change_pct": (cur_mean / ref_mean - 1) * 100 if ref_mean else np.nan,
                     "shift_in_std": (cur_mean - ref_mean) / ref_std if ref_std else np.nan})
    return pd.DataFrame(rows)


def error_table(reference: pd.DataFrame, current: pd.DataFrame, models=MODELS) -> pd.DataFrame:
    """MAPE of every model before and after, and how many times worse it got."""
    rows = []
    for m in models:
        before = mape(reference["demand_mw"], reference[m])
        after = mape(current["demand_mw"], current[m])
        rows.append({"model": m, "reference_mape_pct": before, "current_mape_pct": after,
                     "times_worse": after / before})
    return pd.DataFrame(rows)