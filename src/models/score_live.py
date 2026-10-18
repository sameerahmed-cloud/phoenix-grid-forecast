"""Score the live forecasts we logged against what actually happened.

Pure functions only: no database and no network, so every rule below can be unit-tested with
small fake tables. Loading the log and the actuals is a separate step.

Rules (none of them looks at how big a forecast error was):
  1. Broken predictions (missing or not positive) are counted and never scored.
  2. Near-identical re-runs inside one 6-hour window count once: the earliest.
  3. Only hours at least 48 hours old are scored, because EIA revises recent values.
  4. Actuals must be 'usable' under the same cleaning the training data got.
  5. Model and baseline are scored on the same hours: both need a usable actual.
  6. Results are reported by lead time, and the headline excludes the 'under_1h' bucket,
     which is nearly a nowcast.
"""

import numpy as np
import pandas as pd

from src.models.baselines import mae, mape, peak_mask

MATURITY_HOURS = 48
DEDUP_WINDOW = "6h"
BUCKET_EDGES = [-np.inf, 1, 6, 12, np.inf]
BUCKET_LABELS = ["under_1h", "1_6h", "6_12h", "12h_plus"]


def prepare_forecasts(log: pd.DataFrame, now: pd.Timestamp):
    """Apply the rules that only need the forecast log. Returns (kept rows, funnel counts)."""
    f = log.copy()
    f["made_at_utc"] = pd.to_datetime(f["made_at_utc"], utc=True)
    f["period_utc"] = pd.to_datetime(f["period_utc"], utc=True)
    f["predicted_mw"] = pd.to_numeric(f["predicted_mw"], errors="coerce")
    funnel = {"logged": len(f)}

    broken = ~np.isfinite(f["predicted_mw"]) | (f["predicted_mw"] <= 0)
    funnel["broken_prediction"] = int(broken.sum())
    f = f[~broken].copy()

    # Keep the earliest forecast per utility, target hour and 6-hour window of when it was made.
    f["window"] = f["made_at_utc"].dt.floor(DEDUP_WINDOW)
    f = f.sort_values("made_at_utc").drop_duplicates(["utility", "period_utc", "window"], keep="first")
    funnel["after_dedup"] = len(f)

    f = f[f["period_utc"] <= now - pd.Timedelta(hours=MATURITY_HOURS)].copy()
    funnel["mature"] = len(f)

    f["lead_h"] = (f["period_utc"] - f["made_at_utc"]) / pd.Timedelta(hours=1)
    f["bucket"] = pd.cut(f["lead_h"], BUCKET_EDGES, labels=BUCKET_LABELS, right=False)
    return f.drop(columns="window").reset_index(drop=True), funnel


def attach_actuals(f: pd.DataFrame, actuals: pd.DataFrame):
    """Add the actual demand and the copy-yesterday baseline. Returns (scored rows, funnel counts).

    `actuals` needs the columns utility, period_utc, demand_mw, usable.
    """
    a = actuals.copy()
    a["period_utc"] = pd.to_datetime(a["period_utc"], utc=True)
    a = a.loc[a["usable"], ["utility", "period_utc", "demand_mw"]]

    out = f.merge(a.rename(columns={"demand_mw": "actual_mw"}), on=["utility", "period_utc"], how="left")
    # Copy-yesterday for hour H is the actual demand at H minus 24 hours, so shift actuals forward a day.
    prev = a.assign(period_utc=a["period_utc"] + pd.Timedelta(hours=24))
    prev = prev.rename(columns={"demand_mw": "naive_24h"})
    out = out.merge(prev, on=["utility", "period_utc"], how="left")

    funnel = {"no_usable_actual": int(out["actual_mw"].isna().sum()),
              "no_usable_baseline": int((out["actual_mw"].notna() & out["naive_24h"].isna()).sum())}
    scored = out.dropna(subset=["actual_mw", "naive_24h"]).reset_index(drop=True)
    funnel["scored"] = len(scored)
    return scored, funnel


def _row(g: pd.DataFrame, utility: str, label: str) -> dict:
    pk = peak_mask(g)
    y, p, b = g["actual_mw"], g["predicted_mw"], g["naive_24h"]
    return {"utility": utility, "slice": label, "hours": len(g),
            "MAE_model": mae(y, p), "MAE_naive": mae(y, b),
            "MAPE_model": mape(y, p), "MAPE_naive": mape(y, b),
            "peak_hours": int(pk.sum()),
            "peak_MAPE_model": mape(y[pk], p[pk]), "peak_MAPE_naive": mape(y[pk], b[pk])}


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per utility and lead-time slice, plus the headline (everything except under_1h)."""
    s = scored.rename(columns={"period_utc": "period"})
    rows = []
    for utility, u in s.groupby("utility"):
        head = u[u["bucket"].astype(str) != "under_1h"]
        if len(head):
            rows.append(_row(head, utility, "headline_1h_plus"))
        for b in BUCKET_LABELS:
            g = u[u["bucket"].astype(str) == b]
            if len(g):
                rows.append(_row(g, utility, b))
    return pd.DataFrame(rows)