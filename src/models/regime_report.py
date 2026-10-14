"""Scoreboard split into 'before' and 'after' the SRP data change (Apr 2026), plus a look at the odd months."""

import pandas as pd

from src.models.baselines import LOCAL_TZ, mape, peak_mask

MODELS = ["naive_24h", "utility_df", "lgbm_level", "lgbm_relative"]
CUTOFF = pd.Timestamp("2026-04-01", tz=LOCAL_TZ)

p = pd.read_parquet("reports/predictions.parquet")
p["regime"] = (p["period"] >= CUTOFF).map({False: "before_2026-04", True: "from_2026-04"})

rows = []
for (resp, reg), g in p.groupby(["respondent", "regime"]):
    pk = peak_mask(g)
    for m in MODELS:
        rows.append({"respondent": resp, "regime": reg, "model": m, "hours": len(g),
                     "MAPE_pct": mape(g["demand_mw"], g[m]),
                     "peak_MAPE_pct": mape(g.loc[pk, "demand_mw"], g.loc[pk, m])})
print(pd.DataFrame(rows).round(2).to_string(index=False), "\n")

# What do SRP's swings look like day by day in May 2026? (daily average, lowest and highest hour)
f = pd.read_parquet("data/processed/features.parquet")
f["local"] = f["period"].dt.tz_convert(LOCAL_TZ)
may = f[(f["local"] >= "2026-05-01") & (f["local"] < "2026-06-01")].copy()
may["day"] = may["local"].dt.date
daily = may.groupby(["day", "respondent"])["demand_mw"].agg(["mean", "min", "max"]).round(0).unstack()
print(daily.to_string())