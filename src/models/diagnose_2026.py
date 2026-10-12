"""Did the level or the day-to-day behavior of the data change in 2025-2026?"""

import pandas as pd

f = pd.read_parquet("data/processed/features.parquet")
f["ym"] = f["period"].dt.tz_convert("America/Phoenix").dt.strftime("%Y-%m")
f = f[f["ym"] >= "2025-01"].sort_values(["respondent", "period"])

# 1) Average demand per month, SRP relative to AZPS, and average temperature.
lvl = f.groupby(["ym", "respondent"])["demand_mw"].mean().unstack().round(0)
lvl["SRP_over_AZPS"] = (lvl["SRP"] / lvl["AZPS"]).round(3)
temp = f[f["respondent"] == "AZPS"].groupby("ym")["temperature_2m"].mean().round(1).rename("avg_temp_C")
print(lvl.join(temp).to_string(), "\n")

# 2) Typical day-to-day change at the same hour, using the median so one spike can't distort it.
f["day_change_pct"] = (f["demand_mw"] / f["lag_24"] - 1).abs() * 100
print(f.groupby(["ym", "respondent"])["day_change_pct"].median().unstack().round(1).to_string())