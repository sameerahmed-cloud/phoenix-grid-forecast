"""Backtest: model vs copy-yesterday by calendar month, per utility.

Run from the project folder:  python -m src.models.monthly_vs_naive
Reads reports/predictions.parquet (made by src.models.train_lgbm). Months are Phoenix time and are
pooled across years, so "11" means every November in the backtest (2023, 2024, 2025).
The backtest used actual weather, so it shows how the model does when the weather input is perfect.
"""

import pandas as pd

from src.models.baselines import LOCAL_TZ, mape

p = pd.read_parquet("reports/predictions.parquet")
p = p.dropna(subset=["demand_mw", "lgbm_level", "naive_24h"])
p["month"] = p["period"].dt.tz_convert(LOCAL_TZ).dt.month

rows = [{"respondent": r, "month": m, "hours": len(g),
         "model_mape": mape(g["demand_mw"], g["lgbm_level"]),
         "naive_mape": mape(g["demand_mw"], g["naive_24h"])}
        for (r, m), g in p.groupby(["respondent", "month"])]
table = pd.DataFrame(rows)
table["model_wins"] = table["model_mape"] < table["naive_mape"]
print(table.round(2).to_string(index=False))