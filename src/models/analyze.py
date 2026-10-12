"""Where does the model fail? Reads the monthly CSV and breaks errors down."""

import pandas as pd

m = pd.read_csv("reports/monthly_mape.csv")
m["year"] = m["ym"].str[:4]
m["month"] = m["ym"].str[5:].astype(int)

# 1) Worst months for the best model, with the utility alongside for context
worst = m.sort_values("lgbm_level", ascending=False).groupby("respondent").head(8)
print(worst[["respondent", "ym", "naive_24h", "utility_df", "lgbm_level"]].round(1).to_string(index=False), "\n")

# 2) Average error by calendar month, to see seasonal weakness (e.g. monsoon, shoulder seasons)
print(m.groupby(["respondent", "month"])[["utility_df", "lgbm_level"]].mean().round(1).unstack(0).to_string())