"""Expanding-window backtest: LightGBM vs the baselines, logged to MLflow."""

from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd

from src.features.build import FEATURES
from src.models.baselines import LOCAL_TZ, load_eval_table, mae, mape, peak_mask

FEATURES_PATH = Path("data/processed/features.parquet")
REPORT_DIR = Path("reports")
FIRST_TEST = pd.Timestamp("2023-01-01", tz="UTC")  # same evaluation start as the baselines
MODES = ["level", "relative"]
MODELS = ["naive_24h", "utility_df", "lgbm_level", "lgbm_relative"]

# Fixed settings on purpose: tuning them against the test blocks would leak the answer.
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=42, verbose=-1)


def fold_edges(g: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """3-month test blocks from 2023 to the last available hour."""
    starts = pd.date_range(FIRST_TEST, g["period"].max(), freq="3MS")
    ends = list(starts[1:]) + [g["period"].max() + pd.Timedelta(hours=1)]
    return list(zip(starts, ends))


def make_target(df: pd.DataFrame, mode: str) -> pd.Series:
    # "relative": demand as a multiple of last week's average level, so growth is not something to learn
    return df["demand_mw"] / df["roll_mean_168"] if mode == "relative" else df["demand_mw"]


def to_mw(pred, df: pd.DataFrame, mode: str):
    return pred * df["roll_mean_168"] if mode == "relative" else pred


def backtest(df: pd.DataFrame, mode: str):
    """Train on everything before each block, predict the block. Returns predictions + last models."""
    preds, last_models = [], {}
    for resp, g in df.groupby("respondent"):  # one model per utility: their demand patterns differ
        g = g.sort_values("period")
        for start, end in fold_edges(g):
            train = g[g["period"] < start]
            test = g[(g["period"] >= start) & (g["period"] < end)]
            if test.empty:
                continue
            model = lgb.LGBMRegressor(**PARAMS)
            model.fit(train[FEATURES], make_target(train, mode))
            out = test[["period", "respondent"]].copy()
            out[f"lgbm_{mode}"] = to_mw(model.predict(test[FEATURES]), test, mode)
            preds.append(out)
            last_models[resp] = model
    return pd.concat(preds, ignore_index=True), last_models


def summarize(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for resp, g in common.groupby("respondent"):
        pk = peak_mask(g)
        for m in MODELS:
            rows.append({"respondent": resp, "model": m, "hours": len(g),
                         "MAE_MW": mae(g["demand_mw"], g[m]),
                         "MAPE_pct": mape(g["demand_mw"], g[m]),
                         "peak_MAPE_pct": mape(g.loc[pk, "demand_mw"], g.loc[pk, m])})
    return pd.DataFrame(rows)


def main() -> None:
    # roll_mean_168 is needed by the "relative" target; dropping its ~47 NaN rows keeps both modes identical
    df = pd.read_parquet(FEATURES_PATH).dropna(subset=["roll_mean_168"])

    ev = load_eval_table()  # actuals + baselines + utility forecast, same as last step
    common, models = ev, {}
    for mode in MODES:
        pred, models[mode] = backtest(df, mode)
        common = common.merge(pred, on=["period", "respondent"], how="inner")
    # Score every model on exactly the same hours: all predictions AND the utility forecast must exist.
    common = common.dropna(subset=MODELS).copy()
    common["year"] = common["period"].dt.tz_convert(LOCAL_TZ).dt.year
    common["ym"] = common["period"].dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m")
    
    REPORT_DIR.mkdir(exist_ok=True)
    common.to_parquet(REPORT_DIR / "predictions.parquet", index=False)  # lets analysis scripts reuse the hourly results

    table = summarize(common)
    print(table.round(2).to_string(index=False), "\n")

    by_year = [{"respondent": r, "year": y, "model": m, "MAPE_pct": mape(g["demand_mw"], g[m])}
               for (r, y), g in common.groupby(["respondent", "year"]) for m in MODELS]
    print(pd.DataFrame(by_year).pivot_table(index=["respondent", "year"], columns="model",
                                            values="MAPE_pct")[MODELS].round(2), "\n")

    # Month-by-month errors saved for inspection, plus a simple win count against the utility.
    monthly = pd.DataFrame([{"respondent": r, "ym": ym, **{m: mape(g["demand_mw"], g[m]) for m in MODELS}}
                            for (r, ym), g in common.groupby(["respondent", "ym"])])
    REPORT_DIR.mkdir(exist_ok=True)
    monthly.round(2).to_csv(REPORT_DIR / "monthly_mape.csv", index=False)
    for resp, g in monthly.groupby("respondent"):
        wins = (g["lgbm_relative"] < g["utility_df"]).sum()
        print(f"{resp}: lgbm_relative beat the utility forecast in {wins} of {len(g)} months")

    print("\nTop features (gain), last fold, relative target:")
    for resp, model in models["relative"].items():
        imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
        print(resp, (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict())

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("phoenix-grid-forecast")
    for mode in MODES:
        with mlflow.start_run(run_name=f"lgbm_{mode}"):
            mlflow.log_params({**PARAMS, "target": mode, "n_features": len(FEATURES),
                               "cv": "expanding window, 3-month blocks from 2023-01"})
            for _, r in table[table["model"] == f"lgbm_{mode}"].iterrows():
                mlflow.log_metrics({f"{r['respondent']}_MAPE": r["MAPE_pct"],
                                    f"{r['respondent']}_peak_MAPE": r["peak_MAPE_pct"],
                                    f"{r['respondent']}_MAE": r["MAE_MW"]})
    print("\nLogged to MLflow. View with: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()