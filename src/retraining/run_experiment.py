"""Historical champion-versus-challenger experiment, on the data you already have.

Run from the project folder, in the normal .venv:
    python -m src.retraining.run_experiment [--mlflow]
Default question: SRP's data changed in April 2026. Would a model refreshed through May have beaten
one trained only through March, on June to August, hours that neither model has seen?
Both models are trained with the same settings (PARAMS), so only the training window differs.
The comparison rules live in compare.py; this file only trains the models and prints the results.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from src.features.build import FEATURES
from src.models.train_lgbm import PARAMS
from src.retraining.compare import MARGIN, MIN_HOURS, run_experiment

FEATURES_PATH = Path("data/processed/features.parquet")


def train_lightgbm(rows: pd.DataFrame, target: pd.Series):
    """The same recipe as train_final.py: one LightGBM model, level target, fixed settings."""
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(rows, target)
    return model


def log_to_mlflow(utility: str, args, result: dict) -> None:
    import mlflow  # imported here so the experiment also runs without --mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("phoenix-grid-forecast-retraining")
    with mlflow.start_run(run_name=f"{utility}_champion_vs_challenger"):
        mlflow.log_params({**PARAMS, "utility": utility, "champion_through": args.champion_through,
                           "challenger_through": args.challenger_through, "holdout_end": args.holdout_end,
                           "margin": MARGIN, "min_hours": MIN_HOURS})
        for _, r in result["table"].iterrows():
            mlflow.log_metric(f"{r['model']}_MAPE_pct", r["MAPE_pct"])
        mlflow.set_tag("decision", "promote" if result["decision"]["promote"] else "keep champion")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a refreshed model with the current one.")
    parser.add_argument("--champion-through", default="2026-04-01",
                        help="champion trained on data before this date")
    parser.add_argument("--challenger-through", default="2026-06-01",
                        help="challenger trained before this date; the holdout starts here")
    parser.add_argument("--holdout-end", default="2026-09-01", help="the holdout ends before this date")
    parser.add_argument("--mlflow", action="store_true",
                        help="also record the results in the local MLflow database")
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_PATH).dropna(subset=["roll_mean_168"])
    print(f"Settings used for both models: {PARAMS}")
    for utility in sorted(df["respondent"].unique()):
        result = run_experiment(df, utility, *(pd.Timestamp(d, tz="UTC") for d in (
            args.champion_through, args.challenger_through, args.holdout_end)), train_lightgbm, FEATURES)
        rows = result["rows"]
        print(f"\n===== {utility} =====")
        print(f"Champion trained on {rows['champion']} hours before {args.champion_through}; "
              f"challenger on {rows['challenger']} hours before {args.challenger_through}.")
        print(f"Holdout: {rows['holdout']} hours from {args.challenger_through} to {args.holdout_end}.\n")
        print(result["table"].round(2).to_string(index=False))
        verdict = "PROMOTE the challenger" if result["decision"]["promote"] else "KEEP the champion"
        print(f"\nVerdict: {verdict}")
        for line in result["decision"]["reasons"]:
            print(f"  {line}")
        if args.mlflow:
            log_to_mlflow(utility, args, result)


if __name__ == "__main__":
    main()