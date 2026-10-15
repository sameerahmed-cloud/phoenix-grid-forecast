"""Train one final model per utility on ALL available data and save it for the API."""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from src.features.build import FEATURES
from src.models.train_lgbm import (
    PARAMS,  # same settings as the backtest, so nothing differs
)

FEATURES_PATH = Path("data/processed/features.parquet")
MODEL_DIR = Path("models")


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH).dropna(subset=["roll_mean_168"])
    MODEL_DIR.mkdir(exist_ok=True)

    info = {}
    for resp, g in df.groupby("respondent"):  # one model per utility, as in the backtest
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(g[FEATURES], g["demand_mw"])  # "level" target: it won on average
        joblib.dump(model, MODEL_DIR / f"{resp}.joblib")
        info[resp] = {"trained_through_utc": str(g["period"].max()),
                      "training_rows": len(g), "target": "demand_mw (level)"}
        print(resp, info[resp])

    info["features"] = FEATURES
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()