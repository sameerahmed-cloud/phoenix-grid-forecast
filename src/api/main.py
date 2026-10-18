"""FastAPI service: request a day and a utility, get 24 hourly demand forecasts."""

import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.api.store import COLUMNS, save_forecasts
from src.features.build import FEATURES, LOCAL_TZ
from src.features.live import (
    build_live_features,
    fetch_recent_demand,
    fetch_weather_forecast,
)
from src.features.validate_live import (
    DataValidationError,
    check_fresh,
    validate_live_inputs,
)

MODEL_DIR = Path("models")
FEATURES_PATH = Path("data/processed/features.parquet")
state: dict = {}

LOG_PATH = Path(os.getenv("LOG_PATH", "logs/forecast_log.csv"))
CACHE_MINUTES = 30


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: load models and the feature table into memory."""
    state["models"] = {r: joblib.load(MODEL_DIR / f"{r}.joblib") for r in ["AZPS", "SRP"]}
    state["info"] = json.loads((MODEL_DIR / "model_info.json").read_text())
    if FEATURES_PATH.exists():  # only the historical replay endpoint needs this
        feats = pd.read_parquet(FEATURES_PATH)
        feats["local_date"] = feats["period"].dt.tz_convert(LOCAL_TZ).dt.date
        feats["local_hour"] = feats["period"].dt.tz_convert(LOCAL_TZ).dt.hour
        state["features"] = feats
    else:
        state["features"] = None
    yield


app = FastAPI(title="Phoenix Grid Forecast", lifespan=lifespan)


class HourForecast(BaseModel):
    hour_local: int
    period_utc: str
    predicted_mw: float
    actual_mw: float | None = None


class ForecastResponse(BaseModel):
    utility: str
    day: date
    model_trained_through_utc: str
    forecasts: list[HourForecast]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return state["info"]


@app.get("/predict", response_model=ForecastResponse)
def predict(utility: Literal["AZPS", "SRP"], day: date):
    f = state["features"]
    if f is None:
        raise HTTPException(status_code=503, detail="Replay data is not available in this deployment")
    rows = f[(f["respondent"] == utility) & (f["local_date"] == day)].sort_values("period")
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No feature data for {utility} on {day}")

    preds = state["models"][utility].predict(rows[FEATURES])
    return ForecastResponse(
        utility=utility,
        day=day,
        model_trained_through_utc=state["info"][utility]["trained_through_utc"],
        forecasts=[
            HourForecast(hour_local=int(h), period_utc=p.isoformat(),
                         predicted_mw=round(float(y), 1), actual_mw=float(a))
            for h, p, y, a in zip(rows["local_hour"], rows["period"], preds, rows["demand_mw"])
        ],
    )


class LiveForecastResponse(BaseModel):
    utility: str
    made_at_utc: str
    latest_actual_utc: str
    note: str
    forecasts: list[HourForecast]


def get_live_inputs() -> dict:
    """Fetch fresh data, but reuse it for 30 minutes so we don't hammer EIA and Open-Meteo."""
    now = datetime.now(timezone.utc)
    cached = state.get("live")
    if cached and (now - cached["at"]).total_seconds() < CACHE_MINUTES * 60:
        return cached
    state["live"] = {"at": now, "demand": fetch_recent_demand(), "weather": fetch_weather_forecast()}
    return state["live"]


def log_predictions(utility, made_at, rows, preds) -> None:
    """Save every forecast, so we can later compare it with what actually happened."""
    out = pd.DataFrame({
        "period_utc": rows["period"].reset_index(drop=True),
        "predicted_mw": preds.round(1),
    })
    out["made_at_utc"] = pd.Timestamp(made_at)
    out["utility"] = utility
    out["model_trained_through_utc"] = pd.Timestamp(state["info"][utility]["trained_through_utc"])
    save_forecasts(out[COLUMNS], LOG_PATH)


@app.get("/forecast", response_model=LiveForecastResponse)
def forecast(utility: Literal["AZPS", "SRP"]):
    try:
        live = get_live_inputs()
        validate_live_inputs(live["demand"], live["weather"], utility)
        rows, last = build_live_features(utility, live["demand"], live["weather"])
        check_fresh(last, datetime.now(timezone.utc))
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream data source failed: {e}")
    except DataValidationError as e:
        state.pop("live", None)  # don't keep reusing data that failed its checks
        raise HTTPException(status_code=503, detail=f"Input data failed a check: {e}")
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    preds = state["models"][utility].predict(rows[FEATURES])
    made_at = datetime.now(timezone.utc)
    log_predictions(utility, made_at, rows, preds)

    return LiveForecastResponse(
        utility=utility,
        made_at_utc=made_at.isoformat(),
        latest_actual_utc=last.isoformat(),
        note="Next 24 hours after the newest published demand hour. Uses forecast weather as input.",
        forecasts=[HourForecast(hour_local=int(h), period_utc=p.isoformat(), predicted_mw=round(float(y), 1))
                   for h, p, y in zip(rows["hour"], rows["period"], preds)],
    )