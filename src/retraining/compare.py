"""Champion-versus-challenger comparison, as plain functions.

The champion is the model in use. The challenger is trained on newer data. Both are scored on the same
holdout hours, which neither has seen, and the challenger is promoted only if it clearly wins.
There is no LightGBM here: the trainer is passed in, so every rule can be tested with a tiny fake model.
"""

import pandas as pd

from src.models.baselines import mae, mape, peak_mask

MARGIN = 0.05     # the challenger's error must be at least 5% lower (relative) than the champion's
MIN_HOURS = 500   # about three weeks of holdout


def split_windows(g: pd.DataFrame, champion_through, challenger_through, holdout_end):
    """(champion training rows, challenger training rows, holdout rows). Boundaries are exclusive ends.

    The holdout starts exactly where the challenger's training stops, so neither model has seen it.
    """
    if not champion_through <= challenger_through < holdout_end:
        raise ValueError("need champion_through <= challenger_through < holdout_end")
    return (g[g["period"] < champion_through],
            g[g["period"] < challenger_through],
            g[(g["period"] >= challenger_through) & (g["period"] < holdout_end)])


def score(holdout: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """One row per contestant, all scored on exactly the same holdout hours."""
    pk = peak_mask(holdout)
    y = holdout["demand_mw"]
    rows = []
    for name, pred in predictions.items():
        p = pd.Series(pred, index=holdout.index)
        rows.append({"model": name, "hours": len(holdout),
                     "MAE_MW": mae(y, p), "MAPE_pct": mape(y, p),
                     "peak_hours": int(pk.sum()), "peak_MAPE_pct": mape(y[pk], p[pk])})
    return pd.DataFrame(rows)


def decide(table: pd.DataFrame, margin: float = MARGIN, min_hours: int = MIN_HOURS) -> dict:
    """Promote only if every condition holds. Returns the verdict and a plain-words line per condition."""
    mape_by_model = table.set_index("model")["MAPE_pct"]
    hours = int(table["hours"].iloc[0])
    champion, challenger = mape_by_model["champion"], mape_by_model["challenger"]
    baseline = mape_by_model["copy_yesterday"]
    checks = [
        (hours >= min_hours, f"holdout has {hours} hours (needs at least {min_hours})"),
        (challenger < champion and challenger <= (1 - margin) * champion,  # a tie is never an improvement
         (f"challenger {challenger:.2f}% vs champion {champion:.2f}% "
         f"(needs to be at least {margin:.0%} lower)")),
        (challenger < baseline, f"challenger {challenger:.2f}% vs copy-yesterday {baseline:.2f}%"),
    ]
    return {"promote": all(ok for ok, _ in checks),
            "reasons": [("PASS: " if ok else "FAIL: ") + text for ok, text in checks]}


def run_experiment(df: pd.DataFrame, utility: str, champion_through, challenger_through, holdout_end,
                   train_fn, features: list) -> dict:
    """Train both models on their own windows and compare them on the shared holdout.

    `df` needs period, respondent, demand_mw, lag_24 and every column in `features`.
    `train_fn(rows, target)` returns an object with .predict(rows).
    """
    g = df[df["respondent"] == utility].dropna(subset=[*features, "demand_mw", "lag_24"])
    champion_rows, challenger_rows, holdout = split_windows(
        g, champion_through, challenger_through, holdout_end)
    for name, part in [("champion training", champion_rows), ("challenger training", challenger_rows),
                       ("holdout", holdout)]:
        if part.empty:
            raise ValueError(f"{utility}: the {name} window has no data")

    champion = train_fn(champion_rows[features], champion_rows["demand_mw"])
    challenger = train_fn(challenger_rows[features], challenger_rows["demand_mw"])
    table = score(holdout, {"champion": champion.predict(holdout[features]),
                            "challenger": challenger.predict(holdout[features]),
                            "copy_yesterday": holdout["lag_24"].to_numpy()})
    return {"table": table, "decision": decide(table),
            "rows": {"champion": len(champion_rows), "challenger": len(challenger_rows),
                     "holdout": len(holdout)}}