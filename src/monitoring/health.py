"""Weekly health check on the graded live forecasts (the forecast_scores table).

Pure functions: no database and no network. Judgments use the headline rows only (lead time of
1 hour or more, the same rule the scoring uses) and compare the model with copy-yesterday on the
same hours. Statuses: ok, warn (shown in the report), fail (turns the run red), skip (not enough
data to judge).
"""

import pandas as pd

from src.models.baselines import mape

WINDOW_DAYS = 7
HISTORY_DAYS = 28
MIN_HOURS_WINDOW = 72      # distinct graded hours (3 days) needed before the last week is judged
MIN_HOURS_HISTORY = 288    # 12 days of graded hours needed before "getting worse" is judged
WORSE_FACTOR = 1.25        # warn when the last week is this much worse than the last 28 days
BIAS_LIMIT_PCT = 3.0       # warn when the model is consistently this far too low or too high
STALLED_AFTER_DAYS = 4     # the newest graded hour is normally 2 to 3 days old (48-hour wait)
REQUIRED_COLUMNS = ["utility", "bucket", "period_utc", "predicted_mw", "actual_mw", "naive_24h"]


def _in_window(g: pd.DataFrame, now: pd.Timestamp, days: int) -> pd.DataFrame:
    return g[(g["period_utc"] > now - pd.Timedelta(days=days)) & (g["period_utc"] <= now)]


def _bias_pct(g: pd.DataFrame) -> float:
    """Average signed error as a percent of actual demand. Positive means the model is too low."""
    return float(((g["actual_mw"] - g["predicted_mw"]) / g["actual_mw"]).mean() * 100)


def _by_bucket(g: pd.DataFrame) -> pd.DataFrame:
    rows = [{"bucket": str(b), "graded_rows": len(x),
             "model_mape_pct": mape(x["actual_mw"], x["predicted_mw"]),
             "naive_mape_pct": mape(x["actual_mw"], x["naive_24h"])}
            for b, x in g.groupby(g["bucket"].astype(str), sort=False)]
    return pd.DataFrame(rows)


def check_utility(utility: str, g: pd.DataFrame, now: pd.Timestamp) -> dict:
    """Run every check for one utility. `g` holds that utility's graded rows."""
    head = g[g["bucket"].astype(str) != "under_1h"]
    w7, w28 = _in_window(head, now, WINDOW_DAYS), _in_window(head, now, HISTORY_DAYS)
    hours7, hours28 = w7["period_utc"].nunique(), w28["period_utc"].nunique()
    checks = []

    age_days = (now - g["period_utc"].max()) / pd.Timedelta(days=1)
    if age_days > STALLED_AFTER_DAYS:
        checks.append(("pipeline_stalled", "fail",
                       f"newest graded hour is {age_days:.1f} days old (normally 2 to 3)"))
    else:
        checks.append(("pipeline_stalled", "ok", f"newest graded hour is {age_days:.1f} days old"))

    result = {"utility": utility, "hours_7d": hours7, "hours_28d": hours28,
              "buckets": _by_bucket(_in_window(g, now, WINDOW_DAYS))}
    if hours7 < MIN_HOURS_WINDOW:
        checks.append(("enough_data", "skip",
                       f"only {hours7} graded hours in the last {WINDOW_DAYS} days; need {MIN_HOURS_WINDOW}"))
    else:
        model, naive = mape(w7["actual_mw"], w7["predicted_mw"]), mape(w7["actual_mw"], w7["naive_24h"])
        result.update(model_mape_7d=model, naive_mape_7d=naive)
        verdict = "fail" if model > naive else "ok"
        checks.append(("model_vs_naive", verdict,
                       f"model {model:.2f}% vs copy-yesterday {naive:.2f}% over {WINDOW_DAYS} days"))

        bias = _bias_pct(w7)
        result["bias_pct_7d"] = bias
        checks.append(("bias", "warn" if abs(bias) > BIAS_LIMIT_PCT else "ok",
                       f"{bias:+.1f}% (positive means too low; warn beyond {BIAS_LIMIT_PCT:.0f}%)"))

        if hours28 >= MIN_HOURS_HISTORY:
            baseline = mape(w28["actual_mw"], w28["predicted_mw"])
            result["model_mape_28d"] = baseline
            checks.append(("getting_worse", "warn" if model > WORSE_FACTOR * baseline else "ok",
                           f"last {WINDOW_DAYS} days {model:.2f}% vs last {HISTORY_DAYS} days {baseline:.2f}%"))
        else:
            checks.append(("getting_worse", "skip",
                           f"only {hours28} graded hours in {HISTORY_DAYS} days; need {MIN_HOURS_HISTORY}"))
    result["checks"] = checks
    return result


def evaluate(scores: pd.DataFrame, now: pd.Timestamp) -> list:
    """One result per utility. An empty table gives an empty list."""
    if scores.empty:
        return []
    s = scores[REQUIRED_COLUMNS].copy()
    s["period_utc"] = pd.to_datetime(s["period_utc"], utc=True)
    return [check_utility(u, g, now) for u, g in s.groupby("utility", sort=True)]


def has_failures(results: list) -> bool:
    return any(status == "fail" for r in results for _, status, _ in r["checks"])


def _overall(checks: list) -> str:
    statuses = {status for _, status, _ in checks}
    if "fail" in statuses:
        return "FAIL"
    if "warn" in statuses:
        return "WARN"
    # Only the week itself being unjudgeable counts. A missing 28-day history is just a skipped detail.
    return "NOT ENOUGH DATA" if any(name == "enough_data" for name, _, _ in checks) else "OK"


def to_markdown(results: list, now: pd.Timestamp) -> str:
    """A plain-text report (no emoji, so it prints safely in any terminal)."""
    lines = [f"## Forecast health check, {now:%Y-%m-%d %H:%M} UTC", ""]
    if not results:
        return "\n".join(lines + ["No graded forecasts yet, so there is nothing to check."])
    for r in results:
        lines += [f"### {r['utility']}: {_overall(r['checks'])}", "",
                  "| Check | Status | Detail |", "|---|---|---|"]
        lines += [f"| {name} | {status.upper()} | {detail} |" for name, status, detail in r["checks"]]
        if len(r["buckets"]):
            lines += ["", f"Last {WINDOW_DAYS} days by lead time:", "",
                      "| Lead time | Graded rows | Model MAPE % | Copy-yesterday MAPE % |", "|---|---|---|---|"]
            lines += [f"| {b.bucket} | {b.graded_rows} | {b.model_mape_pct:.2f} | {b.naive_mape_pct:.2f} |"
                      for b in r["buckets"].itertuples()]
        lines.append("")
    return "\n".join(lines)