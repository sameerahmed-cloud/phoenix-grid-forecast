import numpy as np
import pandas as pd

from src.models.score_live import (
    attach_actuals,
    prepare_forecasts,
    score_all,
    summarize,
    to_score_records,
)

FAR_FUTURE = pd.Timestamp("2027-01-01", tz="UTC")  # "now" far enough ahead that nothing is too fresh


def log(made_at, period, pred, utility="AZPS"):
    """Build a tiny forecast log. made_at/period/pred can each be a scalar or a list."""
    period = period if isinstance(period, list) else [period]
    n = len(period)
    pred = pred if isinstance(pred, list) else [pred] * n
    made_at = made_at if isinstance(made_at, list) else [made_at] * n
    return pd.DataFrame({"made_at_utc": made_at, "utility": utility,
                         "period_utc": period, "predicted_mw": pred})


def actuals_series(start, n, utility="AZPS", usable=None):
    """Hourly actuals where each value reveals its hour: 1000 + hours since `start`."""
    period = pd.date_range(start, periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"utility": utility, "period_utc": period, "demand_mw": 1000.0 + np.arange(n)})
    df["usable"] = True
    for ts in usable or []:
        df.loc[df["period_utc"] == pd.Timestamp(ts, tz="UTC"), "usable"] = False
    return df


def test_lead_time_buckets():
    f = log("2026-10-16 14:38", ["2026-10-16 14:00", "2026-10-16 20:00",
                                 "2026-10-16 23:00", "2026-10-17 13:00"], 5000.0)
    kept, _ = prepare_forecasts(f, FAR_FUTURE)
    got = dict(zip(kept["period_utc"].dt.strftime("%m-%d %H"), kept["bucket"].astype(str)))
    assert got == {"10-16 14": "under_1h", "10-16 20": "1_6h", "10-16 23": "6_12h", "10-17 13": "12h_plus"}
    assert kept.loc[0, "lead_h"] < 0  # the first hour had already started when the forecast was made


def test_duplicates_in_one_window_keep_the_earliest():
    target = "2026-10-17 12:00"
    f = pd.concat([log("2026-10-16 14:38", target, 5000.0),   # window 12:00-18:00
                   log("2026-10-16 15:10", target, 5100.0),   # same window: dropped
                   log("2026-10-16 18:05", target, 5200.0)])  # next window: kept
    kept, funnel = prepare_forecasts(f, FAR_FUTURE)
    assert sorted(kept["predicted_mw"]) == [5000.0, 5200.0]
    assert funnel["after_dedup"] == 2


def test_only_hours_at_least_48h_old_are_scored():
    now = pd.Timestamp("2026-10-20 12:00", tz="UTC")
    f = log("2026-10-17 00:00", ["2026-10-18 12:00", "2026-10-18 13:00"], 5000.0)
    kept, funnel = prepare_forecasts(f, now)
    assert list(kept["period_utc"].dt.strftime("%H")) == ["12"]  # exactly 48h old is in, 47h is out
    assert funnel["mature"] == 1


def test_broken_predictions_are_counted_not_scored():
    f = log("2026-10-16 02:00", ["2026-10-16 10:00", "2026-10-16 11:00",
                                 "2026-10-16 12:00", "2026-10-16 13:00"], [np.nan, 0.0, -5.0, 4800.0])
    kept, funnel = prepare_forecasts(f, FAR_FUTURE)
    assert funnel["broken_prediction"] == 3
    assert list(kept["predicted_mw"]) == [4800.0]


def test_baseline_is_the_actual_from_24_hours_earlier():
    actuals = actuals_series("2026-10-15 00:00", 72)            # through 2026-10-17 23:00
    kept, _ = prepare_forecasts(log("2026-10-17 00:00", "2026-10-17 12:00", 5000.0), FAR_FUTURE)
    scored, funnel = attach_actuals(kept, actuals)
    assert funnel["scored"] == 1
    assert scored.loc[0, "actual_mw"] == 1060.0                 # 60 hours after the start
    assert scored.loc[0, "naive_24h"] == 1036.0                 # the same hour one day earlier


def test_unusable_actual_or_baseline_drops_the_row_and_is_counted():
    kept, _ = prepare_forecasts(log("2026-10-17 00:00", "2026-10-17 12:00", 5000.0), FAR_FUTURE)
    bad_actual = actuals_series("2026-10-15 00:00", 72, usable=["2026-10-17 12:00"])
    _, f1 = attach_actuals(kept, bad_actual)
    assert (f1["no_usable_actual"], f1["no_usable_baseline"], f1["scored"]) == (1, 0, 0)
    bad_baseline = actuals_series("2026-10-15 00:00", 72, usable=["2026-10-16 12:00"])
    _, f2 = attach_actuals(kept, bad_baseline)
    assert (f2["no_usable_actual"], f2["no_usable_baseline"], f2["scored"]) == (0, 1, 0)


def test_other_utilities_do_not_mix():
    azps = actuals_series("2026-10-15 00:00", 72, "AZPS")
    srp = actuals_series("2026-10-15 00:00", 72, "SRP")
    srp["demand_mw"] += 90000  # a clearly different level, so any mixing would show
    actuals = pd.concat([azps, srp])
    kept, _ = prepare_forecasts(log("2026-10-17 00:00", "2026-10-17 12:00", 5000.0, "AZPS"), FAR_FUTURE)
    scored, _ = attach_actuals(kept, actuals)
    assert scored.loc[0, "actual_mw"] == 1060.0


def test_worked_example_numbers():
    # Predicted 5,000, actual 5,400, copy-yesterday 5,100: the baseline wins this hour.
    f = log("2026-10-16 02:00", "2026-10-16 22:00", 5000.0)    # lead time 20 hours
    actuals = pd.DataFrame({"utility": "AZPS",
                            "period_utc": pd.to_datetime(["2026-10-15 22:00", "2026-10-16 22:00"], utc=True),
                            "demand_mw": [5100.0, 5400.0], "usable": True})
    kept, _ = prepare_forecasts(f, FAR_FUTURE)
    scored, _ = attach_actuals(kept, actuals)
    table = summarize(scored)
    head = table[table["slice"] == "headline_1h_plus"].iloc[0]
    assert head["hours"] == 1
    assert (head["MAE_model"], head["MAE_naive"]) == (400.0, 300.0)
    assert abs(head["MAPE_model"] - 7.407) < 0.001 and abs(head["MAPE_naive"] - 5.556) < 0.001
    assert head["peak_hours"] == 0 and np.isnan(head["peak_MAPE_model"])  # October is outside the peak season
    assert (table["slice"] == "12h_plus").any()


def test_under_1h_hours_stay_out_of_the_headline():
    f = log("2026-10-16 14:38", ["2026-10-16 14:00", "2026-10-16 20:00"], 5000.0)
    actuals = actuals_series("2026-10-15 00:00", 72)
    kept, _ = prepare_forecasts(f, FAR_FUTURE)
    scored, _ = attach_actuals(kept, actuals)
    table = summarize(scored)
    assert table.loc[table["slice"] == "headline_1h_plus", "hours"].iloc[0] == 1
    assert table.loc[table["slice"] == "under_1h", "hours"].iloc[0] == 1


def test_score_all_chains_the_rules_and_reports_every_drop():
    f = pd.concat([log("2026-10-16 02:00", "2026-10-16 22:00", [5000.0]),
                   log("2026-10-16 02:30", "2026-10-16 22:00", [5050.0]),   # duplicate window
                   log("2026-10-16 02:00", "2026-10-16 23:00", [np.nan])])  # broken prediction
    actuals = pd.DataFrame({"utility": "AZPS",
                            "period_utc": pd.to_datetime(["2026-10-15 22:00", "2026-10-16 22:00"], utc=True),
                            "demand_mw": [5100.0, 5400.0], "usable": True})
    scored, funnel, table = score_all(f, actuals, FAR_FUTURE)
    assert funnel == {"logged": 3, "broken_prediction": 1, "after_dedup": 1, "mature": 1,
                      "no_usable_actual": 0, "no_usable_baseline": 0, "scored": 1}
    assert len(scored) == 1 and len(table) == 2  # the headline row plus the 12h_plus row


def test_nothing_mature_yet_returns_empty_results_without_crashing():
    f = log("2026-10-16 02:00", "2026-10-16 22:00", 5000.0)
    actuals = actuals_series("2026-10-15 00:00", 72)
    now = pd.Timestamp("2026-10-16 23:00", tz="UTC")  # the hour is only 1 hour old
    scored, funnel, table = score_all(f, actuals, now)
    assert funnel["mature"] == 0 and funnel["scored"] == 0
    assert scored.empty and table.empty


def test_score_records_are_plain_python_values_in_table_order():
    f = log("2026-10-16 02:00", "2026-10-16 22:00", 5000.0)
    f["model_trained_through_utc"] = "2026-10-08 00:00"
    actuals = pd.DataFrame({"utility": "AZPS",
                            "period_utc": pd.to_datetime(["2026-10-15 22:00", "2026-10-16 22:00"], utc=True),
                            "demand_mw": [5100.0, 5400.0], "usable": True})
    scored, _, _ = score_all(f, actuals, FAR_FUTURE)
    records = to_score_records(scored)
    assert len(records) == 1 and len(records[0]) == 9
    made_at, utility, period, trained, lead_h, bucket, predicted, actual, naive = records[0]
    assert made_at.tzinfo is not None and period.tzinfo is not None and trained.tzinfo is not None
    assert (utility, bucket, lead_h) == ("AZPS", "12h_plus", 20.0)
    assert (predicted, actual, naive) == (5000.0, 5400.0, 5100.0)
    assert all(type(v) is float for v in (lead_h, predicted, actual, naive))  # not numpy floats