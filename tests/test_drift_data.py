import pandas as pd
import pytest

from src.monitoring.drift_data import (
    DRIFT_COLUMNS,
    FEATURE_COLUMNS,
    MODEL,
    MODELS,
    build_table,
    error_table,
    make_windows,
    shift_table,
)


def predictions(periods, utility="SRP", demand=5000.0, predicted=5100.0):
    periods = pd.to_datetime(periods, utc=True)
    df = pd.DataFrame({"period": periods, "respondent": utility, "demand_mw": demand})
    for m in MODELS:
        df[m] = predicted
    return df


def features(periods, utility="SRP"):
    periods = pd.to_datetime(periods, utc=True)
    df = pd.DataFrame({"period": periods, "respondent": utility})
    for c in FEATURE_COLUMNS:
        df[c] = 20.0
    return df


def test_the_table_joins_on_hour_and_utility_and_adds_the_percent_error():
    hours = ["2026-04-10 12:00", "2026-04-10 13:00"]
    table = build_table(predictions(hours, demand=5000.0, predicted=5250.0), features(hours))
    assert len(table) == 2
    assert (table["abs_pct_error"] == 5.0).all()  # |5000 - 5250| / 5000


def test_hours_without_matching_features_are_dropped_not_guessed():
    preds = predictions(["2026-04-10 12:00", "2026-04-10 13:00"])
    table = build_table(preds, features(["2026-04-10 12:00"]))
    assert list(table["period"].dt.strftime("%H")) == ["12"]


def test_missing_feature_columns_give_a_clear_error():
    with pytest.raises(ValueError, match="cdh"):
        build_table(predictions(["2026-04-10 12:00"]), features(["2026-04-10 12:00"]).drop(columns="cdh"))


def test_windows_use_phoenix_months_not_utc_months():
    # 03:00 UTC on 1 April is still 8 PM on 31 March in Phoenix, so it belongs to March: outside the window.
    # 03:00 UTC on 1 July is 8 PM on 30 June in Phoenix, so it belongs to June: inside the window.
    hours = ["2025-04-01 03:00", "2025-04-01 08:00", "2025-07-01 03:00", "2025-07-01 08:00",
             "2026-05-15 12:00", "2026-01-15 12:00"]
    table = build_table(predictions(hours), features(hours))
    reference, current = make_windows(table, "SRP")
    assert list(reference["period"].dt.strftime("%m-%d %H")) == ["04-01 08", "07-01 03"]
    assert list(current["period"].dt.strftime("%m-%d")) == ["05-15"]  # January is in neither window


def test_other_utilities_are_not_mixed_in():
    hours = ["2025-05-10 12:00"]
    table = pd.concat([build_table(predictions(hours, "SRP"), features(hours, "SRP")),
                       build_table(predictions(hours, "AZPS"), features(hours, "AZPS"))])
    reference, _ = make_windows(table, "SRP")
    assert set(reference["respondent"]) == {"SRP"} and len(reference) == 1


def test_shift_table_reports_the_percent_change_and_the_move_in_standard_deviations():
    reference = pd.DataFrame({c: [10.0, 20.0, 30.0] for c in DRIFT_COLUMNS})  # mean 20, std 10
    current = pd.DataFrame({c: [30.0, 40.0, 50.0] for c in DRIFT_COLUMNS})    # mean 40
    row = shift_table(reference, current).set_index("column").loc["demand_mw"]
    assert row["reference_mean"] == 20.0 and row["current_mean"] == 40.0
    assert row["change_pct"] == 100.0 and row["shift_in_std"] == 2.0


def test_shift_table_survives_a_column_that_did_not_vary():
    flat = pd.DataFrame({c: [5.0, 5.0, 5.0] for c in DRIFT_COLUMNS})
    row = shift_table(flat, flat).set_index("column").loc["demand_mw"]
    assert row["change_pct"] == 0.0 and pd.isna(row["shift_in_std"])  # no spread, so no standard-deviation move


def test_error_table_shows_before_after_and_how_many_times_worse():
    reference = pd.DataFrame({"demand_mw": [100.0, 100.0], **{m: [96.0, 104.0] for m in MODELS}})  # 4% error
    current = pd.DataFrame({"demand_mw": [100.0, 100.0], **{m: [88.0, 112.0] for m in MODELS}})    # 12% error
    row = error_table(reference, current).set_index("model").loc[MODEL]
    assert abs(row["reference_mape_pct"] - 4.0) < 1e-9 and abs(row["current_mape_pct"] - 12.0) < 1e-9
    assert abs(row["times_worse"] - 3.0) < 1e-9