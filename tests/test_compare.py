import numpy as np
import pandas as pd
import pytest

from src.retraining.compare import decide, run_experiment

FEATS = ["t", "lag_24"]


def T(date):
    return pd.Timestamp(date, tz="UTC")


def make_df(utility="SRP", before=5000.0, after=6000.0, shift_on="2026-04-15", baseline_factor=1.2):
    """Hourly demand that steps up on `shift_on`. Copy-yesterday (lag_24) is off by baseline_factor."""
    period = pd.date_range("2026-01-01", "2026-09-01", freq="h", inclusive="left", tz="UTC")
    demand = np.where(period >= T(shift_on), after, before)
    return pd.DataFrame({"period": period, "respondent": utility, "demand_mw": demand,
                         "lag_24": demand * baseline_factor, "t": np.arange(len(period))})


class MeanModel:
    """A stand-in model: it always predicts the average demand it was trained on."""

    def __init__(self, level):
        self.level = level

    def predict(self, rows):
        return np.full(len(rows), self.level)


def mean_trainer(rows, target):
    return MeanModel(float(target.mean()))


def experiment(df, trainer=mean_trainer, champion="2026-04-01", challenger="2026-06-01", end="2026-09-01"):
    return run_experiment(df, "SRP", T(champion), T(challenger), T(end), trainer, FEATS)


def mape_of(result, model):
    return float(result["table"].set_index("model").loc[model, "MAPE_pct"])


def test_a_challenger_that_saw_the_new_level_is_promoted():
    r = experiment(make_df())
    assert abs(mape_of(r, "champion") - 16.667) < 0.01    # trained before the step: predicts 5,000 for 6,000
    assert abs(mape_of(r, "challenger") - 11.48) < 0.01   # trained through May: part of the step is in it
    assert abs(mape_of(r, "copy_yesterday") - 20.0) < 1e-9
    assert r["decision"]["promote"] is True
    assert all(line.startswith("PASS") for line in r["decision"]["reasons"])


def test_a_short_holdout_blocks_promotion():
    r = experiment(make_df(), end="2026-06-11")  # only 240 hours
    assert r["decision"]["promote"] is False
    assert "FAIL: holdout has 240 hours" in r["decision"]["reasons"][0]


def test_two_identical_models_do_not_clear_the_margin():
    r = experiment(make_df(), trainer=lambda rows, target: MeanModel(5500.0))
    assert r["decision"]["promote"] is False
    assert r["decision"]["reasons"][1].startswith("FAIL")


def test_a_challenger_that_cannot_beat_copy_yesterday_is_not_promoted():
    r = experiment(make_df(baseline_factor=1.0))  # copy-yesterday is perfect here
    assert r["decision"]["promote"] is False
    assert r["decision"]["reasons"][1].startswith("PASS") and r["decision"]["reasons"][2].startswith("FAIL")


def test_neither_model_ever_sees_the_holdout():
    df = make_df()
    seen = []

    def spy(rows, target):
        seen.append(int(rows["t"].max()))
        return MeanModel(float(target.mean()))

    experiment(df, trainer=spy)
    def hours_before(date):
        return int((T(date) - df["period"].min()) / pd.Timedelta(hours=1))

    assert seen[0] < hours_before("2026-04-01")   # the champion stopped before its cutoff
    assert seen[1] < hours_before("2026-06-01")   # the challenger stopped before the holdout began


def test_the_cutoffs_must_be_in_order():
    with pytest.raises(ValueError, match="champion_through"):
        experiment(make_df(), champion="2026-07-01", challenger="2026-06-01")


def test_an_empty_window_gives_a_clear_error():
    with pytest.raises(ValueError, match="champion training window has no data"):
        experiment(make_df(), champion="2025-12-01")


def test_both_models_and_the_baseline_are_scored_on_the_same_hours():
    r = experiment(make_df())
    assert set(r["table"]["hours"]) == {r["rows"]["holdout"]}
    assert r["rows"]["holdout"] == 92 * 24  # June, July and August


def test_another_utility_does_not_leak_in():
    both = pd.concat([make_df("SRP"), make_df("AZPS", before=1.0, after=1.0)])
    assert mape_of(experiment(both), "challenger") == mape_of(experiment(make_df()), "challenger")


def test_rows_with_missing_inputs_are_dropped_from_every_window():
    df = make_df()
    holdout_rows = df.index[(df["period"] >= T("2026-06-01")) & (df["period"] < T("2026-06-02"))]
    df.loc[holdout_rows[:5], "lag_24"] = np.nan
    assert experiment(df)["rows"]["holdout"] == 92 * 24 - 5


def test_a_summer_holdout_reports_peak_hours():
    table = experiment(make_df())["table"]
    assert (table["peak_hours"] > 300).all() and table["peak_MAPE_pct"].notna().all()


def test_the_five_percent_margin_is_checked_exactly():
    def verdict(challenger_mape):
        table = pd.DataFrame({"model": ["champion", "challenger", "copy_yesterday"], "hours": 600,
                              "MAPE_pct": [4.0, challenger_mape, 5.0]})
        return decide(table)["promote"]

    assert verdict(3.79) is True    # more than 5% lower than 4.00
    assert verdict(3.81) is False   # not quite 5% lower


def test_a_tie_is_never_an_improvement_even_at_zero_error():
    table = pd.DataFrame({"model": ["champion", "challenger", "copy_yesterday"], "hours": 600,
                          "MAPE_pct": [0.0, 0.0, 5.0]})
    verdict = decide(table)
    assert verdict["promote"] is False and verdict["reasons"][1].startswith("FAIL")