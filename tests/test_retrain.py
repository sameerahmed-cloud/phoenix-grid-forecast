import io
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest

from src.retraining import retrain as r

FEATS = ["t", "lag_24"]


def T(date):
    return pd.Timestamp(date, tz="UTC")


class MeanModel:
    """A stand-in model that always predicts one number. It must be importable so it can be reloaded."""

    def __init__(self, level):
        self.level = level

    def predict(self, rows):
        return np.full(len(rows), self.level)


def mean_trainer(rows, target):
    return MeanModel(float(target.mean()))


def make_table(utility="SRP", baseline_factor=1.2):
    """120 days of hourly demand that steps from 5,000 to 6,000 MW on 1 March. Copy-yesterday is 20% off."""
    period = pd.date_range("2026-01-01", periods=120 * 24, freq="h", tz="UTC")
    demand = np.where(period >= T("2026-03-01"), 6000.0, 5000.0)
    return pd.DataFrame({"period": period, "respondent": utility, "demand_mw": demand,
                         "lag_24": demand * baseline_factor, "t": np.arange(len(period))})


def evaluate(trained_through="2026-03-01", trainer=mean_trainer, champion_level=5000.0):
    return r.evaluate_utility(make_table(), MeanModel(champion_level), T(trained_through), trainer, FEATS)


def test_the_holdout_is_the_newest_28_days_and_new_data_is_what_lies_between():
    plan = r.plan_windows(T("2026-04-30 23:00"), T("2026-03-01"))
    assert plan["holdout_end"] == T("2026-05-01") and plan["holdout_start"] == T("2026-04-03")
    assert plan["new_data_days"] == 33.0 and plan["enough_new_data"] is True
    # the champion's cutoff falls inside the holdout, so there is no new data at all
    assert r.plan_windows(T("2026-04-30 23:00"), T("2026-04-20"))["enough_new_data"] is False


def test_a_challenger_that_learned_the_new_level_becomes_a_candidate():
    result = evaluate()
    assert result["verdict"] == "promote" and result["challenger"] is not None
    assert all(line.startswith("PASS") for line in result["reasons"])
    assert set(result["table"]["model"]) == {"champion", "challenger", "copy_yesterday"}
    assert set(result["table"]["hours"]) == {28 * 24}
    assert result["trained_through"] == T("2026-04-02 23:00")  # the hour before the holdout begins


def test_with_too_little_new_data_nothing_is_trained_and_the_comparison_is_skipped():
    def must_not_train(rows, target):
        raise AssertionError("no challenger should be trained when there is too little new data")

    result = evaluate(trained_through="2026-03-21", trainer=must_not_train)  # 13 days before the holdout
    assert result["verdict"] == "skipped" and "only 13.0 days" in result["reasons"][0]
    assert result["challenger"] is None and result["table"] is None


def test_a_champion_that_has_seen_part_of_the_holdout_cannot_be_compared_fairly():
    result = evaluate(trained_through="2026-04-20", trainer=lambda rows, target: 1 / 0)
    assert result["verdict"] == "skipped" and "already seen part of the holdout" in result["reasons"][0]


def test_a_challenger_that_is_no_better_is_not_kept():
    result = evaluate(trainer=lambda rows, target: MeanModel(5000.0))  # identical to the champion
    assert result["verdict"] == "keep" and result["challenger"] is None
    assert result["reasons"][1].startswith("FAIL")


def test_the_challenger_never_trains_on_the_holdout():
    seen = []

    def spy(rows, target):
        seen.append(int(rows["t"].max()))
        return MeanModel(float(target.mean()))

    result = evaluate(trainer=spy)
    holdout_start_t = int((result["holdout_start"] - T("2026-01-01")) / pd.Timedelta(hours=1))
    assert seen[0] < holdout_start_t


def registry_fakes(monkeypatch, champion_level=5000.0, with_champion=True, trained_through="2026-03-01"):
    """Replace the registry functions with recorders, so no database is needed."""
    calls = {"added": [], "logged": []}
    champion = io.BytesIO()
    joblib.dump(MeanModel(champion_level), champion)
    row = {"version": "v1", "trained_through_utc": T(trained_through), "model_file": champion.getvalue()}
    monkeypatch.setattr(r, "get_model", lambda conn, utility: row if with_champion else None)
    monkeypatch.setattr(r, "versions_of", lambda conn, utility: ["v1"])
    monkeypatch.setattr(r, "add_model", lambda *args: calls["added"].append(args))
    monkeypatch.setattr(r, "log_run", lambda conn, run: calls["logged"].append(run))
    return calls


def both_utilities():
    return pd.concat([make_table("AZPS"), make_table("SRP")], ignore_index=True)


def test_a_run_stores_each_winner_as_the_next_version_and_logs_every_comparison(monkeypatch):
    calls = registry_fakes(monkeypatch)
    results = r.run(object(), both_utilities(), mean_trainer, FEATS)
    assert {u: x["candidate_version"] for u, x in results.items()} == {"AZPS": "v2", "SRP": "v2"}
    assert [a[1:4] for a in calls["added"]] == [("AZPS", "v2", "candidate"), ("SRP", "v2", "candidate")]
    stored = joblib.load(io.BytesIO(calls["added"][0][6]))  # the stored file loads back as a working model
    assert float(stored.predict(pd.DataFrame({"t": [0]}))[0]) < 6000.0
    assert [x["verdict"] for x in calls["logged"]] == ["promote", "promote"]
    first = calls["logged"][0]
    assert first["candidate_version"] == "v2" and first["holdout_hours"] == 28 * 24


def test_a_dry_run_compares_but_stores_and_logs_nothing(monkeypatch):
    calls = registry_fakes(monkeypatch)
    results = r.run(object(), both_utilities(), mean_trainer, FEATS, dry_run=True)
    assert calls == {"added": [], "logged": []}
    assert results["SRP"]["candidate_version"] == "(dry run)"


def test_a_skipped_comparison_is_logged_without_a_candidate(monkeypatch):
    calls = registry_fakes(monkeypatch)
    r.run(object(), both_utilities(), mean_trainer, FEATS, min_new_days=40)  # 33 days is not enough
    assert calls["added"] == [] and {x["verdict"] for x in calls["logged"]} == {"skipped"}
    assert "champion_mape" not in calls["logged"][0]  # nothing was scored, so there is nothing to record


def test_a_missing_champion_points_to_the_seed_command(monkeypatch):
    registry_fakes(monkeypatch, with_champion=False)
    with pytest.raises(ValueError, match="seed"):
        r.run(object(), both_utilities(), mean_trainer, FEATS)


def test_the_report_names_each_verdict_and_gives_the_promote_command(monkeypatch):
    registry_fakes(monkeypatch)
    report = r.to_markdown(r.run(object(), both_utilities(), mean_trainer, FEATS), T("2026-05-01 09:00"))
    assert "### SRP: CANDIDATE READY (v2)" in report
    assert "python -m src.retraining.promote --utility SRP --version v2" in report
    assert "| challenger |" in report and report.isascii()
    skipped = r.to_markdown(r.run(object(), both_utilities(), mean_trainer, FEATS, min_new_days=40,
                                  dry_run=True), T("2026-05-01 09:00"))
    assert "### SRP: SKIPPED" in skipped and "promote --utility" not in skipped


def test_the_report_never_prints_a_negative_number_of_days(monkeypatch):
    registry_fakes(monkeypatch, trained_through="2026-04-20")  # the champion's cutoff is inside the holdout
    results = r.run(object(), both_utilities(), mean_trainer, FEATS, dry_run=True)
    report = r.to_markdown(results, T("2026-05-01 09:00"))
    assert "already covers part of it" in report and "never saw" not in report and "-" not in "".join(
        line for line in report.splitlines() if "days of data" in line)


def test_the_candidate_list_is_empty_when_nobody_won(monkeypatch):
    registry_fakes(monkeypatch)
    winners = r.run(object(), both_utilities(), mean_trainer, FEATS)
    assert r.candidate_list(winners) == "AZPS:v2,SRP:v2"
    losers = r.run(object(), both_utilities(), lambda rows, target: MeanModel(5000.0), FEATS)
    assert r.candidate_list(losers) == ""


def test_github_files_get_the_report_and_the_candidate_list(monkeypatch):
    registry_fakes(monkeypatch)
    results = r.run(object(), both_utilities(), mean_trainer, FEATS)
    folder = tempfile.mkdtemp()
    summary, output, report_file = (os.path.join(folder, name)
                                    for name in ("summary.md", "output.txt", "report.md"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", summary)
    monkeypatch.setenv("GITHUB_OUTPUT", output)
    r.write_github_files("REPORT", results, report_file)
    with open(output, encoding="utf-8") as f:
        assert f.read() == "candidates=AZPS:v2,SRP:v2\n"
    with open(summary, encoding="utf-8") as f, open(report_file, encoding="utf-8") as g:
        assert f.read() == g.read() == "REPORT\n"


def test_main_runs_end_to_end_as_a_dry_run(monkeypatch):
    registry_fakes(monkeypatch)

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")
    monkeypatch.setattr(r, "load_dotenv", lambda: None)
    monkeypatch.setattr(r, "load_table", lambda url: both_utilities())
    monkeypatch.setattr(r, "train_lightgbm", mean_trainer)
    monkeypatch.setattr(r, "FEATURES", FEATS)
    monkeypatch.setattr(r.psycopg, "connect", lambda url, connect_timeout=None: FakeConn())
    r.main(["--dry-run"])


def test_a_missing_database_url_stops_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(r, "load_dotenv", lambda: None)
    try:
        r.main([])
    except SystemExit as e:
        assert "DATABASE_URL" in str(e.code)
    else:
        raise AssertionError("the job should have stopped")