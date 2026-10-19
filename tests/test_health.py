import numpy as np
import pandas as pd

from src.monitoring.health import evaluate, has_failures, to_markdown

NOW = pd.Timestamp("2026-10-27", tz="UTC")
END = "2026-10-25"  # the newest graded hour is normally 2 to 3 days old


def scores(end, days, utility="AZPS", model_err=0.04, naive_err=0.06, bucket="6_12h", bias=0.0):
    """Hourly graded rows ending at `end`. Errors alternate in sign, so the bias stays near `bias`."""
    period = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=days * 24, freq="h")
    sign = np.where(np.arange(len(period)) % 2 == 0, 1.0, -1.0)
    actual = 5000.0
    return pd.DataFrame({"utility": utility, "bucket": bucket, "period_utc": period, "actual_mw": actual,
                         "predicted_mw": actual * (1 - bias + sign * model_err),
                         "naive_24h": actual * (1 + sign * naive_err)})


def statuses(result):
    return {name: status for name, status, _ in result["checks"]}


def details(result):
    return {name: detail for name, _, detail in result["checks"]}


def test_a_model_that_beats_copy_yesterday_is_ok():
    [r] = evaluate(scores(END, 10), NOW)
    assert statuses(r)["model_vs_naive"] == "ok"
    assert statuses(r)["bias"] == "ok" and statuses(r)["pipeline_stalled"] == "ok"
    assert not has_failures([r])


def test_a_model_worse_than_copy_yesterday_turns_the_run_red():
    [r] = evaluate(scores(END, 10, model_err=0.08, naive_err=0.05), NOW)
    assert statuses(r)["model_vs_naive"] == "fail" and has_failures([r])
    assert "model 8.00% vs copy-yesterday 5.00%" in details(r)["model_vs_naive"]


def test_too_little_data_skips_instead_of_judging():
    [r] = evaluate(scores(END, 2), NOW)  # 48 graded hours, fewer than the 72 needed
    assert statuses(r)["enough_data"] == "skip" and "model_vs_naive" not in statuses(r)
    assert not has_failures([r])
    assert "only 48" in details(r)["enough_data"]


def test_under_1h_rows_do_not_count_toward_the_judgment():
    mixed = pd.concat([scores(END, 10), scores(END, 10, model_err=0.5, bucket="under_1h")])
    [r] = evaluate(mixed, NOW)
    assert statuses(r)["model_vs_naive"] == "ok"                   # the terrible nowcast rows are ignored
    assert "under_1h" in set(r["buckets"]["bucket"])               # but they still show in the lead-time table


def test_rows_older_than_the_week_do_not_count_for_the_week():
    mixed = pd.concat([scores("2026-10-12", 10, model_err=0.5), scores(END, 10)])
    [r] = evaluate(mixed, NOW)
    assert abs(r["model_mape_7d"] - 4.0) < 0.01       # only the recent, good hours
    assert r["model_mape_28d"] > 20                   # the 28-day window does include the old bad ones
    assert statuses(r)["getting_worse"] == "ok"       # the last week is better than the month, not worse


def test_consistent_bias_warns_but_does_not_turn_the_run_red():
    [r] = evaluate(scores(END, 10, model_err=0.02, naive_err=0.10, bias=0.06), NOW)
    assert statuses(r)["bias"] == "warn" and r["bias_pct_7d"] > 5
    assert not has_failures([r])


def test_a_recent_slide_in_accuracy_warns():
    older = scores("2026-10-19", 15, model_err=0.03, naive_err=0.10)
    recent = scores(END, 5, model_err=0.06, naive_err=0.10)
    [r] = evaluate(pd.concat([older, recent]), NOW)
    assert statuses(r)["getting_worse"] == "warn"
    assert statuses(r)["model_vs_naive"] == "ok" and not has_failures([r])


def test_a_stalled_pipeline_turns_the_run_red():
    [r] = evaluate(scores("2026-10-20", 10), NOW)  # newest graded hour is 7 days old
    assert statuses(r)["pipeline_stalled"] == "fail" and has_failures([r])
    [ok] = evaluate(scores("2026-10-24", 10), NOW)  # 3 days old is normal because of the 48-hour wait
    assert statuses(ok)["pipeline_stalled"] == "ok"


def test_each_utility_is_judged_on_its_own():
    both = pd.concat([scores(END, 10, "AZPS", model_err=0.08, naive_err=0.05), scores(END, 10, "SRP")])
    results = evaluate(both, NOW)
    verdicts = {r["utility"]: statuses(r)["model_vs_naive"] for r in results}
    assert verdicts == {"AZPS": "fail", "SRP": "ok"}


def test_no_scores_at_all_is_not_a_failure():
    results = evaluate(scores(END, 10).iloc[0:0], NOW)
    assert results == [] and not has_failures(results)
    assert "No graded forecasts yet" in to_markdown(results, NOW)


def test_the_report_names_every_check_and_the_verdict():
    results = evaluate(scores(END, 10, model_err=0.08, naive_err=0.05), NOW)
    report = to_markdown(results, NOW)
    assert "### AZPS: FAIL" in report and "| model_vs_naive | FAIL |" in report
    assert "Last 7 days by lead time" in report and "6_12h" in report
    assert report.isascii()  # no emoji, so it prints safely in any terminal


def test_a_healthy_week_is_ok_even_without_28_days_of_history():
    # Only 10 days exist, so the optional 28-day comparison is skipped. That must not hide a good verdict.
    report = to_markdown(evaluate(scores(END, 10), NOW), NOW)
    assert "### AZPS: OK" in report and "| getting_worse | SKIP |" in report


def test_too_little_data_says_so_in_the_headline():
    assert "### AZPS: NOT ENOUGH DATA" in to_markdown(evaluate(scores(END, 2), NOW), NOW)