"""Propose a better model: train a challenger on newer data and compare it with the champion.

Run by the Retrain workflow on GitHub, or by hand (DATABASE_URL must be set):
    python -m src.retraining.retrain [--holdout-days 28] [--min-new-days 14] [--dry-run]
For each utility it reads the champion from the registry, trains a challenger on everything before the
newest 28 days, and scores both on those 28 days, which neither has seen. The decision rules live in
compare.py. A winner is stored as a CANDIDATE and nothing else changes: only the promote command, run
from your laptop, can make a model live. A comparison with less than 14 days of new data is skipped, since
the challenger would be nearly the same model as the champion.
"""

import argparse
import io
import os

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv

from src.api.model_store import UTILITIES
from src.features.build import FEATURES
from src.retraining.compare import decide, score
from src.retraining.registry import (
    add_model,
    get_model,
    log_run,
    next_version,
    versions_of,
)
from src.retraining.training_data import load_table

HOLDOUT_DAYS = 28
MIN_NEW_DAYS = 14


def train_lightgbm(rows: pd.DataFrame, target: pd.Series):
    """The same recipe as train_final.py. Imported here so the rest of this file needs no LightGBM."""
    import lightgbm as lgb

    from src.models.params import PARAMS

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(rows, target)
    return model


def plan_windows(latest_period, champion_trained_through, holdout_days=HOLDOUT_DAYS,
                 min_new_days=MIN_NEW_DAYS) -> dict:
    """The holdout is the newest `holdout_days` of data. The days between the champion's cutoff and the
    holdout are the new data the challenger gets to learn from."""
    holdout_end = pd.Timestamp(latest_period) + pd.Timedelta(hours=1)
    holdout_start = holdout_end - pd.Timedelta(days=holdout_days)
    new_days = (holdout_start - pd.Timestamp(champion_trained_through)) / pd.Timedelta(days=1)
    return {"holdout_start": holdout_start, "holdout_end": holdout_end, "new_data_days": new_days,
            "enough_new_data": new_days >= min_new_days}


def evaluate_utility(rows: pd.DataFrame, champion, trained_through, train_fn, features,
                     holdout_days=HOLDOUT_DAYS, min_new_days=MIN_NEW_DAYS) -> dict:
    """Compare a freshly trained challenger with the champion for one utility.

    Returns the windows, a verdict ('promote', 'keep' or 'skipped'), the reasons, the metrics table, and
    the challenger model when it should be kept as a candidate.
    """
    g = rows.dropna(subset=[*features, "demand_mw", "lag_24"])
    windows = plan_windows(g["period"].max(), trained_through, holdout_days, min_new_days)
    result = {**windows, "table": None, "challenger": None, "training_rows": 0, "trained_through": None}
    if not windows["enough_new_data"]:
        if windows["new_data_days"] <= 0:
            reason = ("the champion has already seen part of the holdout, "
                      "so a fair comparison is not possible yet")
        else:
            reason = (f"only {windows['new_data_days']:.1f} days of data are newer than the champion; "
                      f"{min_new_days} are needed for a meaningful comparison")
        return {**result, "verdict": "skipped", "reasons": [reason]}

    train = g[g["period"] < windows["holdout_start"]]
    holdout = g[(g["period"] >= windows["holdout_start"]) & (g["period"] < windows["holdout_end"])]
    if train.empty or holdout.empty:
        return {**result, "verdict": "skipped", "reasons": ["the training or holdout window has no data"]}

    challenger = train_fn(train[features], train["demand_mw"])
    table = score(holdout, {"champion": champion.predict(holdout[features]),
                            "challenger": challenger.predict(holdout[features]),
                            "copy_yesterday": holdout["lag_24"].to_numpy()})
    decision = decide(table)
    return {**result, "table": table, "reasons": decision["reasons"],
            "verdict": "promote" if decision["promote"] else "keep",
            "challenger": challenger if decision["promote"] else None,
            "training_rows": len(train), "trained_through": train["period"].max()}


def store_candidate(conn, utility: str, result: dict, metrics: dict) -> str:
    """Save the winning challenger in the registry under the next free version name."""
    version = next_version(versions_of(conn, utility))
    buffer = io.BytesIO()
    joblib.dump(result["challenger"], buffer)
    details = {**metrics, "champion_version": result["champion_version"],
               "holdout_start": str(result["holdout_start"]), "holdout_end": str(result["holdout_end"])}
    add_model(conn, utility, version, "candidate", result["trained_through"], result["training_rows"],
              buffer.getvalue(), details)
    return version


def run(conn, table: pd.DataFrame, train_fn, features, dry_run: bool = False,
        holdout_days=HOLDOUT_DAYS, min_new_days=MIN_NEW_DAYS) -> dict:
    """Evaluate every utility, store any winning candidate, and log each comparison."""
    results = {}
    for utility in UTILITIES:
        champion_row = get_model(conn, utility)
        if champion_row is None:
            raise ValueError(
                f"{utility} has no champion in the registry; run the registry seed command first")
        champion = joblib.load(io.BytesIO(champion_row["model_file"]))
        result = evaluate_utility(table[table["respondent"] == utility], champion,
                                  champion_row["trained_through_utc"], train_fn, features,
                                  holdout_days, min_new_days)
        result["champion_version"], result["candidate_version"] = champion_row["version"], None

        metrics, hours = {}, None
        if result["table"] is not None:
            by_model = result["table"].set_index("model")["MAPE_pct"]
            hours = int(result["table"]["hours"].iloc[0])
            metrics = {"champion_mape": float(by_model["champion"]),
                       "challenger_mape": float(by_model["challenger"]),
                       "baseline_mape": float(by_model["copy_yesterday"])}
        if result["verdict"] == "promote":
            result["candidate_version"] = ("(dry run)" if dry_run
                                           else store_candidate(conn, utility, result, metrics))
        if not dry_run:
            log_run(conn, {"utility": utility, "champion_version": result["champion_version"],
                           "holdout_start": result["holdout_start"], "holdout_end": result["holdout_end"],
                           "holdout_hours": hours, "new_data_days": float(result["new_data_days"]),
                           "verdict": result["verdict"], "reasons": result["reasons"],
                           "candidate_version": result["candidate_version"], **metrics})
        results[utility] = result
    return results


def to_markdown(results: dict, now) -> str:
    """The report for the GitHub run's summary page and for the issue that announces a candidate."""
    lines = [f"## Retrain check, {now:%Y-%m-%d %H:%M} UTC", ""]
    for utility, r in results.items():
        label = {"promote": f"CANDIDATE READY ({r['candidate_version']})",
                 "keep": "KEEP THE CHAMPION", "skipped": "SKIPPED"}[r["verdict"]]
        if r["new_data_days"] > 0:
            overlap = f"with {r['new_data_days']:.1f} days of data the champion never saw"
        else:
            overlap = "and the champion's training already covers part of it"
        lines += [f"### {utility}: {label}", "",
                    (f"Champion {r['champion_version']}. Holdout {r['holdout_start']:%Y-%m-%d} to "
                    f"{r['holdout_end']:%Y-%m-%d}, {overlap}."), ""]
        if r["table"] is not None:
            lines += ["| Model | Hours | MAPE % | MAE MW | Peak MAPE % |", "|---|---|---|---|---|"]
            lines += [f"| {t.model} | {t.hours} | {t.MAPE_pct:.2f} | {t.MAE_MW:.0f} | "
                      f"{t.peak_MAPE_pct:.2f} |" for t in r["table"].itertuples()]
            lines.append("")
        lines += [f"- {reason}" for reason in r["reasons"]]
        if r["verdict"] == "promote":
            version = r["candidate_version"]
            command = f"python -m src.retraining.promote --utility {utility} --version {version}"
            lines += ["", f"To put it live, run from your laptop: `{command}`"]
        lines.append("")
    return "\n".join(lines)


def candidate_list(results: dict) -> str:
    """For example 'SRP:v2,AZPS:v3', or an empty string when nothing won."""
    winners = [f"{u}:{r['candidate_version']}" for u, r in results.items() if r["verdict"] == "promote"]
    return ",".join(winners)


def write_github_files(report: str, results: dict, report_path: str = "retrain_report.md") -> None:
    """On GitHub: add the report to the run summary and tell later steps whether a candidate exists."""
    summary, output = os.getenv("GITHUB_STEP_SUMMARY"), os.getenv("GITHUB_OUTPUT")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"candidates={candidate_list(results)}\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train a challenger and compare it with the champion.")
    parser.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
    parser.add_argument("--min-new-days", type=int, default=MIN_NEW_DAYS)
    parser.add_argument("--dry-run", action="store_true", help="compare, but store and log nothing")
    args = parser.parse_args(argv)

    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Put it in .env or in the environment.")

    table = load_table(url)
    with psycopg.connect(url, connect_timeout=10) as conn:
        results = run(conn, table, train_lightgbm, FEATURES, args.dry_run,
                      args.holdout_days, args.min_new_days)
    report = to_markdown(results, pd.Timestamp.now(tz="UTC"))
    print(report)
    if not args.dry_run:
        write_github_files(report, results)


if __name__ == "__main__":
    main()