"""Build the drift reports for the SRP data change of April 2026.

Run from the project folder, inside the MONITORING environment (it needs Evidently):
    python -m src.monitoring.drift_report
Reads reports/predictions.parquet (made by src.models.train_lgbm) and data/processed/features.parquet.
Writes an HTML report and a JSON file per utility into reports/drift/.
The data preparation lives in drift_data.py; this file only prints the tables and calls Evidently.
"""

from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.monitoring.drift_data import (
    CURRENT_MONTHS,
    DRIFT_COLUMNS,
    REFERENCE_MONTHS,
    build_table,
    error_table,
    make_windows,
    shift_table,
)

OUT_DIR = Path("reports/drift")


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame):
    """Compare the two windows column by column. The current data goes first, the reference second."""
    definition = DataDefinition(numerical_columns=DRIFT_COLUMNS)
    ref = Dataset.from_pandas(reference[DRIFT_COLUMNS].reset_index(drop=True), data_definition=definition)
    cur = Dataset.from_pandas(current[DRIFT_COLUMNS].reset_index(drop=True), data_definition=definition)
    return Report([DataDriftPreset(method="psi")]).run(cur, ref)


def main() -> None:
    predictions = pd.read_parquet("reports/predictions.parquet")
    features = pd.read_parquet("data/processed/features.parquet")
    table = build_table(predictions, features)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for utility in sorted(table["respondent"].unique()):
        reference, current = make_windows(table, utility)
        print(f"\n===== {utility}: {'+'.join(REFERENCE_MONTHS)} (reference, {len(reference)} hours) "
              f"vs {'+'.join(CURRENT_MONTHS)} (current, {len(current)} hours) =====")
        print("\nHow far each column moved:")
        print(shift_table(reference, current).round(2).to_string(index=False))
        print("\nModel error before and after:")
        print(error_table(reference, current).round(2).to_string(index=False))

        result = evidently_report(reference, current)
        result.save_html(str(OUT_DIR / f"drift_{utility}.html"))
        result.save_json(str(OUT_DIR / f"drift_{utility}.json"))
        print(f"\nSaved reports/drift/drift_{utility}.html and .json")


if __name__ == "__main__":
    main()