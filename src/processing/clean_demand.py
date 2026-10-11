"""Clean raw EIA demand: flag bad hours, repair short gaps, write a model-ready table."""

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/eia_2022-01-01_2026-10-08.parquet")
OUT = Path("data/processed/demand_clean.parquet")

SPIKE_RATIO = 2.0       # more than 2x the local median is treated as an error
DROP_RATIO = 0.5        # less than 0.5x the local median is treated as an error
MAX_REPAIR_HOURS = 3    # only bad runs this short get interpolated


def clean_one(s: pd.Series) -> pd.DataFrame:
    """Clean one utility's demand series (index = UTC timestamps)."""
    # Put the series on a complete hourly grid so missing hours become explicit NaN rows.
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))

    # Centered 25-hour rolling median: robust to spikes and adapts to season.
    median = s.rolling(25, center=True, min_periods=5).median()
    ratio = s / median

    reason = np.select(
        [s.isna(), s <= 0, ratio > SPIKE_RATIO, ratio < DROP_RATIO],
        ["missing", "non_positive", "spike", "collapse"],
        default="",
    )
    # Bad readings come in clusters, and the edge hours of a cluster often slip past the ratio test.
    # Also flag any hour directly before or after a flagged hour.
    core_bad = pd.Series(reason != "", index=s.index)
    neighbor = (core_bad.shift(1, fill_value=False) | core_bad.shift(-1, fill_value=False)) & ~core_bad
    reason = np.where(neighbor, "neighbor", reason)
    bad = pd.Series(reason != "", index=s.index)

    # Length of each consecutive run of bad hours, so we repair short runs only.
    run_id = (bad != bad.shift()).cumsum()
    run_len = bad.groupby(run_id).transform("sum")
    repairable = bad & (run_len <= MAX_REPAIR_HOURS)

    # Interpolating over the whole series would invent data across long outages.
    # Pandas' `limit` would also fill the first N hours of a long gap, which is wrong,
    # so we compute the interpolation and apply it only to repairable runs.
    masked = s.mask(bad)
    interpolated = masked.interpolate(method="time", limit_area="inside")
    final = masked.copy()
    final[repairable] = interpolated[repairable]

    return pd.DataFrame(
        {
            "demand_mw": final,
            "flag_reason": reason,
            "was_repaired": repairable,
            "usable": final.notna(),
        }
    )


def main() -> None:
    raw = pd.read_parquet(RAW)
    d = raw[raw["type"] == "D"]

    parts = []
    for resp, g in d.groupby("respondent"):
        out = clean_one(g.set_index("period")["value"].sort_index())
        out["respondent"] = resp
        parts.append(out.rename_axis("period").reset_index())
    clean = pd.concat(parts, ignore_index=True)

    flagged = clean[clean["flag_reason"] != ""]
    print("Flagged by reason:")
    print(flagged.groupby(["respondent", "flag_reason"]).size(), "\n")
    print("Repaired:", clean.groupby("respondent")["was_repaired"].sum().to_dict())
    print("Still unusable:", (~clean["usable"]).groupby(clean["respondent"]).sum().to_dict(), "\n")
    print("Flagged hours by UTC hour of day (07 = midnight in Phoenix):")
    print(flagged.groupby(flagged["period"].dt.hour).size().to_string(), "\n")
    print("SRP flagged sample:")
    print(flagged[flagged["respondent"] == "SRP"].head(30).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(OUT, index=False)
    print(f"\nSaved {len(clean):,} rows to {OUT}")


if __name__ == "__main__":
    main()