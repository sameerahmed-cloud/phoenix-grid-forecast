import numpy as np
import pandas as pd

from src.features.build import add_demand_lags


def make_series(n: int = 400) -> pd.DataFrame:
    period = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"period": period, "demand_mw": np.arange(n, dtype=float)})


def test_lags_point_to_the_right_hour():
    g = add_demand_lags(make_series())
    t = 300
    assert g.loc[t, "lag_24"] == g.loc[t - 24, "demand_mw"]
    assert g.loc[t, "lag_168"] == g.loc[t - 168, "demand_mw"]


def test_features_ignore_the_last_23_hours():
    base = make_series()
    changed = base.copy()
    t = 300
    changed.loc[t - 23 : t, "demand_mw"] = 99999.0  # corrupt everything newer than 24h before t
    a = add_demand_lags(base).loc[t]
    b = add_demand_lags(changed).loc[t]
    for col in ["lag_24", "lag_48", "lag_168", "roll_mean_24", "roll_mean_168"]:
        assert a[col] == b[col], f"{col} leaks recent demand"