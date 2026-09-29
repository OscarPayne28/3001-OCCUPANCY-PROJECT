import numpy as np
import pandas as pd

from occupancy.features import add_time_features, add_lag_features


def test_add_time_features_known_values():
    # 2024-01-01 00:00 UTC = 2024-01-01 11:00 Melbourne (a Monday)
    df = pd.DataFrame({"time": pd.to_datetime(["2024-01-01 00:00:00"], utc=True)})
    out = add_time_features(df)
    assert out["local_hour"].iloc[0] == 11.0
    assert out["dow"].iloc[0] == 0          # Monday
    assert out["is_weekend"].iloc[0] == False
    assert np.isclose(out["hour_sin"].iloc[0], np.sin(2 * np.pi * 11 / 24))


def test_add_lag_features_shifts_within_group_only():
    df = pd.DataFrame({
        "space": ["A", "A", "A", "B", "B"],
        "value": [10, 20, 30, 100, 200],
    })
    out = add_lag_features(df, "space", "value", lags=[1])
    # first row of each group has no lag -> NaN
    assert pd.isna(out["value_lag1"].iloc[0])
    assert out["value_lag1"].iloc[1] == 10
    assert out["value_lag1"].iloc[2] == 20
    assert pd.isna(out["value_lag1"].iloc[3])   # start of group B
    assert out["value_lag1"].iloc[4] == 100