import pandas as pd
from sklearn.dummy import DummyRegressor

from occupancy.evaluate import time_split, compare_at_horizon


def test_time_split_respects_order():
    df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")})
    train, test, cutoff = time_split(df, train_frac=0.8)
    assert train["time"].max() < cutoff <= test["time"].min()
    assert len(train) + len(test) == len(df)


def test_compare_at_horizon_persistence_is_exact_for_constant_series():
    # a perfectly constant series: persistence should have zero error
    times = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")
    raw = pd.DataFrame({"space": ["A"] * 20, "time": times, "headcount": [5] * 20})
    encoded = raw.assign(feat=0.0)

    result = compare_at_horizon(
        raw, encoded, ["feat"], {"dummy": DummyRegressor(strategy="mean")},
        horizon_bins=1, train_frac=0.5,
    )
    assert result["persistence_MAE"] == 0.0