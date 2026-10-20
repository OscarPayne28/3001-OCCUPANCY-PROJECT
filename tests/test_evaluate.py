import pandas as pd
from sklearn.dummy import DummyRegressor

from occupancy.evaluate import compare_at_horizon, forecast_split


def test_forecast_split_target_is_in_the_future_and_split_is_chronological():
    times = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")
    raw = pd.DataFrame({"space": "A", "time": times, "headcount": range(20)})
    encoded = raw.assign(feat=0.0)

    s = forecast_split(raw, encoded, ["feat"], horizon_bins=3, train_frac=0.5)

    assert len(s.y) == 17                          # last 3 rows have no future value
    assert (s.y.values == s.current.values + 3).all()
    t = encoded.loc[s.X.index, "time"]
    assert (t[s.is_train] < s.cutoff).all()
    assert (t[~s.is_train] >= s.cutoff).all()


def test_target_does_not_cross_between_spaces():
    times = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    a = pd.DataFrame({"space": "A", "time": times, "headcount": range(10)})
    b = pd.DataFrame({"space": "B", "time": times, "headcount": range(100, 110)})
    raw = pd.concat([a, b], ignore_index=True)
    encoded = raw.assign(feat=0.0)

    s = forecast_split(raw, encoded, ["feat"], horizon_bins=2, train_frac=0.5)

    assert len(s.y) == 16                          # 8 valid rows per space
    assert (s.y.values == s.current.values + 2).all()


def test_compare_at_horizon_persistence_is_exact_for_constant_series():
    times = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")
    raw = pd.DataFrame({"space": ["A"] * 20, "time": times, "headcount": [5] * 20})
    encoded = raw.assign(feat=0.0)

    result = compare_at_horizon(
        raw, encoded, ["feat"], {"dummy": DummyRegressor(strategy="mean")},
        horizon_bins=1, train_frac=0.5,
    )
    assert result["persistence_MAE"] == 0.0