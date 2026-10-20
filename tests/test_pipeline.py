import numpy as np
import pandas as pd
import pytest

from occupancy.loading import convert_occupancy_csv
from occupancy.pipeline import grid_occupancy, run_comparison


def test_grid_occupancy_matches_hand_calculation():
    base = pd.Timestamp("2024-01-01", tz="UTC")

    def events(space, secs, values):
        return pd.DataFrame({
            "id": range(len(secs)),
            "floorspaceid": space,
            "collecteddate": base + pd.to_timedelta(secs, unit="s"),
            "headcount": values,
        })

    occ = pd.concat([
        events("aaaaaaaa-0000", [0, 450, 900, 1800], [2, 4, 0, 0]),
        events("bbbbbbbb-0000", [0, 900, 1800], [1, 1, 1]),
    ], ignore_index=True)

    out = grid_occupancy(occ, "15min")

    a = out[out["space"] == "aaaaaaaa"].sort_values("time")
    b = out[out["space"] == "bbbbbbbb"].sort_values("time")
    assert list(a["headcount"]) == [3.0, 0.0]      # (2*450 + 4*450) / 900, then 0
    assert list(b["headcount"]) == [1.0, 1.0]


def test_grid_occupancy_rejects_missing_columns():
    with pytest.raises(ValueError):
        grid_occupancy(pd.DataFrame({"floorspaceid": ["a"]}))


def test_convert_occupancy_csv_roundtrip(tmp_path):
    csv = tmp_path / "occ.csv"
    pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "deviceid": "dev",
        "floorspaceid": "space-1",
        "occupancystatus": "CurrentlyOccupied",
        "headcount": [1, 2, 3, 2, 0],
        "collecteddate": ["2023-11-24 00:36:52+11"] * 5,
        "occupancystatuschangedate": ["2023-11-24 00:36:52+11"] * 5,
        "previousoccupancystatus": "NotOccupied",
    }).to_csv(csv, index=False)
    out = tmp_path / "out" / "occ.parquet"

    n = convert_occupancy_csv(csv, out, chunksize=2)   # forces several chunks

    df = pd.read_parquet(out)
    assert n == 5 and len(df) == 5
    assert list(df["headcount"]) == [1, 2, 3, 2, 0]
    # 00:36:52 at +11 is 13:36:52 UTC on the previous day
    assert df["collecteddate"].iloc[0] == pd.Timestamp("2023-11-23 13:36:52", tz="UTC")


def test_convert_occupancy_csv_rejects_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"id": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        convert_occupancy_csv(csv, tmp_path / "x.parquet")


def test_run_comparison_returns_one_row_per_horizon():
    n = 700
    times = pd.date_range("2024-03-04", periods=n, freq="15min", tz="UTC")
    cycle = np.clip(np.round(3 * np.sin(2 * np.pi * np.arange(n) / 96) + 2), 0, None)
    grid = pd.concat([
        pd.DataFrame({"time": times, "space": "A", "headcount": cycle}),
        pd.DataFrame({"time": times, "space": "B", "headcount": np.roll(cycle, 10)}),
    ], ignore_index=True)

    results, best = run_comparison(
        grid,
        lags=[1, 2],
        param_grid={"max_depth": [2, 3], "min_samples_leaf": [5]},
        horizons={"1bin": 1, "2bins": 2},
        tune_horizon_bins=2,
    )

    assert list(results["horizon"]) == ["1bin", "2bins"]
    mae_cols = ["persistence_MAE", "tree_MAE", "linear_MAE"]
    assert results[mae_cols].notna().all().all()
    assert set(best) == {"max_depth", "min_samples_leaf"}