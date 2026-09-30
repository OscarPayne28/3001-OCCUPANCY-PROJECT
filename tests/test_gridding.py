import numpy as np
import pandas as pd

from occupancy.gridding import step_function_to_grid, bin_staleness

BASE = pd.Timestamp("2024-01-01", tz="UTC")


def _times(seconds):
    return pd.Series(BASE + pd.to_timedelta(seconds, unit="s"))


def test_half_and_half_bin_is_the_average():
    # value 2 for 450 s, then 4 for 450 s, then 0
    out = step_function_to_grid(_times([0, 450, 900, 1800]), [2, 4, 0, 0], "15min")
    assert len(out) == 2
    assert out.iloc[0] == 3.0
    assert out.iloc[1] == 0.0


def test_partial_first_bin_is_dropped_and_value_is_held():
    # first event at 300 s, so the [0, 900) bin is only partly known
    out = step_function_to_grid(_times([300, 1200, 2700]), [1, 3, 3], "15min")
    assert len(out) == 2
    assert np.isclose(out.iloc[0], (300 * 1 + 600 * 3) / 900)
    assert out.iloc[1] == 3.0


def test_long_silence_holds_the_last_value():
    out = step_function_to_grid(_times([0, 86400]), [5, 5], "15min")
    assert len(out) == 96
    assert (out == 5.0).all()

def test_bin_staleness_flags_long_silence():
    # event at 0s, next event 3 hours later — every bin in between is stale under a 2hr cap
    out = bin_staleness(_times([0, 10800]), "15min", max_gap_hours=2.0)
    assert out.sum() > 0
    assert out.iloc[0] == False   # first bin, right after the event, is fresh
    assert out.iloc[-1] == False  # last bin ends exactly at the second event


def test_bin_staleness_all_fresh_when_gap_is_small():
    out = bin_staleness(_times([0, 600, 1200, 1800]), "15min", max_gap_hours=2.0)
    assert (out == False).all()