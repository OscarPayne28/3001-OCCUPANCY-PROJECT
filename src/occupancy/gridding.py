"""Convert irregular event streams into regular time series."""
import numpy as np
import pandas as pd


def step_function_to_grid(times: pd.Series, values, freq: str = "15min") -> pd.Series:
    """Time-weighted mean of a piecewise-constant signal on a regular grid.

    Each value is assumed to hold until the next event. Only bins that lie
    fully between the first and last event are returned, so nothing is
    extrapolated beyond the recorded data.

    Args:
        times: Timezone-aware event timestamps, sorted ascending. Ties are
            allowed; the later row wins.
        values: Array-like of the signal after each event (same length as ``times``).
        freq: Bin width as a pandas offset string, such as ``"15min"``.

    Returns:
        Series indexed by bin start (UTC), holding the mean of the signal
        over each bin.

    Raises:
        ValueError: If ``times`` and ``values`` differ in length or have fewer
            than two events.
    """
    v = np.asarray(values, dtype=float)
    t = (
        times.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        .astype("datetime64[ms]").astype("int64") / 1000.0
    )
    if len(t) != len(v) or len(t) < 2:
        raise ValueError("need at least two events, with one value per timestamp")

    step = pd.Timedelta(freq).total_seconds()
    first = np.floor(t[0] / step) * step
    edges = np.arange(first, t[-1] + step, step)

    # integral of the signal at each event time
    cum = np.concatenate(([0.0], np.cumsum(v[:-1] * np.diff(t))))

    def integral(e):
        idx = np.searchsorted(t, e, side="right") - 1
        return cum[idx] + v[idx] * (e - t[idx])

    start, end = edges[:-1], edges[1:]
    ok = (start >= t[0]) & (end <= t[-1])
    means = (integral(end[ok]) - integral(start[ok])) / step
    index = pd.to_datetime(start[ok], unit="s", utc=True)
    return pd.Series(means, index=index, name="mean")

def bin_staleness(times: pd.Series, freq: str = "15min", max_gap_hours: float = 2.0) -> pd.Series:
    """Flag grid bins whose value is held from an event more than ``max_gap_hours`` old.

    For each bin, finds the time since the most recent event at or before the
    bin's end, and marks the bin unreliable if that gap exceeds ``max_gap_hours``.

    Args:
        times: Timezone-aware event timestamps, sorted ascending.
        freq: Bin width, matching what was passed to ``step_function_to_grid``.
        max_gap_hours: Bins held from an event more than this many hours stale
            are flagged.

    Returns:
        Boolean Series indexed by bin start (UTC), True where the bin should
        be treated as missing (too stale to trust).

    Raises:
        ValueError: If ``times`` has fewer than two events.
    """
    t = (
        times.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        .astype("datetime64[ms]").astype("int64") / 1000.0
    )
    if len(t) < 2:
        raise ValueError("need at least two events")

    step = pd.Timedelta(freq).total_seconds()
    first = np.floor(t[0] / step) * step
    edges = np.arange(first, t[-1] + step, step)
    start, end = edges[:-1], edges[1:]
    ok = (start >= t[0]) & (end <= t[-1])

    idx_at_end = np.searchsorted(t, end[ok], side="right") - 1
    gap_seconds = end[ok] - t[idx_at_end]
    stale = gap_seconds > (max_gap_hours * 3600)

    index = pd.to_datetime(start[ok], unit="s", utc=True)
    return pd.Series(stale, index=index, name="stale")