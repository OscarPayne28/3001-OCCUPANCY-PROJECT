"""Feature engineering for occupancy forecasting."""
import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Add calendar features derived from a UTC timestamp column.

    Args:
        df: DataFrame containing a timezone-aware UTC timestamp column.
        time_col: Name of that column.

    Returns:
        A copy of ``df`` with added columns: ``local_hour`` (0-24, fractional),
        ``dow`` (0=Monday ... 6=Sunday), ``is_weekend`` (bool), and cyclic
        encodings ``hour_sin``/``hour_cos`` so the model sees hour 23 and
        hour 0 as close together.

    Raises:
        KeyError: If ``time_col`` is not in ``df``.
    """
    out = df.copy()
    local = out[time_col].dt.tz_convert("Australia/Melbourne")
    out["local_hour"] = local.dt.hour + local.dt.minute / 60
    out["dow"] = local.dt.dayofweek
    out["is_weekend"] = out["dow"] >= 5
    out["hour_sin"] = np.sin(2 * np.pi * out["local_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["local_hour"] / 24)
    return out


def add_lag_features(
    df: pd.DataFrame, group_col: str, value_col: str, lags: list[int]
) -> pd.DataFrame:
    """Add lagged values of ``value_col`` within each group.

    Rows must already be sorted by time within each group before calling
    this function, since it shifts by row position, not by timestamp.

    Args:
        df: DataFrame sorted by time within each group.
        group_col: Column identifying each independent series (e.g. space).
        value_col: Column to lag.
        lags: Lag lengths in rows. For a 15-minute grid, [1, 4, 96] gives
            15 minutes, 1 hour and 1 day of lookback.

    Returns:
        A copy of ``df`` with one new column per lag, named
        ``{value_col}_lag{n}``. The first ``n`` rows of each group are NaN.
    """
    out = df.copy()
    for lag in lags:
        out[f"{value_col}_lag{lag}"] = out.groupby(group_col)[value_col].shift(lag)
    return out

def add_closure_flag(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Flag rows likely to fall in the summer teaching closure.

    This is an approximation (20 Dec to 10 Jan each year), not confirmed
    closure dates, since none were provided with the dataset.

    Args:
        df: DataFrame with a timezone-aware timestamp column.
        time_col: Name of that column.

    Returns:
        A copy of ``df`` with an added boolean column ``likely_closure``.
    """
    out = df.copy()
    local = out[time_col].dt.tz_convert("Australia/Melbourne")
    out["likely_closure"] = ((local.dt.month == 12) & (local.dt.day >= 20)) | (
        (local.dt.month == 1) & (local.dt.day <= 10)
    )
    return out

def build_features(
    grid: pd.DataFrame,
    lags: list[int],
    roll_window: int = 4,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the full model-ready feature table from a gridded occupancy series.

    Applies time features, a closure flag, lag features, a rolling mean,
    one-hot encodes ``space``, and drops rows with any missing feature
    (e.g. the first lags*bin_width of each space's history).

    Args:
        grid: DataFrame with columns ``time`` (UTC, tz-aware), ``space``
            (categorical label) and ``headcount``, one row per space per bin,
            as produced by gridding each space with ``step_function_to_grid``.
        lags: Lag lengths in rows to compute via ``add_lag_features``.
        roll_window: Window size (in rows) for the rolling mean of headcount.

    Returns:
        A tuple of (encoded feature DataFrame, list of feature column names).
        The returned DataFrame also retains ``time``, ``space`` and
        ``headcount`` for downstream splitting and evaluation.

    Raises:
        KeyError: If ``grid`` is missing ``time``, ``space`` or ``headcount``.
    """
    df = grid.sort_values(["space", "time"]).copy()
    df = add_time_features(df, time_col="time")
    df = add_closure_flag(df, time_col="time")
    df = add_lag_features(df, group_col="space", value_col="headcount", lags=lags)
    df["roll_1hr"] = df.groupby("space")["headcount"].transform(
        lambda s: s.rolling(roll_window, min_periods=1).mean()
    )
    df["occupied"] = df["headcount"] > 0
    df = df.dropna()

    encoded = pd.get_dummies(df, columns=["space"], prefix="space")
    space_cols = [c for c in encoded.columns if c.startswith("space_")]
    feature_cols = (
        ["hour_sin", "hour_cos", "dow", "is_weekend", "likely_closure", "roll_1hr"]
        + [f"headcount_lag{l}" for l in lags]
        + space_cols
    )
    return encoded, feature_cols
