"""Model comparison utilities for occupancy forecasting."""
from typing import NamedTuple

import pandas as pd
from sklearn.metrics import mean_absolute_error


class ForecastSplit(NamedTuple):
    """Features, shifted target and chronological split for one forecast horizon.

    Attributes:
        X: Feature rows that have a known future target.
        y: Headcount ``horizon_bins`` ahead of each row of ``X``.
        is_train: True for rows before the time cutoff.
        current: Headcount at each row's own time (the persistence forecast).
        cutoff: Timestamp separating training from test rows.
    """

    X: pd.DataFrame
    y: pd.Series
    is_train: pd.Series
    current: pd.Series
    cutoff: pd.Timestamp


def forecast_split(
    raw: pd.DataFrame,
    encoded: pd.DataFrame,
    feature_cols: list[str],
    horizon_bins: int,
    train_frac: float = 0.8,
) -> ForecastSplit:
    """Build the shifted target and chronological train/test split for one horizon.

    The target is shifted within each space, so a row is never paired with
    another space's future. Rows without a known future value are dropped.
    This is the single place that defines the split, used by both tuning and
    evaluation so the two cannot disagree.

    Args:
        raw: Unencoded DataFrame with ``space`` and ``headcount``, sorted by
            time within each space, with the same index as ``encoded``.
        encoded: Feature DataFrame with a ``time`` column.
        feature_cols: Feature column names to return in ``X``.
        horizon_bins: Number of bins ahead to forecast.
        train_frac: Fraction of the time range used for training.

    Returns:
        A ``ForecastSplit``.
    """
    target = raw.groupby("space")["headcount"].shift(-horizon_bins)
    valid = target.notna()
    t = encoded.loc[valid, "time"]
    cutoff = t.quantile(train_frac)
    return ForecastSplit(
        X=encoded.loc[valid, feature_cols],
        y=target[valid],
        is_train=t < cutoff,
        current=raw.loc[valid, "headcount"],
        cutoff=cutoff,
    )


def compare_at_horizon(
    raw: pd.DataFrame,
    encoded: pd.DataFrame,
    feature_cols: list[str],
    models: dict,
    horizon_bins: int,
    train_frac: float = 0.8,
) -> dict:
    """Compare model MAE against a persistence baseline at one forecast horizon.

    Args:
        raw: Unencoded DataFrame with ``space``, ``time``, ``headcount``,
            aligned row-for-row with ``encoded``.
        encoded: Feature-encoded DataFrame (output of ``build_features``).
        feature_cols: Feature column names to train on.
        models: Mapping of model name to an unfitted scikit-learn estimator.
        horizon_bins: Number of bins ahead to forecast.
        train_frac: Fraction of the time range used for training.

    Returns:
        Dict with ``n_rows``, ``persistence_MAE``, and one ``{name}_MAE`` key
        per entry in ``models``.
    """
    s = forecast_split(raw, encoded, feature_cols, horizon_bins, train_frac)

    result = {"n_rows": len(s.y)}
    result["persistence_MAE"] = mean_absolute_error(
        s.y[~s.is_train], s.current[~s.is_train]
    )
    for name, model in models.items():
        model.fit(s.X[s.is_train], s.y[s.is_train])
        predictions = model.predict(s.X[~s.is_train])
        result[f"{name}_MAE"] = mean_absolute_error(s.y[~s.is_train], predictions)
    return result