"""Model comparison utilities for occupancy forecasting."""
import pandas as pd
from sklearn.metrics import mean_absolute_error


def time_split(df: pd.DataFrame, time_col: str = "time", train_frac: float = 0.8):
    """Split a DataFrame into train/test by time, preserving chronological order.

    Args:
        df: DataFrame with a timestamp column.
        time_col: Name of that column.
        train_frac: Fraction of the time range assigned to training.

    Returns:
        (train_df, test_df, cutoff) where ``cutoff`` is the timestamp boundary.
    """
    cutoff = df[time_col].quantile(train_frac)
    return df[df[time_col] < cutoff], df[df[time_col] >= cutoff], cutoff


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
    target = raw.groupby("space")["headcount"].shift(-horizon_bins)
    valid = target.notna()

    X = encoded.loc[valid, feature_cols]
    y = target[valid]
    cutoff = encoded.loc[valid, "time"].quantile(train_frac)
    is_train = encoded.loc[valid, "time"] < cutoff

    result = {"n_rows": int(valid.sum())}
    result["persistence_MAE"] = mean_absolute_error(
        y[~is_train], raw.loc[valid, "headcount"][~is_train]
    )
    for name, model in models.items():
        model.fit(X[is_train], y[is_train])
        result[f"{name}_MAE"] = mean_absolute_error(y[~is_train], model.predict(X[~is_train]))
    return result