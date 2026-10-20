"""Model construction and hyperparameter tuning for occupancy forecasting."""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from occupancy.evaluate import forecast_split


def make_models(tree_params: dict) -> dict:
    """Create the unfitted models compared in the evaluation.

    Args:
        tree_params: Keyword arguments for ``DecisionTreeRegressor``
            (for example ``max_depth`` and ``min_samples_leaf``).

    Returns:
        Dict mapping model name (``"tree"``, ``"linear"``) to an unfitted
        scikit-learn estimator.
    """
    return {
        "tree": DecisionTreeRegressor(**tree_params, random_state=0),
        "linear": LinearRegression(),
    }


def training_rows(raw, encoded, feature_cols, horizon_bins, train_frac=0.8):
    """Return training features and the shifted target, in chronological order.

    Uses ``forecast_split`` so it shares the exact split used for evaluation.
    Only rows before the time cutoff are returned, so the test period is
    never used for tuning.

    Args:
        raw: Unencoded DataFrame with ``space``, ``time`` and ``headcount``.
        encoded: Feature DataFrame with the same index as ``raw``.
        feature_cols: Feature column names.
        horizon_bins: Number of bins ahead to forecast.
        train_frac: Fraction of the time range used for training.

    Returns:
        (X, y, cutoff): features and target sorted by time, and the cutoff timestamp.
    """
    s = forecast_split(raw, encoded, feature_cols, horizon_bins, train_frac)
    t = encoded.loc[s.X.index, "time"]
    order = t[s.is_train].sort_values().index
    return s.X.loc[order], s.y.loc[order], s.cutoff


def tune_tree(raw, encoded, feature_cols, param_grid, horizon_bins, train_frac=0.8, n_splits=5):
    """Grid-search decision-tree settings with chronological cross-validation.

    Args:
        raw: Unencoded DataFrame (see ``training_rows``).
        encoded: Feature DataFrame (see ``training_rows``).
        feature_cols: Feature column names.
        param_grid: Dict of ``DecisionTreeRegressor`` settings to search.
        horizon_bins: Forecast horizon to tune at, in bins.
        train_frac: Fraction of the time range used for training.
        n_splits: Number of ``TimeSeriesSplit`` folds.

    Returns:
        The fitted ``GridSearchCV`` object. ``best_params_`` holds the chosen settings.
    """
    X, y, _ = training_rows(raw, encoded, feature_cols, horizon_bins, train_frac)
    search = GridSearchCV(
        DecisionTreeRegressor(random_state=0),
        param_grid,
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=-1,
    )
    search.fit(X, y)
    return search