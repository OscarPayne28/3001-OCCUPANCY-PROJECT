"""End-to-end occupancy forecasting pipeline.

Run from the repository root (after ``pip install -e .``)::

    python -m occupancy.pipeline data/raw/5occupancySensor_MayToDec2024_9MRows.zip

This converts the raw event log to Parquet, grids it into 15-minute bins,
builds features, tunes the decision tree and compares it with persistence and
linear regression at four horizons. Results are written to ``results/``.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from occupancy.evaluate import compare_at_horizon
from occupancy.features import build_features
from occupancy.gridding import step_function_to_grid
from occupancy.loading import convert_occupancy_csv
from occupancy.models import make_models, tune_tree

DEFAULT_LAGS = [1, 2, 4, 8, 16, 32, 96, 96 * 7]
DEFAULT_PARAM_GRID = {"max_depth": [4, 6, 8, 10, 12], "min_samples_leaf": [5, 20, 50, 100]}
DEFAULT_HORIZONS = {"15min": 1, "1hr": 4, "4hr": 16, "1day": 96}
TUNE_HORIZON_BINS = 16  # tune at the 4-hour horizon


def grid_occupancy(occ: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Convert the occupancy event log into a regular grid for every space.

    Each space is gridded separately with ``step_function_to_grid``. Spaces
    are labelled by the first 8 characters of their ``floorspaceid``, because
    the room mapping is unreliable (see DATA_ISSUES.md).

    Args:
        occ: Event log with columns ``id``, ``floorspaceid``, ``collecteddate``
            (UTC, timezone-aware) and ``headcount``.
        freq: Bin width, such as ``"15min"``.

    Returns:
        DataFrame with columns ``time`` (bin start, UTC), ``headcount``
        (time-weighted mean) and ``space``.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = {"id", "floorspaceid", "collecteddate", "headcount"} - set(occ.columns)
    if missing:
        raise ValueError(f"occupancy events are missing columns: {sorted(missing)}")

    grids = []
    for sid, grp in occ.groupby("floorspaceid"):
        grp = grp.sort_values(["collecteddate", "id"])
        hc = step_function_to_grid(grp["collecteddate"], grp["headcount"], freq)
        grids.append(hc.rename("headcount").to_frame().assign(space=sid[:8]))
    return pd.concat(grids).rename_axis("time").reset_index()


def run_comparison(
    grid: pd.DataFrame,
    lags=None,
    param_grid=None,
    horizons=None,
    tune_horizon_bins: int = TUNE_HORIZON_BINS,
    train_frac: float = 0.8,
):
    """Build features, tune the tree, and compare models at each horizon.

    Args:
        grid: Gridded occupancy (output of ``grid_occupancy``).
        lags: Lag lengths in bins. Defaults to ``DEFAULT_LAGS``.
        param_grid: Decision-tree settings to search. Defaults to ``DEFAULT_PARAM_GRID``.
        horizons: Mapping of label to horizon in bins. Defaults to ``DEFAULT_HORIZONS``.
        tune_horizon_bins: Horizon (in bins) at which the tree is tuned.
        train_frac: Fraction of the time range used for training.

    Returns:
        (results, best_params): a DataFrame with one row per horizon and the
        MAE of each method, and the tuned decision-tree settings.
    """
    lags = DEFAULT_LAGS if lags is None else lags
    param_grid = DEFAULT_PARAM_GRID if param_grid is None else param_grid
    horizons = DEFAULT_HORIZONS if horizons is None else horizons

    raw, encoded, feature_cols = build_features(grid, lags=lags)
    search = tune_tree(raw, encoded, feature_cols, param_grid, tune_horizon_bins, train_frac)

    rows = []
    for label, bins in horizons.items():
        r = compare_at_horizon(
            raw, encoded, feature_cols, make_models(search.best_params_), bins, train_frac
        )
        rows.append({"horizon": label, **r})
    return pd.DataFrame(rows), search.best_params_


def run_all(source, processed_dir="data/processed", results_dir="results", freq="15min"):
    """Run the full pipeline from the raw occupancy file to the results table.

    Args:
        source: Raw occupancy CSV, or a zip containing exactly one CSV.
        processed_dir: Where intermediate Parquet files are written (not tracked by git).
        results_dir: Where ``results.csv`` and ``best_params.json`` are written.
        freq: Grid bin width.

    Returns:
        (results, best_params) as returned by ``run_comparison``.
    """
    processed_dir, results_dir = Path(processed_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    events_path = processed_dir / "occupancy.parquet"
    convert_occupancy_csv(source, events_path)

    grid = grid_occupancy(pd.read_parquet(events_path), freq)
    grid.to_parquet(processed_dir / "occupancy_15min.parquet")

    results, best = run_comparison(grid)
    results.to_csv(results_dir / "results.csv", index=False)
    (results_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    return results, best


def main(argv=None):
    """Command-line entry point: ``python -m occupancy.pipeline <source>``."""
    parser = argparse.ArgumentParser(description="Run the occupancy forecasting pipeline.")
    parser.add_argument("source", help="occupancy CSV, or a zip containing one CSV")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)

    results, best = run_all(args.source, args.processed_dir, args.results_dir)
    print(results.to_string(index=False))
    print("best tree settings:", best)


if __name__ == "__main__":
    main()