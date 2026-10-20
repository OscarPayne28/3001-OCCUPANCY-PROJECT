import pandas as pd

from occupancy.models import make_models, training_rows


def test_make_models_applies_tree_params():
    models = make_models({"max_depth": 3, "min_samples_leaf": 7})
    assert set(models) == {"tree", "linear"}
    assert models["tree"].max_depth == 3
    assert models["tree"].min_samples_leaf == 7


def test_training_rows_are_chronological_and_target_is_in_the_future():
    times = pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC")
    raw = pd.DataFrame({"space": "A", "time": times, "headcount": range(40)})
    encoded = raw.assign(feat=0.0)

    X, y, cutoff = training_rows(raw, encoded, ["feat"], horizon_bins=4, train_frac=0.5)

    t = encoded.loc[X.index, "time"]
    assert t.is_monotonic_increasing          # folds can be chronological
    assert (t < cutoff).all()                 # no test-period rows
    # row i has headcount i, so its target 4 bins ahead must be i + 4
    assert (y.values == X.index.values + 4).all()