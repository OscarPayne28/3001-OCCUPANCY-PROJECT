import pandas as pd
import pytest

from occupancy.loading import flatten_env_chunk

GOOD = (
    '[{"values": [{"time": 1702273965, "value": 603}],'
    ' "variable": {"id": 67, "name": "Carbon dioxide", "unit": "ppm"}},'
    ' {"values": [{"time": 1702273965, "value": 23.8}],'
    ' "variable": {"id": 84, "name": "Temperature", "unit": "°C"}}]'
)


def test_flatten_one_row():
    chunk = pd.DataFrame({"sensorid": [6012002000241], "jsondata": [GOOD]})
    out = flatten_env_chunk(chunk)
    assert len(out) == 2
    assert set(out["variable"]) == {"Carbon dioxide", "Temperature"}
    assert out["time"].iloc[0] == pd.Timestamp("2023-12-11 05:52:45", tz="UTC")


def test_bad_json_is_skipped_with_warning():
    chunk = pd.DataFrame({"sensorid": [1, 2], "jsondata": [GOOD, "not json"]})
    with pytest.warns(UserWarning):
        out = flatten_env_chunk(chunk)
    assert len(out) == 2