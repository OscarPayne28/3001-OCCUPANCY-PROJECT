import json
import warnings

import pandas as pd


def flatten_env_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Flatten environmental-sensor batch messages into one row per reading.

    Args:
        chunk: DataFrame with columns ``sensorid`` and ``jsondata``, where
            ``jsondata`` is a JSON list of ``{"values": [...], "variable": {...}}``.

    Returns:
        DataFrame with columns ``sensorid`` (str), ``time`` (UTC datetime),
        ``variable`` (str), ``unit`` (str) and ``value`` (float32).

    Warns:
        UserWarning: If any rows contain missing or invalid JSON. These rows
            are skipped and counted.
    """
    records = []
    n_bad = 0
    for sensorid, raw in zip(chunk["sensorid"], chunk["jsondata"]):
        try:
            payload = json.loads(raw)
            for var in payload:
                name = var["variable"]["name"]
                unit = var["variable"]["unit"]
                for v in var["values"]:
                    records.append((str(sensorid), v["time"], name, unit, v["value"]))
        except (TypeError, ValueError, KeyError):
            n_bad += 1

    if n_bad:
        warnings.warn(f"Skipped {n_bad} rows with missing or invalid jsondata")

    out = pd.DataFrame.from_records(
        records, columns=["sensorid", "time", "variable", "unit", "value"]
    )
    out["time"] = pd.to_datetime(out["time"], unit="s", utc=True)
    out["value"] = out["value"].astype("float32")
    return out