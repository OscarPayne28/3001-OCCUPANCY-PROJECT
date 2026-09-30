import json
import warnings
from collections import Counter

import pandas as pd


def flatten_env_chunk(chunk: pd.DataFrame, problems: Counter | None = None) -> pd.DataFrame:
    """Flatten environmental-sensor batch messages into one row per reading.

    Args:
        chunk: DataFrame with columns ``sensorid`` and ``jsondata``, where
            ``jsondata`` is a JSON list of ``{"values": [...], "variable": {...}}``.
        problems: Optional Counter. If given, counts of skipped items are added
            to it (keyed by problem type) and no warning is raised, so counts
            can be accumulated across many chunks.

    Returns:
        DataFrame with columns ``sensorid`` (str), ``time`` (UTC datetime),
        ``variable`` (str), ``unit`` (str) and ``value`` (float32).

    Warns:
        UserWarning: If ``problems`` is None and anything had to be skipped.
    """
    local = problems if problems is not None else Counter()
    records = []

    for sensorid, raw in zip(chunk["sensorid"], chunk["jsondata"]):
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            local["row: missing or invalid json"] += 1
            continue
        if not isinstance(payload, list):
            local["row: json is not a list"] += 1
            continue

        for var in payload:
            try:
                name = var["variable"]["name"]
                unit = var["variable"]["unit"]
                values = var["values"]
            except (TypeError, KeyError):
                local["variable: malformed"] += 1
                continue
            for v in values:
                try:
                    records.append((str(sensorid), v["time"], name, unit, v["value"]))
                except (TypeError, KeyError):
                    local[f"reading missing time/value: {name}"] += 1

    if problems is None and local:
        warnings.warn(f"Skipped items while flattening: {dict(local)}")

    out = pd.DataFrame.from_records(
        records, columns=["sensorid", "time", "variable", "unit", "value"]
    )
    out["time"] = pd.to_datetime(out["time"], unit="s", utc=True)
    out["value"] = out["value"].astype("float32")
    return out

def load_locations(path: str) -> pd.DataFrame:
    """Load the sensor-locations spreadsheet.

    Args:
        path: Path to the Excel file (ENV sheet).

    Returns:
        DataFrame with one row per sensor. The ``ID`` column is kept as text
        to avoid it being read as a number.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return pd.read_excel(path, dtype={"ID": str})