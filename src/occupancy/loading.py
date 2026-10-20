import json
import warnings
from collections import Counter

import pandas as pd

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


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

REQUIRED_OCC_COLUMNS = {
    "id", "floorspaceid", "headcount", "collecteddate", "occupancystatuschangedate",
}


def convert_occupancy_csv(src, out_path, chunksize: int = 500_000) -> int:
    """Convert the raw occupancy CSV (or a single-file zip of it) to Parquet.

    Reads in chunks so the 9-million-row file never has to fit in memory.
    Timestamps are parsed to UTC, because the source mixes +10 and +11
    offsets across daylight saving.

    Args:
        src: Path to the CSV, or to a zip containing exactly one CSV.
        out_path: Where to write the Parquet file. Parent folders are created.
        chunksize: Rows read per chunk.

    Returns:
        The number of rows written.

    Raises:
        ValueError: If required columns are missing or the file has no rows.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer, n_rows = None, 0
    try:
        for chunk in pd.read_csv(src, chunksize=chunksize, dtype={"headcount": "int16"}):
            missing = REQUIRED_OCC_COLUMNS - set(chunk.columns)
            if missing:
                raise ValueError(f"occupancy file is missing columns: {sorted(missing)}")
            chunk["collecteddate"] = pd.to_datetime(chunk["collecteddate"], utc=True)
            chunk["occupancystatuschangedate"] = pd.to_datetime(
                chunk["occupancystatuschangedate"], utc=True
            )
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            n_rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    if n_rows == 0:
        raise ValueError("occupancy file contains no rows")
    return n_rows