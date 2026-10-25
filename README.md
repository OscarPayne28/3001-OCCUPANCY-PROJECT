# 3001-OCCUPANCY-PROJECT
# MMA3001 Individual Project: Occupancy Forecasting

Forecasting the headcount of five monitored spaces at four horizons (15 minutes,
1 hour, 4 hours and 1 day ahead) from their own occupancy history. The intended
use is pre-activating or standing down audio-visual (AV) equipment.

Author: OSCAR LEE PAYNE, 34962026. Unit: MMA3001 Numerical Methods and Machine Learning.

## Purpose

The occupancy sensors log an event only when something changes, so the raw data
is irregular. The project converts it to a regular 15-minute series by exact
integration of the step signal, builds calendar and lag features, tunes a decision
tree with time-ordered cross-validation, and compares it with a persistence
baseline and Linear Regression on a later test period. The full write-up is the
submitted report.

## Data

The data is not included in this repository (size, and it belongs to the data
provider). Obtain the Dataset 2 files from the unit and place them in `data/raw/`:

- `5occupancySensor_MayToDec2024_9MRows.zip` (used by the pipeline; the zip
  contains one CSV and is read directly)
- `5EnvSensor_MayToDec2024_180kRows.zip` and `Sensor ID and Locations.xlsx` were
  used only for the data audit (`notebooks/01_data_audit.ipynb`)

The file names say May to December 2024, but the occupancy data actually spans
2023-11-23 to 2026-04-22. The room mapping for the occupancy data is unreliable
and could not be corrected, so spaces are labelled by the first 8 characters of
their `floorspaceid`. See `DATA_ISSUES.md` for every data problem found.

## Setup

Requires Python 3.10 or newer (developed on 3.12).

```
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Run the whole workflow from the repository root:

```
python -m occupancy.pipeline data/raw/5occupancySensor_MayToDec2024_9MRows.zip
```

This converts the CSV to Parquet, grids each space into 15-minute bins, builds
features, tunes the tree, and compares the methods at all four horizons. It writes:

- `data/processed/occupancy.parquet` and `data/processed/occupancy_15min.parquet`
  (not tracked by git)
- `results/results.csv` and `results/best_params.json`

It took about 4 minutes on the author's laptop (12 logical cores): about 2 min 48 s
for the CSV conversion and about 75 s for the remaining stages.

### Results (test MAE, in people; lower is better)

| Horizon | Persistence | Linear Regression | Decision Tree |
|---|---|---|---|
| 15 min | 0.211 | 0.273 | 0.256 |
| 1 hour | 0.370 | 0.424 | 0.364 |
| 4 hours | 0.599 | 0.606 | 0.466 |
| 1 day | 0.528 | 0.542 | 0.464 |

Tuned tree settings: `max_depth` 12, `min_samples_leaf` 50. The tree is clearly
better than persistence from 4 hours; persistence is better at 15 minutes and the
two are comparable at 1 hour. Results are by space and day type in the report.

### Notebooks

The notebooks are exploratory and are not needed for the results above.
`notebooks/01_data_audit.ipynb` is the data audit. `notebooks/02_modelling.ipynb`
produced the report figures in `report/`; run the pipeline first, because the
notebook reads files it writes.

## Testing

```
pytest
pytest --html=docs/test_report.html --self-contained-html
```

24 tests, with the latest report in `docs/test_report.html`.

## Documentation

HTML documentation of the source code is generated with pdoc and stored in
`docs/`. Open `docs/index.html`. To regenerate:

```
pdoc src/occupancy -o docs --docformat google
```

## Repository structure

```
src/occupancy/
    loading.py     CSV to Parquet conversion; environmental audit helpers
    gridding.py    events to a regular time grid (exact integration)
    features.py    calendar, lag, rolling and closure features
    models.py      model construction and chronological tuning
    evaluate.py    forecast split and comparison against persistence
    pipeline.py    end-to-end run (python -m occupancy.pipeline)
tests/             pytest tests, one file per module
notebooks/         data audit and modelling notebooks
docs/              HTML documentation and test report
report/            figures used in the report
results/           outputs written by the pipeline
DATA_ISSUES.md    data problems found and how they were handled
AI_USAGE.md        disclosure of AI use
```

## Limitations

- The room mapping is unreliable, so results are by anonymised space, not by room
- The environmental data cannot be linked to the occupancy spaces and is not used
- The test period covers four of the five spaces, with a single chronological split
- The tree is evaluated on headcount error, not on occupied-versus-empty detection

## AI use and licence

AI assistance is disclosed in the AI reflection section of the project report.
The code is released under the licence in `LICENSE`.