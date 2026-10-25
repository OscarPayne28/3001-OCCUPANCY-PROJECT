# Data Issues Log

Every data problem found while auditing Dataset 2 (Monash Smart Infrastructure
occupancy and environmental sensor data), how it was found, and how it shaped the
project.

---

## 1. Occupancy-to-room mapping is unreliable and cannot be fixed

**Found:** `Sensor ID and Locations.xlsx` has a second sheet, named
`OCC (Incorrect Data - SpaceFloo...)` (Excel truncates sheet names), meant to map
the occupancy file's `floorspaceid` values to rooms through its `SpaceId` and
`FloorId` columns. Of the 5 `floorspaceid` values in the occupancy data, only 2
appear in `SpaceId` and none in `FloorId`. The sheet has visible defects: one row
with the ID `missing` (note "Missing Id"), three IDs marked `(?)`, and notes reading
"Not found ID," and `#REF!`.

**How:** set intersection between the unique `floorspaceid` values in the occupancy
data and the sheet's `SpaceId` and `FloorId` columns, then a review of the `Note`
column.

**Confirmation:** an existing EdStem thread on this exact problem (endorsed answer,
relaying confirmation from Keenan at MSI) states that the space and floor IDs are
not synced at the cloud level and that no corrected mapping can be provided.

**Impact:** occupancy spaces cannot be reliably tied to rooms, or joined to the
environmental data.

**Handling:** the mapping gap is treated as a stated constraint. Spaces are labelled
by the first 8 characters of their `floorspaceid`. The two room matches the sheet
does give are treated as unverified and are not used in the results.

---

## 2. Environmental and occupancy sensors cannot be linked

**Found:** the environmental readings file has 5 sensors (`6012002000227`, `...241`,
`...326`, `...777`, `...869`). Only `6012002000869` (EMS107, room G.38) appears in
the `ENV` sheet. The `ENV` sheet does list environmental sensors in G.20 (ES1G01 to
ES1G03) and G.25 (ES1G09 to ES1G12 and ES1G14), the two rooms that the unreliable
occupancy sheet associates with two occupancy spaces. Those sensors have
`6012003`-prefixed IDs, and none of them appear in the readings file.

**How:** intersection of the readings file's `sensorid` values with the `ENV` sheet's
`ID` column, then filtering the `ENV` sheet by `Floor/Room`.

**Impact:** no space has both a located environmental sensor and a located
occupancy space, so environmental readings cannot be used as predictors.

**Handling:** the project forecasts occupancy from occupancy history and calendar
features only. The environmental code is part of the audit and is not used by the
final pipeline.

---

## 3. Formaldehyde readings missing from most of the environmental data

**Found:** a first parsing attempt discarded 113,380 of 179,447 environmental
messages (63%). Every failing message was structurally the same as a passing one,
except that its `Formaldehyde` entry had a `time` but no `value`. This begins about
66,000 messages into the file and continues to the end.

**How:** a diagnostic that classified every `KeyError` by missing key and variable,
run block by block, with the raw offending JSON printed for inspection.

**Impact:** Formaldehyde is unusable for most of the data (present in 66,067 of
179,447 messages). The other variables are unaffected.

**Handling:** `flatten_env_chunk` now skips only the unreadable reading, not the
whole message, and records how many of each kind were skipped in a counter passed in
by the caller. It is covered by unit tests. After the fix, 1,824,135 readings were
extracted.

---

## 4. Duplicate sensor ID in the locations sheet

**Found:** ID `6012003000691` is given to two sensors on the `ENV` sheet: `ES2G14`
(room G.52, Workshop) and `ES2G01` (room G.61, East Entrance). Their physical labels
differ, so this looks like a data-entry error.

**How:** `locations["ID"].is_unique` returned `False`.

**Impact:** minor. It affects 2 of 78 rows, and neither sensor is in the readings
file.

**Handling:** documented only; not corrected.

---

## 5. File names do not match the date ranges

**Found:** both files are named `...MayToDec2024...`, but the occupancy data runs from
2023-11-23 to 2026-04-22 and the environmental data from 2023-05-05 to 2026-04-22.

**How:** `min()` and `max()` of the parsed timestamps.

**Impact:** none on the results, since the real range was used throughout.

**Handling:** the actual timestamps are used at every stage.

---

## 6. Long silences in the occupancy data are ambiguous

**Found:** the occupancy data is event-based, so quiet periods produce gaps of hours
or days. After an occupied reading, the next event arrives within 498 s in 99.9% of
cases (maximum about 9.5 hours). All 31 silences longer than one day follow an empty
reading, and they cluster around 20 December to 10 January, with a few weekends. The
two longest last about 15.6 days and start on 23 December 2023. This is consistent
with closures, but a sensor outage while a space was empty would look identical.

**How:** time to the next event for every row, grouped by the status of the row
before the gap.

**Handling:** the last known value is held through silences. A staleness cap was
tested as a sensitivity check: bins held from an event older than 2, 4, 8 and 24
hours would be masked, affecting 21.3%, 13.4%, 5.9% and 1.2% of bins. At 2 hours,
combined with the lag-feature requirement, usable rows fell from 339,021 to 155,394
and every method got worse (15-minute error for persistence 0.211 to 0.337, and for
the tree 0.256 to 0.419), because the removed bins were the quiet, easily predicted
ones. The cap was not adopted.

---

## 7. Large headcounts are unverified

**Found:** headcount reaches 72, and 0.17% of event rows are above 30.

**Handling:** values were kept, since there is no independent source to confirm or
reject them. This is a limit on what the forecasts can claim.