# Warsaw Animated Transit Map

Animated density visualization of Warsaw's public transit (trams, buses, trains) based on GTFS schedule data.

---

## Quick Start (Run This Order)

### Step 0 — Set the date in `config.py`
```python
ANALYSIS_DATE = "20260427"  # YYYYMMDD format
```

### Step 1 — Download fresh GTFS data
```bash
python core/gtfs_downloader.py
```
Output: `data/raw/warsaw_gtfs_YYYYMMDD_HHMMSS/`

### Step 2 — Build route lines
```bash
python core/route_line_builder.py
```
Output: `data/processed/route_lines_continuous.shp`

### Step 3 — Create junction segments
```bash
python core/junction_segmenter.py
```
Output: `data/processed/tram_junction_segments.shp`, `bus_*`, `train_*`
> Tip: Load these in QGIS to verify before rendering the animation.

### Step 4 — Render animation
```bash
python animate/animate_full_density.py
```
Output: `animate/_output/warsaw_transit_density_full.mp4`

---

## Key Config Options

**`config.py`**
```python
ANALYSIS_DATE = "20260427"  # Date to analyze
```

**`core/junction_segmenter.py`**
```python
TEST_MODE = False        # True = 1km test area only (faster)
TEST_AREA_SIZE = 1000    # Size of test area in meters
```

**`animate/animate_full_density.py`**
```python
VEHICLE_FILTER = None    # 'Tram', 'Bus', 'Train', or None for all
FPS = 20
DURATION = 90            # seconds
HOURS = 6                # hours of transit to animate
```

---

## When to Re-Download GTFS Data

GTFS data is date-specific. If your target date is more than ~2 months ahead of the last download, fetch fresh data (Step 1). The existing data in `data/raw/` shows its download date in the folder name.

---

## Project Structure

```
gtfs_schedules_city/
├── config.py                        ← START HERE: set ANALYSIS_DATE
├── core/
│   ├── gtfs_downloader.py           ← Step 1: download GTFS
│   ├── gtfs_parser.py               ← (called internally, don't run directly)
│   ├── route_line_builder.py        ← Step 2: build route lines
│   └── junction_segmenter.py        ← Step 3: create segments
├── animate/
│   └── animate_full_density.py      ← Step 4: render animation
├── visualization/
│   ├── map_generator.py             ← Optional: generate static PNG maps
│   └── config_viz.py
├── x_capacity_calculator.py         ← Optional: add passenger capacity estimates
├── x_create_segment_diagnostic.py   ← Optional: diagnostic/debug tool
├── data/
│   ├── raw/                         ← Downloaded GTFS (gitignored)
│   └── processed/                   ← Intermediate shapefiles (gitignored)
└── archive/                         ← Old/unused scripts (ignore)
```

---

## Vehicle Snap Distances (junction_segmenter)

| Vehicle | Snap distance |
|---------|--------------|
| Tram    | 50m          |
| Bus     | 30m          |
| Train   | 50m          |

---

## Troubleshooting

**April 27 not found in GTFS data** → Re-run Step 1 to download fresh data.

**Animation too slow / out of memory** → Reduce `HOURS`, `DURATION`, or `FPS` in `animate_full_density.py`.

**Only a few segments with activity** → Check `ANALYSIS_DATE` in `config.py` matches the date in your GTFS download.

**Want to test quickly** → Set `TEST_MODE = True` in `junction_segmenter.py` and `VEHICLE_FILTER = 'Tram'` in the animation script.

---

## Data Source

Warsaw GTFS data from [mkuran.pl](https://mkuran.pl/gtfs/) (ZTM Warszawa).
