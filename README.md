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

### Step 1b — Merge KM and WKD train data (optional, for full train coverage)
```bash
python core/km_gtfs_merger.py
```
Downloads Koleje Mazowieckie and WKD feeds and merges them into a combined directory (`warsaw_gtfs_*_with_km_wkd/`). Use this merged directory as input for subsequent steps if you want all train lines.

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
├── bus_lanes/                       ← Bus lane need map (separate pipeline, see below)
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

## Bus Lane Need Map (`bus_lanes/`)

Maps of where many buses run **without** a bus lane (Miasto Jest Nasze post), plus tools that helped fix bus
lanes in OpenStreetMap. Separate pipeline, only needs a fresh GTFS download (Step 1).
**Run order, data sources and lessons learnt: [`bus_lanes/README.md`](bus_lanes/README.md).**

---

## Vehicle Snap Distances (junction_segmenter)

| Vehicle | Snap distance |
|---------|--------------|
| Tram    | 50m          |
| Bus     | 30m          |
| Train   | 50m          |

---

## Troubleshooting

**Target date not found in GTFS data** → Re-run Step 1 to download fresh data.

**Animation too slow / out of memory** → Reduce `HOURS`, `DURATION`, or `FPS` in `animate_full_density.py`.

**Only a few segments with activity** → Check `ANALYSIS_DATE` in `config.py` matches the date in your GTFS download.

**Want to test quickly** → Set `TEST_MODE = True` in `junction_segmenter.py` and `VEHICLE_FILTER = 'Tram'` in the animation script.

---

## Data Source

Warsaw GTFS data from [mkuran.pl](https://mkuran.pl/gtfs/) (ZTM Warszawa).

---

## Live ZTM vehicle positions (`live/`) — separate sub-project

Goal: find where trams and buses get stuck, from real GPS positions (dane.um.warszawa.pl API, key in `live/api_keys.py`, git-ignored).

```bash
python live/collect.py            # polls every 10 s, runs until Ctrl+C
```
- Writes `live/data/positions_YYYYMMDD.csv` (appends; new file at midnight).
- Skips stale fixes (> 60 s old) and repeated identical fixes per vehicle.
- Network-proof: 3 attempts per fetch (2 s / 5 s backoff), outages + recovery logged to `live/data/collect_errors.log`, auto-restart on unexpected crash.
- No history on the API side — anything not collected is lost.

Stuck points: `python live/stuck_points.py [--date YYYYMMDD]` — matches each fix to its own line's GTFS shapes
(latest `data/raw/warsaw_gtfs_*`, must contain the collection day), keeps intervals < 5 km/h, drops time at served stops
(bus 35 m / tram 50 m), near termini (150 m), stand-stills > 10 min (layovers) and fixes > 40 m off route.
Output: `live/data/stuck_cells_<date>.gpkg` (50 m grid: tram_min, bus_min, vehicles, top_lines, near_stop) + `stuck_<date>.png`.
Safe to run while the collector is writing. ~3 min for half a day of data.
Also `passes_per_h` + `sec_per_pass` (stuck seconds per passing vehicle) and `stuck_per_vehicle_<date>.png`
(cells with >= 12 vehicles/h only — lower cut-offs surface suburban layover spots). Caches intervals in `live/data/_cache/`.

Line speeds: `python live/line_speeds.py [--date] [--fresh]` (uses the stuck_points cache) → `line_speeds_<date>.csv`,
`line_speed_ranking_<date>.png` (all trams, 15 fastest/slowest buses; lines >= 5 vehicle-hours) and
`line_charts_<date>/<mode>_<line>.png` (speed by hour vs mode average + moving/at stop/stuck split).
Speed = distance / time in service, incl. stop dwell and stuck time, excl. terminus areas and layovers.

Route profiles: `python live/route_profiles.py [--mode Bus] [--lines 1 14] [--hours 7 10]` → `route_profiles_<date>/`
one PNG per line+direction: speed in 100 m bins along the main GTFS shape (all runs of the day) + seconds per run at
platforms / stuck per bin, stop names on the x axis — the city-wide version of the speed strip in
`D:/QGIS/gpx_tracker` (tram audit). Direction = progress along the shape over ±5 intervals. Bins CSV alongside.
Checked against the 23 Nov GPX ride: Pl. Zawiszy, Okopowa, Bitwy Warszawskiej, Banacha exit and the 19 km/h
Most Gdański–Wybrzeże Helskie restriction all show up as typical, not one-off.
Compare: `python live/compare_lines.py [--mode] [--lines 2 31]` overlays route profiles (speed + cumulative minutes
along the route, per direction); default = fastest vs slowest line from `line_speeds_<date>.csv`.

Other analysis: `delay_map.py` (tram slow-time per junction segment), `map_traces.py` (GPS traces PNG/GeoJSON),
`animate_live.py` (video). Status: collection day 2026-05-10 done; full-day scan started 2026-11-24 04:27.
First stuck-point pass done on 04:26–14:00 (top: Bitwy Warszawskiej, Pole Mokotowskie, Okopowa, Pl. Zawiszy — trams, all day,
not depot pull-outs). Per-vehicle loss and line speeds added (04:27–14:19: trams avg 18.7 km/h, fastest 2 / slowest 31; buses 20.0 km/h,
slowest 106/128/356). Next: merge cells into junction clusters, full-day rerun, rush-hour split.

**Tram analysis vs timetable (`live/analysis/`, see its README)** — done for the full day 2026-11-24 (04:27–23:12).
Every tram fix tied to its GTFS trip via line + brigade (97 % of fixes, 6,914 / 7,139 trips), so delay, direction and
actual tram model are known per trip. Junction approaches per travel heading, signal stop rates (OSM signals),
Wilanów corridor verdict (priority works on most of it; Goworka × Puławska N, Belwederska, Św. Bonifacego don't),
priority scenario (≈ 192 tram-h/day, 13.8 trams of peak cycle time), timetable realism, headway regularity, dwell by model.
Run with `py -3.11`. Report artifact: https://claude.ai/artifact/Wrj38hkTvnX8opgUAxGT7w. Next: buses, a full week.
