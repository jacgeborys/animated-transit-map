# Warsaw Transit Density Visualization Pipeline

## Active Pipeline (run in this order)

| # | File | Output |
|---|------|--------|
| 0 | Edit `config.py` → set `ANALYSIS_DATE` | — |
| 1 | `core/gtfs_downloader.py` | `data/raw/warsaw_gtfs_*/` |
| 2 | `core/route_line_builder.py` | `data/processed/route_lines_continuous.shp` |
| 3 | `core/junction_segmenter.py` | `data/processed/{vehicle}_junction_segments.shp` |
| 4 | `animate/animate_full_density.py` | `animate/_output/warsaw_transit_density_full.mp4` |

## Key Config

### config.py
```python
ANALYSIS_DATE = "20260427"  # YYYYMMDD
```

### core/junction_segmenter.py
```python
TEST_MODE = False        # True = 1km test area (faster for iteration)
TEST_AREA_SIZE = 1000
```
Vehicle-specific snap distances: Trams 50m, Buses 30m, Trains 50m.

### animate/animate_full_density.py
```python
VEHICLE_FILTER = None    # 'Tram', 'Bus', 'Train', or None for all
FPS = 20
DURATION = 90
HOURS = 6
```
Tram color scale: `#160000` (dark) → `#ed1f1f` (bright) + glow effect.
Bus/Train: transparency-based.

## What Each Script Does

### gtfs_downloader.py
Downloads latest Warsaw GTFS zip from mkuran.pl, extracts to timestamped folder.
GTFS data is date-specific — re-download if target date is more than ~2 months ahead of last download.

### gtfs_parser.py
Called internally by route_line_builder. Do not run directly.

### route_line_builder.py
Converts GTFS shapes + trips into continuous LineString geometries, one per route. Assigns trip counts per route.

### junction_segmenter.py
- Finds where routes intersect (junctions)
- Clusters nearby junctions into groups (DBSCAN, eps=snap_distance)
- Snaps routes to junction centroids
- Splits routes into segments between junctions
- Merges segments used by multiple routes, **sums trip_counts**
- Example: routes A (100), B (150), C (80) sharing a segment → segment shows **330 trips**

### animate_full_density.py
Loads junction segments, animates vehicles moving along routes. Brightness/color represents density (vehicles/km in rolling 10-min window). Glow effect scales 0–0.6 with density.

## Optional Tools

- `visualization/map_generator.py` — static PNG/PDF/SVG transit maps
- `x_capacity_calculator.py` — adds passenger capacity estimates to segments
- `x_create_segment_diagnostic.py` — debug/diagnostic tool

## Outputs (per vehicle type)

- `{vehicle}_junction_points.shp` — actual detected junction locations
- `{vehicle}_junction_groups.shp` — clustered junction centroids
- `{vehicle}_junction_segments.shp` — **street segments with total trip counts** ← main output

## Archive

Old/unused scripts are in `archive/`. Ignore them.

## Known Limitations

- Route start/end segments not generated (only junction-to-junction)
- Bus/Train color scales not yet defined (transparency only)
- Full city buses: processing time is hours

## Claude Workflow

After every code change, always:
1. `git add` the changed files
2. `git commit` with a short descriptive message
3. `git push` to remote

Do this without asking for confirmation — the user expects it to happen automatically after each task.
