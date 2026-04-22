# Complete Pipeline - From Scratch

## Overview

This pipeline processes GTFS data to create transit frequency visualizations and animations.

## Files Archived (No Longer Needed)

These files have been moved to `archive/` because they create unnecessarily complex outputs:

- ❌ `core/3_geometry_processor.py` - Created 100K+ tiny segments
- ❌ `core/4_segment_aggregator.py` - Created 1.2M segments in 1.8GB files

## New Streamlined Pipeline

### Core Files (Run in Order)

1. **`run_pipeline.py`** ⭐ **START HERE**
   - All-in-one pipeline script
   - Downloads GTFS, builds routes, creates segments

2. **`core/1_gtfs_downloader.py`**
   - Downloads Warsaw GTFS data
   - Output: `data/raw/warsaw_gtfs_YYYYMMDD_HHMMSS/`

3. **`core/2_gtfs_parser.py`**
   - Parses GTFS schedule
   - Filters by date and time
   - Called internally by other scripts

4. **`core/5_route_line_builder.py`**
   - Creates continuous route lines
   - Output: `data/processed/route_lines_continuous.shp` (17MB, ~2000 routes)
   - Output: `data/processed/schedule_for_animation.csv` (73MB)

5. **`visualization/segmenter.py`**
   - Splits routes at intersections (node-to-node)
   - Merges overlapping parallel routes
   - Output: `data/processed/street_segments.shp` (~50MB, ~10-20K segments)

### Optional Files

- **`x_capacity_calculator.py`** - Adds passenger capacity estimates
- **`x_create_segment_diagnostic.py`** - Diagnostic tools

### Animation Files

- **`animate/animate_full_density.py`** - Main animation script

## Quick Start (From Scratch)

### Step 1: Run Complete Pipeline

```bash
python run_pipeline.py
```

This will:
1. Download latest GTFS data (~30 seconds)
2. Build continuous route lines (~1 minute)
3. Export schedule for animation
4. Create street segments (intersection-based) (~5-10 minutes)

**Total time**: ~10-15 minutes

### Step 2: Run Animation

```bash
python animate/animate_full_density.py
```

This creates: `animate/_output/warsaw_transit_density_full.mp4`

## Pipeline Outputs

### data/processed/

- **`route_lines_continuous.shp`** (17MB)
  - One continuous LineString per route
  - Properties: `shape_id`, `route_id`, `vehicle`, `trip_count`, `length`
  - Used for: Animation vehicle movement

- **`schedule_for_animation.csv`** (73MB)
  - All stop times with route/vehicle info
  - Properties: `trip_id`, `route_id`, `shape_id`, `vehicle`, `departure_time_str`, `stop_sequence`
  - Used for: Animation timing

- **`street_segments.shp`** (~50MB)
  - Intersection-to-intersection segments
  - Overlapping routes merged
  - Properties: `geometry`, `route_id`, `vehicle`, `trip_count`, `route_count`, `length`, `bearing`, `route_ids`
  - Used for: Animation display, Static maps in QGIS

### animate/_output/

- **`*.mp4`** - Animation videos

### out/maps/

- **`*.png`**, **`*.geojson`** - Static map outputs

## Command Line Options

### Skip Download (Use Existing GTFS)

```bash
python run_pipeline.py --skip-download
```

### Skip Segment Creation (Use Existing)

```bash
python run_pipeline.py --skip-segments
```

### Both

```bash
python run_pipeline.py --skip-download --skip-segments
```

## What Each Step Produces

| Step | File | Size | Count | Purpose |
|------|------|------|-------|---------|
| 1. Download | `raw/warsaw_gtfs_*/` | ~20MB | N/A | GTFS source data |
| 2. Parse | *(in memory)* | - | ~100K stops | Schedule filtering |
| 3. Routes | `route_lines_continuous.shp` | 17MB | ~2,000 | Vehicle paths |
| 4. Schedule | `schedule_for_animation.csv` | 73MB | ~800K | Departure times |
| 5. Segments | `street_segments.shp` | 50MB | ~10-20K | Street segments |

## Use Cases

### For Animation
```bash
python run_pipeline.py
python animate/animate_full_density.py
```

### For Static Maps (QGIS)
```bash
python run_pipeline.py
# Then load street_segments.shp in QGIS
# Style by: trip_count or route_count
```

### For Capacity Analysis
```bash
python run_pipeline.py
python x_capacity_calculator.py
# Creates: aggregated_*_capacity.shp files
```

## Troubleshooting

**"Route lines not found"**
→ Run `python run_pipeline.py` first

**"Only 2 segments with activity"**
→ This was a bug, now fixed! You should see thousands of segments.

**"Animation too slow"**
→ Reduce FPS or DURATION in `animate/animate_full_density.py`

**"Out of memory"**
→ Your animation is trying to process too much. Reduce `HOURS` or `FRAME_SIZE`

## File Structure

```
gtfs_schedules_city/
├── run_pipeline.py              ⭐ START HERE - Complete pipeline
├── core/
│   ├── 1_gtfs_downloader.py     Step 1: Download GTFS
│   ├── 2_gtfs_parser.py         Step 2: Parse schedule
│   └── 5_route_line_builder.py  Step 3: Build routes
├── visualization/
│   └── segmenter.py             Step 4: Create street segments
├── animate/
│   ├── animate_full_density.py  Main animation
│   └── _output/                 Videos go here
├── data/
│   ├── raw/                     Downloaded GTFS
│   ├── processed/               Intermediate outputs ⭐
│   └── output/                  (unused - from old pipeline)
├── out/
│   └── maps/                    Static map outputs
├── archive/                     Old heavy aggregator files
└── config.py                    Configuration
```

## Next Steps

1. Run the pipeline: `python run_pipeline.py`
2. Check outputs in `data/processed/`
3. Run animation: `python animate/animate_full_density.py`
4. Load `street_segments.shp` in QGIS for static maps

## Configuration

Edit `config.py` to customize:
- `ANALYSIS_DATE` - Which date to analyze
- `PEAK_HOURS` - Time windows to analyze
- `BUFFER_DISTANCE` - Tolerance for merging parallel routes
- `DIRECTION_TOLERANCE` - Bearing tolerance for matching

Edit `animate/animate_full_density.py` to customize:
- `FPS`, `DURATION`, `HOURS` - Animation timing
- `FRAME_SIZE` - Map extent
- `BASE_BRIGHTNESS` - Base line visibility
- `MAX_VEHICLES_FOR_MAX_BRIGHTNESS` - Brightness calibration
