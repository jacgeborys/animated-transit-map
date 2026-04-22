# Warsaw Transit Density Visualization Pipeline

## Overview
This pipeline creates animated density visualizations of Warsaw's public transit (trams, buses, trains) based on GTFS schedule data. It shows how busy each street segment is based on total daily trips.

## Pipeline Steps

### 1. Route Line Builder
**File:** `core/route_line_builder.py`
- Converts GTFS data into continuous route geometries
- **Output:** `route_lines_continuous.shp` with trip counts per route

### 2. Junction Segmenter
**File:** `core/junction_segmenter.py`
- Finds where routes intersect (junctions)
- Clusters nearby junctions into junction groups (centroids)
- Snaps routes to junction centroids
- Splits routes into segments between junctions
- **Merges segments** used by multiple routes and **sums trip_counts**
- **Output:** `{vehicle}_junction_segments.shp` with total trips per segment

**Key Config:**
```python
TEST_MODE = False  # True = 1km test area, False = full city
TEST_AREA_SIZE = 1000  # Size of test area (meters)
```

**Vehicle-specific snap distances:**
- Trams: 50m
- Buses: 30m
- Trains: 50m

### 3. Animation
**File:** `animate/animate_full_density.py`
- Loads junction segments with trip counts
- Animates vehicles moving along routes
- Shows density as color/brightness based on rolling window occupancy
- **Trams:** Color gradient (#160000 dark → #ed1f1f bright) + glow effect
- **Buses/Trains:** Transparency-based approach

**Key Config:**
```python
FPS = 20
DURATION = 90  # seconds
HOURS = 6  # hours of transit to show
VEHICLE_FILTER = 'Tram'  # 'Tram', 'Bus', 'Train', or None
TEST_AREA_SIZE = 1000  # 1km x 1km test area
```

## Key Outputs

### Junction Segmenter Outputs (per vehicle type):
- `{vehicle}_junction_points.shp` - Actual detected junction locations
- `{vehicle}_junction_groups.shp` - Clustered junction centroids
- `{vehicle}_junction_segments.shp` - **Street segments with total trip counts**

### Animation Output:
- `warsaw_{vehicle}_transit.mp4` - Animated density visualization

## Data Flow

```
GTFS Data
    ↓
route_line_builder.py → route_lines_continuous.shp (with trip_count per route)
    ↓
junction_segmenter.py → junction_segments.shp (with SUMMED trip_count per segment)
    ↓
animate_full_density.py → animation.mp4 (density visualization)
```

## Important Notes

### Junction Segmenter
- **Creates segments between junctions only** (not route start/end currently)
- **Sums trip_counts** from all routes using same segment
- Example: If routes A (100 trips), B (150 trips), C (80 trips) use same segment → segment shows **330 total trips**
- Uses DBSCAN clustering (eps=snap_distance) for junction grouping
- Uses spatial indexing for performance

### Animation
- **Brightness/Color** represents density (vehicles/km in rolling 10-min window)
- **Tram-specific:** Color scale instead of transparency
- **Glow effect:** Scales from 0 (low density) to 0.6 (high density)
- Vehicles follow actual routes, density shown on simplified segments

## Testing Workflow

1. **Test on small area first:**
   ```python
   TEST_MODE = True
   TEST_AREA_SIZE = 1000  # 1km x 1km
   vehicle_types = ['Tram']  # Start with one vehicle type
   ```

2. **Check outputs in QGIS:**
   - Junction points (actual intersections)
   - Junction groups (centroids)
   - Segments (verify trip_count sums correctly)

3. **Run full city:**
   ```python
   TEST_MODE = False
   vehicle_types = ['Tram', 'Bus', 'Train']
   ```

## Known Issues / TODO

- [ ] Route start/end segments not currently generated (only junction-to-junction)
- [ ] Bus/Train color scales not yet defined (using transparency for now)
- [ ] Performance: ~minutes for test area, ~hours for full city buses

## Performance Tips

- Use TEST_MODE for quick iterations
- Process one vehicle type at a time
- Delete output files to force regeneration
- Spatial indexing significantly speeds up processing
