"""
Full City Transit Animation - Optimized
========================================
Uses pre-computed junction segments for accurate, smooth visualization.

Requirements:
  - Run core/junction_segmenter.py first

Features:
  - Junction-based segments (natural convergence/divergence points)
  - Density-based brightness (occupancy rate calculation)
  - Spatial index optimization (fast for many routes)
  - All transit types (trams, buses, trains)
"""
import time

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, LineString
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def interpolate_color(color1_hex, color2_hex, t):
    """Interpolate between two hex colors based on t (0.0 to 1.0)"""
    # Convert hex to RGB
    c1 = tuple(int(color1_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    c2 = tuple(int(color2_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # Interpolate
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)

    # Convert back to hex
    return f'#{r:02x}{g:02x}{b:02x}'

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
ROUTE_LINES = PROJECT_ROOT / "data" / "processed" / "route_lines_continuous.shp"
SCHEDULE = PROJECT_ROOT / "data" / "processed" / "schedule_for_animation.csv"

# Auto-detect latest GTFS download for stop-by-stop interpolation
_gtfs_dirs = sorted((PROJECT_ROOT / "data" / "raw").glob("warsaw_gtfs_*"))
GTFS_STOP_TIMES = _gtfs_dirs[-1] / "stop_times.txt" if _gtfs_dirs else None

# Animation settings
# FPS = 20
# DURATION = 90  # seconds
FPS = 25
DURATION = 180  # seconds
HOURS = 21  # hours of transit to show
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
VEHICLE_FILTER = None  # 'Tram', 'Bus', 'Train', or None for all

# Dynamic paths based on vehicle filter
if VEHICLE_FILTER:
    vehicle_lower = VEHICLE_FILTER.lower()
    JUNCTION_SEGMENTS = PROJECT_ROOT / "data" / "processed" / f"{vehicle_lower}_junction_segments.shp"
    OUTPUT = PROJECT_ROOT / "animate" / "_output" / f"warsaw_{vehicle_lower}_transit.mp4"
else:
    JUNCTION_SEGMENTS = PROJECT_ROOT / "data" / "processed" / "junction_segments.shp"
    OUTPUT = PROJECT_ROOT / "animate" / "_output" / "warsaw_transit_full.mp4"

# Map extent - Central Warsaw
CENTRAL_WARSAW_X = 638000
CENTRAL_WARSAW_Y = 487000
FRAME_SIZE = 22000  # Base frame size (metres)

# Dynamic zoom settings
# Zoom timeline follows the day: wide at dawn, tight at noon, back out by 14:00
ZOOM_START_SIZE  = 30000   # 1.5x base — zoomed out at animation start (~4:00)
ZOOM_MIN_SIZE    = 8000   # 0.7x base — tightest zoom at noon (12:00)
ZOOM_END_SIZE    = FRAME_SIZE  # back to base by 14:00, held for rest of day
ZOOM_IN_START_H  = 4       # hour when zoom-in begins
ZOOM_IN_END_H    = 7      # hour when tightest zoom is reached
ZOOM_OUT_END_H   = 9     # hour when zoom settles back to base


def get_frame_size(current_seconds):
    """Return frame size (metres) for the given animation time using smooth interpolation."""
    def smoothstep(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)

    t_in_start  = ZOOM_IN_START_H  * 3600
    t_in_end    = ZOOM_IN_END_H    * 3600
    t_out_end   = ZOOM_OUT_END_H   * 3600

    if current_seconds <= t_in_start:
        return ZOOM_START_SIZE
    elif current_seconds <= t_in_end:
        t = smoothstep((current_seconds - t_in_start) / (t_in_end - t_in_start))
        return ZOOM_START_SIZE + (ZOOM_MIN_SIZE - ZOOM_START_SIZE) * t
    elif current_seconds <= t_out_end:
        t = smoothstep((current_seconds - t_in_end) / (t_out_end - t_in_end))
        return ZOOM_MIN_SIZE + (ZOOM_END_SIZE - ZOOM_MIN_SIZE) * t
    else:
        return ZOOM_END_SIZE

# Average vehicle speeds (m/s) — used to estimate trip durations for animation
# Tram: ~16 km/h, Bus: ~20 km/h, Train: ~54 km/h, Metro: ~36 km/h
VEHICLE_SPEEDS = {'Tram': 4.5, 'Bus': 5.5, 'Train': 15.0, 'Metro': 10.0}
DEFAULT_SPEED = 4.7  # fallback

# Visual settings
COLORS = {'Tram': '#FF7075', 'Bus': '#B46EFC', 'Train': '#6BC9C6', 'Metro': '#4FC3F7'}
VEHICLE_SIZES = {'Tram': 16, 'Bus': 12, 'Train': 19, 'Metro': 22}
VEHICLE_SIZES = 0.8 * pd.Series(VEHICLE_SIZES)  # Scale down for better proportions
LINE_WIDTHS = {'Tram': 1.5, 'Bus': 1.2, 'Train': 1.5, 'Metro': 2.2}
GLOW_WIDTH = 4.0
GLOW_ALPHA = 0.7
BASE_BRIGHTNESS = 0.35
MAX_BRIGHTNESS = 1.0
OUTLINE_COLORS = {'Tram': '#1a0003', 'Bus': '#0f0018', 'Train': '#001410', 'Metro': '#000e1a'}
VEHICLE_MARKERS = {'Tram': 'D', 'Bus': 'o', 'Train': '^', 'Metro': 's'}

# Color gradients for density visualization
COLOR_GRADIENTS = {
    'Tram':  {'dark': '#3d0005', 'bright': '#ff4444'},
    'Bus':   {'dark': '#2d0040', 'bright': '#cc33ff'},
    'Train': {'dark': '#0f3d38', 'bright': '#52c5b8'},
    'Metro': {'dark': '#001e30', 'bright': '#6dd5fa'},
}

# Z-order layering (background → trains → buses → trams → metro on top)
Z_ORDERS = {
    'Train': {'glow': 10, 'line': 20},
    'Bus':   {'glow': 30, 'line': 40},
    'Tram':  {'glow': 50, 'line': 60},
    'Metro': {'glow': 70, 'line': 80},  # metro always on top — only 2 lines, very distinct
}

# Background map layers (OSM data)
OSM_DIR = Path(r"D:\QGIS\mapy_warszawy_misc\data\osm")
BACKGROUND_LAYERS = {
    'forests':   OSM_DIR / 'forests.gpkg',
    'water':     OSM_DIR / 'water.gpkg',
    'roads':     OSM_DIR / 'roads.shp',
}
LAYER_STYLES = {
    'forests':   {'fc': '#162e16', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'water':     {'fc': '#0f2e45', 'ec': 'none',    'lw': 0,   'zorder': 2},
    'roads':     {'fc': 'none',    'ec': '#383838', 'lw': 0.3, 'zorder': 4},  # default (lowest tier)
}

# Road width tiers by highway class — field name is 'fclass' in Geofabrik OSM exports
ROAD_HIGHWAY_FIELD = 'fclass'
ROAD_TIERS = [
    ({'motorway', 'motorway_link', 'expressway'},        1.4),
    ({'trunk', 'trunk_link', 'primary', 'primary_link'}, 0.85),
    ({'secondary', 'secondary_link'},                    0.5),
]

STREAK_LENGTH = 150

# Density calculation settings
ROLLING_WINDOW_MINUTES = 15
# Direct brightness calibration — set to the vehicles/km value that should hit max brightness.
# Segments above this value are clamped to full brightness.
# Tune based on "Observed max density" printed at end of each run.
# Rule of thumb: set to ~70-80% of observed max so peak segments glow fully.
DENSITY_FOR_MAX_BRIGHTNESS = 40.0  # vehicles/km


def create_animation():
    """Create animation using pre-computed junction segments"""

    # Load pre-computed segments
    if VEHICLE_FILTER:
        # Load single vehicle type
        if not JUNCTION_SEGMENTS.exists():
            logger.error(f"Junction segments not found: {JUNCTION_SEGMENTS}")
            logger.error("Run core/junction_segmenter.py first")
            return 1

        logger.info(f"Loading junction segments from {JUNCTION_SEGMENTS}...")
        segments = gpd.read_file(JUNCTION_SEGMENTS)
        logger.info(f"Loaded {len(segments)} junction-based segments")
    else:
        # Load and combine all vehicle types
        logger.info("Loading junction segments for all vehicle types...")
        all_segments = []

        for vehicle_type in ['tram', 'bus', 'train', 'metro']:
            vehicle_file = PROJECT_ROOT / "data" / "processed" / f"{vehicle_type}_junction_segments.shp"
            if vehicle_file.exists():
                veh_segs = gpd.read_file(vehicle_file)
                # Ensure vehicle_ty field is set correctly (capitalize first letter)
                veh_segs['vehicle_ty'] = vehicle_type.capitalize()
                all_segments.append(veh_segs)
                logger.info(f"  Loaded {len(veh_segs)} {vehicle_type} segments")
            else:
                logger.warning(f"  {vehicle_type}_junction_segments.shp not found, skipping")

        if not all_segments:
            logger.error("No junction segment files found!")
            return 1

        segments = gpd.GeoDataFrame(pd.concat(all_segments, ignore_index=True))
        logger.info(f"Combined total: {len(segments)} junction-based segments")

    # Parse route_ids from comma-separated string back to list
    segments['route_ids_list'] = segments['route_ids'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

    # Load routes
    logger.info("Loading continuous routes...")
    routes = gpd.read_file(ROUTE_LINES)
    routes['route_id'] = routes['route_id'].astype(str)

    # Filter by vehicle type if specified
    if VEHICLE_FILTER:
        routes = routes[routes['vehicle'] == VEHICLE_FILTER].copy()
        logger.info(f"Filtered to {len(routes)} {VEHICLE_FILTER} route shapes")
    else:
        logger.info(f"Loaded {len(routes)} route shapes (all transit types)")

    # Load schedule
    logger.info("Loading schedule...")
    schedule = pd.read_csv(SCHEDULE)
    schedule['route_id'] = schedule['route_id'].astype(str)

    # Filter schedule by vehicle type if specified
    if VEHICLE_FILTER:
        schedule = schedule[schedule['vehicle'] == VEHICLE_FILTER].copy()
        logger.info(f"Filtered schedule to {len(schedule)} {VEHICLE_FILTER} entries")

    # Parse times
    schedule['departure_time'] = pd.to_datetime(
        schedule['departure_time_str'], format='%H:%M:%S', errors='coerce'
    ).dt.time
    schedule = schedule.dropna(subset=['departure_time'])
    schedule['departure_seconds'] = schedule['departure_time'].apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second
    )

    vehicle_desc = f"{VEHICLE_FILTER} " if VEHICLE_FILTER else ""
    logger.info(f"Schedule: {len(schedule)} {vehicle_desc}entries, {schedule['trip_id'].nunique()} trips")

    # Map shape_ids to segment indices
    route_to_segments = {}
    for idx, segment in segments.iterrows():
        # A merged segment serves multiple routes
        for route_id in segment['route_ids_list']:
            # Find all shape_ids for this route_id
            matching_shapes = routes[routes['route_id'] == route_id]['shape_id'].tolist()
            for shape_id in matching_shapes:
                if shape_id not in route_to_segments:
                    route_to_segments[shape_id] = []
                if idx not in route_to_segments[shape_id]:
                    route_to_segments[shape_id].append(idx)

    # Build route lookup
    route_lookup = {}
    for idx, row in routes.iterrows():
        route_lookup[row['shape_id']] = {
            'geometry': row.geometry,
            'vehicle': row['vehicle'],
            'route_id': row['route_id']
        }

    # Load stop-by-stop schedules for accurate position interpolation
    trip_stop_schedule = {}
    if GTFS_STOP_TIMES and GTFS_STOP_TIMES.exists():
        logger.info(f"Loading stop times from {GTFS_STOP_TIMES.parent.name}...")
        trip_ids_needed = set(schedule['trip_id'].unique())
        st_raw = pd.read_csv(GTFS_STOP_TIMES, low_memory=False)

        def _parse_time(t):
            h, m, s = t.split(':')
            return int(h) * 3600 + int(m) * 60 + int(s)

        # Build template lookup for frequency-based trips (e.g. metro: M1:PcM:KAB__25200)
        # Template trips have times starting at 00:00:00 — we add the departure offset
        freq_templates = {}
        for trip_id, grp in st_raw.groupby('trip_id'):
            grp = grp.sort_values('stop_sequence')
            max_dist = grp['shape_dist_traveled'].max()
            if max_dist <= 0 or len(grp) < 2:
                continue
            offsets = grp['departure_time'].apply(_parse_time).values.astype(np.float64)
            progresses = (grp['shape_dist_traveled'] / max_dist).values.astype(np.float64)
            freq_templates[trip_id] = (offsets, progresses)

        # Match schedule trips: direct match first, then frequency-based (template__offset)
        st_filtered = st_raw[st_raw['trip_id'].isin(trip_ids_needed)]
        for trip_id, grp in st_filtered.groupby('trip_id'):
            grp = grp.sort_values('stop_sequence')
            max_dist = grp['shape_dist_traveled'].max()
            if max_dist <= 0 or len(grp) < 2:
                continue
            times = grp['departure_time'].apply(_parse_time).values.astype(np.float64)
            progresses = (grp['shape_dist_traveled'] / max_dist).values.astype(np.float64)
            trip_stop_schedule[trip_id] = (times, progresses)

        # Frequency-based trips: resolve template__offset pattern
        freq_resolved = 0
        for trip_id in trip_ids_needed:
            if trip_id in trip_stop_schedule:
                continue
            if '__' in trip_id:
                parts = trip_id.rsplit('__', 1)
                template_id, offset_str = parts[0], parts[1]
                if template_id in freq_templates and offset_str.lstrip('-').isdigit():
                    offsets, progresses = freq_templates[template_id]
                    departure_sec = float(offset_str)
                    trip_stop_schedule[trip_id] = (offsets + departure_sec, progresses)
                    freq_resolved += 1

        logger.info(f"  Direct stop schedules: {len(trip_stop_schedule) - freq_resolved:,} trips")
        logger.info(f"  Frequency-based (metro etc.): {freq_resolved:,} trips")
        logger.info(f"  Speed-based fallback: {len(trip_ids_needed) - len(trip_stop_schedule):,} trips")
    else:
        logger.warning("stop_times.txt not found — using speed-based interpolation for all vehicles")

    start_seconds = 4 * 3600
    duration_seconds = HOURS * 3600

    density_for_max_brightness = DENSITY_FOR_MAX_BRIGHTNESS
    logger.info(f"Brightness calibration: max brightness at {density_for_max_brightness:.1f} vehicles/km")

    # Build spatial index for fast segment lookups
    logger.info("Building spatial index for segments...")
    from shapely.strtree import STRtree
    segment_geometries = [seg.geometry.buffer(30) for _, seg in segments.iterrows()]
    segment_spatial_index = STRtree(segment_geometries)

    # Pre-extract segment data to avoid repeated DataFrame lookups (major speedup!)
    logger.info("Pre-extracting segment data for faster rendering...")
    segment_coords = []
    segment_vtypes = []
    segment_lengths = []

    for idx, segment in segments.iterrows():
        if hasattr(segment.geometry, 'coords'):
            x, y = segment.geometry.xy
            segment_coords.append((x, y))
            segment_vtypes.append(segment.get('vehicle_ty', 'Tram'))
            segment_lengths.append(segment['length_km'])
        else:
            segment_coords.append(None)
            segment_vtypes.append('Tram')
            segment_lengths.append(0)

    logger.info(f"Pre-extracted {len(segment_coords)} segment geometries")

    # Setup figure (high quality)
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='#000000')
    ax.set_facecolor('#000000')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Set initial extent to the widest zoom level so background layers cover full range
    half_size = ZOOM_START_SIZE / 2
    xmin, xmax = CENTRAL_WARSAW_X - half_size, CENTRAL_WARSAW_X + half_size
    ymin, ymax = CENTRAL_WARSAW_Y - half_size, CENTRAL_WARSAW_Y + half_size
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Draw static background map layers
    logger.info("Drawing background map layers...")
    from shapely.geometry import box as shapely_box
    clip_box = shapely_box(xmin, ymin, xmax, ymax)
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs='EPSG:2180')

    for name, path in BACKGROUND_LAYERS.items():
        if not path.exists():
            logger.warning(f"  Background layer not found, skipping: {path.name}")
            continue
        try:
            bg = gpd.read_file(path)
            if bg.crs is None:
                bg = bg.set_crs('EPSG:4326')
            if bg.crs.to_epsg() != 2180:
                bg = bg.to_crs('EPSG:2180')
            # Fix invalid polygons (buffer(0) destroys lines — skip for those)
            if bg.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).any():
                bg['geometry'] = bg.geometry.buffer(0)
            bg = bg[bg.geometry.is_valid & ~bg.geometry.is_empty]
            bg = gpd.clip(bg, clip_gdf)
            if bg.empty:
                logger.warning(f"  {name}: no features in frame, skipping")
                continue
            style = LAYER_STYLES[name]
            if name == 'roads' and ROAD_HIGHWAY_FIELD in bg.columns:
                # Draw road tiers with varying widths (thicker = higher class)
                drawn = 0
                classified = set()
                for classes, lw in ROAD_TIERS:
                    tier_roads = bg[bg[ROAD_HIGHWAY_FIELD].isin(classes)]
                    if not tier_roads.empty:
                        tier_roads.plot(ax=ax, facecolor='none', edgecolor=style['ec'],
                                        linewidth=lw, zorder=style['zorder'])
                        drawn += len(tier_roads)
                    classified |= classes
                # Draw remaining roads at default (lowest tier) width
                rest = bg[~bg[ROAD_HIGHWAY_FIELD].isin(classified)]
                if not rest.empty:
                    rest.plot(ax=ax, facecolor='none', edgecolor=style['ec'],
                              linewidth=style['lw'], zorder=style['zorder'])
                    drawn += len(rest)
                logger.info(f"  {name}: {drawn} features drawn (tiered widths)")
            else:
                bg.plot(ax=ax, facecolor=style['fc'], edgecolor=style['ec'],
                        linewidth=style['lw'], zorder=style['zorder'])
                logger.info(f"  {name}: {len(bg)} features drawn")
        except Exception as e:
            logger.warning(f"  {name}: failed to load ({e}), skipping")

    # Draw static base network using LineCollection for efficiency
    logger.info("Drawing static base network with LineCollections...")
    from matplotlib.collections import LineCollection

    # Group segments by vehicle type
    segments_by_type = {'Tram': [], 'Bus': [], 'Train': [], 'Metro': []}

    for idx in range(len(segments)):
        coords = segment_coords[idx]
        if coords is None:
            continue

        x, y = coords
        vtype = segment_vtypes[idx]

        # Convert to line segment format
        points = np.column_stack([x, y])
        segments_by_type[vtype].append(points)

    # Create LineCollection for each vehicle type
    for vtype in ['Train', 'Bus', 'Tram', 'Metro']:  # Draw in z-order (metro on top)
        if not segments_by_type[vtype]:
            continue

        gradient = COLOR_GRADIENTS.get(vtype, COLOR_GRADIENTS['Tram'])
        dark_color = gradient['dark']
        line_width = LINE_WIDTHS.get(vtype, 1.5)
        z_order = Z_ORDERS.get(vtype, Z_ORDERS['Tram'])

        lc = LineCollection(segments_by_type[vtype], colors=dark_color,
                           linewidths=line_width, zorder=z_order['line'],
                           capstyle='round', joinstyle='round')
        ax.add_collection(lc)

        logger.info(f"  Added {len(segments_by_type[vtype])} {vtype} base segments")

    logger.info("Static base network added! Will only draw bright segments each frame.")

    # Animation state
    segment_vehicle_visits = [[] for _ in range(len(segments))]
    vehicles = []
    total_frames = FPS * DURATION
    rolling_window_seconds = ROLLING_WINDOW_MINUTES * 60
    observed_max_density = 0.0  # Track actual max density during animation
    dynamic_artists = []  # Track artists created each frame for cleanup

    def update_frame(frame_num):
        nonlocal vehicles, dynamic_artists

        # Remove dynamic artists from previous frame (keep base network)
        for artist in dynamic_artists:
            artist.remove()
        dynamic_artists = []

        # Calculate current time
        current_seconds = start_seconds + (frame_num / total_frames) * duration_seconds
        current_time = f"{int(current_seconds // 3600):02d}:{int((current_seconds % 3600) // 60):02d}"
        frame_duration = duration_seconds / total_frames

        # Dynamic zoom — update axes limits each frame
        half = get_frame_size(current_seconds) / 2
        ax.set_xlim(CENTRAL_WARSAW_X - half, CENTRAL_WARSAW_X + half)
        ax.set_ylim(CENTRAL_WARSAW_Y - half, CENTRAL_WARSAW_Y + half)

        # Add new vehicles
        new_trips = schedule[
            (schedule['departure_seconds'] >= current_seconds - frame_duration/2) &
            (schedule['departure_seconds'] <= current_seconds + frame_duration/2) &
            (schedule['stop_sequence'] == 1)
        ]

        for _, trip in new_trips.iterrows():
            trip_id = trip['trip_id']
            shape_id = trip['shape_id']
            if shape_id in route_lookup:
                route_info = route_lookup[shape_id]

                if trip_id in trip_stop_schedule:
                    sched = trip_stop_schedule[trip_id]
                    end_abs = float(sched[0][-1])
                else:
                    speed = VEHICLE_SPEEDS.get(route_info['vehicle'], DEFAULT_SPEED)
                    end_abs = current_seconds + route_info['geometry'].length / speed
                    sched = None

                vehicles.append({
                    'id': trip_id,
                    'shape_id': shape_id,
                    'vehicle': route_info['vehicle'],
                    'route': route_info['geometry'],
                    'route_id': route_info['route_id'],
                    'start_time': current_seconds,
                    'end_abs': end_abs,
                    'sched': sched,
                    'current_segment': None
                })

        # Update vehicles and track segment occupancy
        active_vehicles = []
        for vehicle in vehicles:
            if current_seconds > vehicle['end_abs']:
                # Vehicle finished - mark exit time
                if vehicle['current_segment'] is not None:
                    seg_idx = vehicle['current_segment']
                    visits = segment_vehicle_visits[seg_idx]
                    for visit in reversed(visits):
                        if visit[0] == vehicle['id'] and visit[2] is None:
                            visit[2] = current_seconds
                            break
                continue

            if vehicle['sched'] is not None:
                times, progresses = vehicle['sched']
                progress = float(np.interp(current_seconds, times, progresses))
            else:
                elapsed = current_seconds - vehicle['start_time']
                duration = vehicle['end_abs'] - vehicle['start_time']
                progress = elapsed / duration if duration > 0 else 0

            position = vehicle['route'].interpolate(progress, normalized=True)
            vehicle['position'] = position
            vehicle['progress'] = progress

            old_segment = vehicle.get('current_segment', None)

            # Find which segment the vehicle is on (using spatial index)
            shape_id = vehicle['shape_id']
            new_segment = None

            if shape_id in route_to_segments:
                # Use spatial index to find nearby segments quickly
                nearby_indices = segment_spatial_index.query(position)

                # Check only nearby segments from this route
                for seg_idx in route_to_segments[shape_id]:
                    if seg_idx in nearby_indices:
                        new_segment = seg_idx
                        break

            # Handle segment transitions
            if new_segment != old_segment:
                # Exit old segment
                if old_segment is not None:
                    visits = segment_vehicle_visits[old_segment]
                    for visit in reversed(visits):
                        if visit[0] == vehicle['id'] and visit[2] is None:
                            visit[2] = current_seconds
                            break

                # Enter new segment
                if new_segment is not None:
                    segment_vehicle_visits[new_segment].append([vehicle['id'], current_seconds, None])
                    vehicle['current_segment'] = new_segment

            active_vehicles.append(vehicle)

        vehicles = active_vehicles

        # Calculate brightness based on occupancy rate
        # Optimization: Track which segments have visits to avoid iterating through all segments
        segment_brightness = np.full(len(segments), BASE_BRIGHTNESS, dtype=np.float32)
        cutoff_time = current_seconds - rolling_window_seconds
        active_segments = set()

        # First pass: collect active segments and clean old visits
        for idx, visits in enumerate(segment_vehicle_visits):
            if visits:
                # Remove old visits outside rolling window
                segment_vehicle_visits[idx] = [v for v in visits if v[2] is None or v[2] >= cutoff_time]
                if segment_vehicle_visits[idx]:
                    active_segments.add(idx)

        # Second pass: calculate brightness only for active segments
        for idx in active_segments:
            visits = segment_vehicle_visits[idx]

            # Calculate total vehicle-seconds in the window
            total_vehicle_seconds = 0
            for vehicle_id, entry_time, exit_time in visits:
                actual_entry = max(entry_time, cutoff_time)
                actual_exit = min(exit_time if exit_time else current_seconds, current_seconds)

                if actual_exit > actual_entry:
                    total_vehicle_seconds += (actual_exit - actual_entry)

            # Occupancy rate: average number of vehicles present
            occupancy_rate = total_vehicle_seconds / rolling_window_seconds

            # Convert to density (vehicles/km)
            length_km = segment_lengths[idx]
            density = occupancy_rate / length_km if length_km > 0 else 0

            # Track actual max density observed
            nonlocal observed_max_density
            if density > observed_max_density:
                observed_max_density = density

            # Convert density to brightness
            brightness_ratio = density / density_for_max_brightness if density_for_max_brightness > 0 else 0
            brightness = BASE_BRIGHTNESS + (MAX_BRIGHTNESS - BASE_BRIGHTNESS) * brightness_ratio
            brightness = min(brightness, MAX_BRIGHTNESS)

            segment_brightness[idx] = brightness

        # Draw only glowing segments (brightness > BASE_BRIGHTNESS)
        # Base network is already in the cached background
        for idx in range(len(segments)):
            brightness = segment_brightness[idx]

            # Skip segments at base brightness - they're in the static background
            if brightness <= BASE_BRIGHTNESS + 0.001:
                continue

            coords = segment_coords[idx]
            if coords is None:
                continue

            x, y = coords
            vtype = segment_vtypes[idx]

            # Normalize brightness to [0, 1]
            normalized_brightness = min(max(brightness, 0.0), 1.0)

            # Get color gradient for this vehicle type
            gradient = COLOR_GRADIENTS.get(vtype, COLOR_GRADIENTS['Tram'])
            dark_color = gradient['dark']
            bright_color = gradient['bright']
            line_width = LINE_WIDTHS.get(vtype, 1.5)
            z_order = Z_ORDERS.get(vtype, Z_ORDERS['Tram'])

            # Interpolate color based on brightness
            line_color = interpolate_color(dark_color, bright_color, normalized_brightness)

            # Glow: scale from 0 (at base brightness) to GLOW_ALPHA (at max brightness)
            glow_progress = (normalized_brightness - BASE_BRIGHTNESS) / (MAX_BRIGHTNESS - BASE_BRIGHTNESS)
            glow_alpha = max(0, glow_progress * GLOW_ALPHA)

            # Glow with bright color and variable transparency (behind the line)
            glow_line = ax.plot(x, y, color=bright_color, linewidth=GLOW_WIDTH,
                   alpha=glow_alpha,
                   solid_capstyle='round',
                   solid_joinstyle='round', zorder=z_order['glow'])[0]
            dynamic_artists.append(glow_line)

            # Main line with interpolated color (full opacity, on top of glow)
            main_line = ax.plot(x, y, color=line_color, linewidth=line_width,
                   alpha=1.0, solid_capstyle='round',
                   solid_joinstyle='round', zorder=z_order['line'])[0]
            dynamic_artists.append(main_line)

        # Draw vehicles grouped by type
        vehicles_by_type = {'Tram': [], 'Bus': [], 'Train': [], 'Metro': []}
        streaks_by_type = {'Tram': [], 'Bus': [], 'Train': [], 'Metro': []}

        for vehicle in vehicles:
            if 'position' in vehicle:
                current_pos = vehicle['position']
                vtype = vehicle['vehicle']
                vehicles_by_type[vtype].append(current_pos)

                # Create streak
                progress = vehicle['progress']
                route = vehicle['route']
                route_length = route.length

                streak_coords = [(current_pos.x, current_pos.y)]
                for i in range(1, 4):
                    back_distance = i * (STREAK_LENGTH / 3)
                    back_progress = progress - (back_distance / route_length)
                    if back_progress >= 0:
                        back_pos = route.interpolate(back_progress, normalized=True)
                        streak_coords.append((back_pos.x, back_pos.y))

                if len(streak_coords) > 1:
                    streaks_by_type[vtype].append(streak_coords)

        # Draw each vehicle type
        for vtype in ['Tram', 'Bus', 'Train', 'Metro']:
            positions = vehicles_by_type[vtype]
            streaks = streaks_by_type[vtype]

            if positions:
                color = COLORS[vtype]
                outline_color = OUTLINE_COLORS[vtype]
                size = VEHICLE_SIZES[vtype]

                # Streaks
                for streak_coords in streaks:
                    xs, ys = [c[0] for c in streak_coords], [c[1] for c in streak_coords]
                    streak_line = ax.plot(xs, ys, color=color, linewidth=3, alpha=0.2,
                           solid_capstyle='round', zorder=70)[0]
                    dynamic_artists.append(streak_line)

                # Dots
                xs, ys = [p.x for p in positions], [p.y for p in positions]
                dots = ax.scatter(xs, ys, s=size, color=color, alpha=0.8,
                          marker=VEHICLE_MARKERS[vtype],
                          edgecolors=outline_color, linewidths=0.6, zorder=80)
                dynamic_artists.append(dots)

        # Text
        title_text = ax.text(0.02, 0.98, f"Warsaw Public Transit - 27.04.2026 {current_time}",
               transform=ax.transAxes, fontsize=24, color='white',
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), zorder=100)
        dynamic_artists.append(title_text)

        count_text = ax.text(0.02, 0.92, f"Active vehicles: {len(vehicles)}",
               transform=ax.transAxes, fontsize=16, color='white',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), zorder=100)
        dynamic_artists.append(count_text)

        credit_text = ax.text(0.98, 0.02, "© Jacek Gęborys",
               transform=ax.transAxes, fontsize=12, color='white',
               verticalalignment='bottom', horizontalalignment='right',
               alpha=0.6, zorder=100)
        dynamic_artists.append(credit_text)

        if frame_num % 60 == 0:
            pct = frame_num / total_frames * 100
            logger.info(f"Frame {frame_num}/{total_frames} ({pct:.0f}%) - {len(vehicles)} active")
            logger.info(f"  Brightness range: {segment_brightness.min():.3f} - {segment_brightness.max():.3f}")
            logger.info(f"  Observed max density so far: {observed_max_density:.1f} vehicles/km")

    # Save — incremental frame-by-frame rendering (safe to interrupt)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {OUTPUT}...")
    plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

    if not Path(FFMPEG_PATH).exists():
        logger.error(f"FFmpeg not found at {FFMPEG_PATH}")
        logger.error("Install FFmpeg or update FFMPEG_PATH")
        plt.close()
        return 1

    writer = animation.FFMpegWriter(fps=FPS, codec='libx264', bitrate=5000,
                                   metadata={'artist': 'Jacek Gęborys'},
                                   extra_args=['-pix_fmt', 'yuv420p',
                                               '-movflags', '+frag_keyframe+empty_moov'])

    logger.info(f"Rendering {total_frames} frames (FPS={FPS}, Duration={DURATION}s)...")
    logger.info("Frames are written to disk incrementally — safe to interrupt at any time.")
    try:
        with writer.saving(fig, OUTPUT, dpi=100):
            for frame_num in range(total_frames):
                update_frame(frame_num)
                writer.grab_frame()
        logger.info(f"Animation saved successfully!")
    except KeyboardInterrupt:
        logger.info("Interrupted — partial video saved to disk.")
        plt.close()
        return 0
    except Exception as e:
        logger.error(f"Error saving animation: {e}")
        logger.error("Try reducing DURATION, FPS, or figure size")
        plt.close()
        return 1
    plt.close()

    logger.info(f"Done! {OUTPUT}")
    logger.info(f"Density statistics:")
    logger.info(f"  Observed max density:          {observed_max_density:.1f} vehicles/km")
    logger.info(f"  DENSITY_FOR_MAX_BRIGHTNESS:    {density_for_max_brightness:.1f} vehicles/km")
    logger.info(f"  Tip: set DENSITY_FOR_MAX_BRIGHTNESS = {observed_max_density * 0.75:.1f} to push peak segments to full brightness")
    return 0


if __name__ == "__main__":
    if not ROUTE_LINES.exists() or not SCHEDULE.exists():
        logger.error("Files not found")
        exit(1)

    exit(create_animation())
