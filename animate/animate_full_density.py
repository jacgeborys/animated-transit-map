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
FPS = 30
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
ZOOM_OUT_END_H   = 10     # hour when zoom settles back to base


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
# VEHICLE_SIZES = 0.9 * pd.Series(VEHICLE_SIZES)  # Scale down for better proportions
LINE_WIDTHS = {'Tram': 1.5, 'Bus': 1.2, 'Train': 1.5, 'Metro': 2.2}
GLOW_WIDTH = 4.00
GLOW_ALPHA = 0.7
BASE_BRIGHTNESS = 0.2
MAX_BRIGHTNESS = 1.0
OUTLINE_COLORS = {'Tram': '#1a0003', 'Bus': '#0f0018', 'Train': '#001410', 'Metro': '#000e1a'}
VEHICLE_MARKERS = {'Tram': 'D', 'Bus': 'o', 'Train': '^', 'Metro': 's'}

# Color gradients for density visualization
# 'dark' matches static base network color; 'bright' is near-white/pastel for luminous glow at max density
COLOR_GRADIENTS = {
    'Tram':  {'dark': '#3d0005', 'bright': '#ffcccc'},
    'Bus':   {'dark': '#2d0040', 'bright': '#e8b3ff'},
    'Train': {'dark': '#0f3d38', 'bright': '#c2f0eb'},
    'Metro': {'dark': '#001e30', 'bright': '#c4eaff'},
}

# Z-order layering (background → trains → buses → trams → metro on top)
Z_ORDERS = {
    'Metro': {'glow': 10, 'line': 20},
    'Train': {'glow': 30, 'line': 40},
    'Bus':   {'glow': 50, 'line': 60},
    'Tram':  {'glow': 70, 'line': 80},

}

# Background map layers (OSM data)
OSM_DIR = Path(r"D:\QGIS\mapy_warszawy_misc\data\osm")
BACKGROUND_LAYERS = {
    'forests':   OSM_DIR / 'forests.gpkg',
    'parks':     OSM_DIR / 'parks.gpkg',
    'meadow':     OSM_DIR / 'meadow.gpkg',
    'leisure':     OSM_DIR / 'leisure.gpkg',
    'leisure_relations':     OSM_DIR / 'leisure_relations.gpkg',
    'grass':     OSM_DIR / 'grass.gpkg',
    'cemeteries': OSM_DIR / 'cemeteries.gpkg',
    'allotments': OSM_DIR / 'allotments.gpkg',
    'water':     OSM_DIR / 'water.gpkg',
    'roads':     OSM_DIR / 'roads.shp',
}
LAYER_STYLES = {
    'forests':   {'fc': '#152b15', 'ec': 'none',    'lw': 0,   'zorder': 5},
    'parks':     {'fc': '#152b15', 'ec': 'none',    'lw': 0,   'zorder': 5},
    'meadow':     {'fc': '#152b15', 'ec': 'none',    'lw': 0,   'zorder': 5},
    'leisure':     {'fc': '#152b15', 'ec': 'none',    'lw': 0,   'zorder': 5},
    'leisure_relations':     {'fc': '#152b15', 'ec': 'none',    'lw': 0,   'zorder': 5},
    'grass':     {'fc': '#152b15', 'ec': 'none',    'lw': 0,   'zorder': 5},
    'cemeteries': {'fc': '#152b15', 'ec': 'none',    'lw': 0,   'zorder': 5},
    'allotments': {'fc': '#152b15', 'ec': 'none',    'lw': 0,   'zorder': 5},
    'water':     {'fc': '#0f2e45', 'ec': 'none',    'lw': 0,   'zorder': 2},
    'roads':     {'fc': 'none',    'ec': '#383838', 'lw': 0.5, 'zorder': 1},  # default (lowest tier)
}

# Road width tiers by highway class — field name is 'fclass' in Geofabrik OSM exports
ROAD_HIGHWAY_FIELD = 'fclass'
ROAD_TIERS = [
    ({'motorway', 'motorway_link', 'expressway'},        1.8),
    ({'trunk', 'trunk_link', 'primary', 'primary_link'}, 1.0),
    ({'secondary', 'secondary_link'},                    0.7),
]

STREAK_LENGTH = 180

# Density calculation settings
ROLLING_WINDOW_MINUTES = 10
SEGMENT_TRACKING_STEP = 10  # real seconds between segment position checks (independent of FPS/DURATION)
# Direct brightness calibration — set to the vehicles/km value that should hit max brightness.
# Segments above this value are clamped to full brightness.
# Tune based on "Observed max density" printed at end of each run.
# Rule of thumb: set to ~70-80% of observed max so peak segments glow fully.
DENSITY_FOR_MAX_BRIGHTNESS = 1000.0  # vehicles/km


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

    # Pre-convert to Nx2 arrays for LineCollection (avoids per-frame conversion)
    segment_arrays = []
    for idx in range(len(segment_coords)):
        coords = segment_coords[idx]
        if coords is not None:
            x, y = coords
            segment_arrays.append(np.column_stack([x, y]))
        else:
            segment_arrays.append(None)

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

    # Pre-create all dynamic artists — updated each frame, never created/destroyed
    seg_glow_lc = {}
    seg_main_lc = {}
    streak_lc = {}
    vehicle_sc = {}
    for vtype in ['Train', 'Bus', 'Tram', 'Metro']:
        z = Z_ORDERS[vtype]
        glow = LineCollection([], linewidths=GLOW_WIDTH, capstyle='round', joinstyle='round', zorder=z['glow'])
        main = LineCollection([], linewidths=LINE_WIDTHS.get(vtype, 1.5), capstyle='round', joinstyle='round', zorder=z['line'] + 0.5)
        streak = LineCollection([], colors=COLORS[vtype], linewidths=3, alpha=0.2, capstyle='round', zorder=70)
        ax.add_collection(glow)
        ax.add_collection(main)
        ax.add_collection(streak)
        seg_glow_lc[vtype] = glow
        seg_main_lc[vtype] = main
        streak_lc[vtype] = streak
        sc = ax.scatter(np.empty(0), np.empty(0), s=VEHICLE_SIZES[vtype],
                        color=COLORS[vtype], alpha=0.8, marker=VEHICLE_MARKERS[vtype],
                        edgecolors=OUTLINE_COLORS[vtype], linewidths=0.6, zorder=80)
        vehicle_sc[vtype] = sc

    title_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=24, color='white',
           verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), zorder=100)
    count_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=16, color='white',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), zorder=100)
    ax.text(0.98, 0.02, "© Jacek Gęborys", transform=ax.transAxes, fontsize=12, color='white',
           verticalalignment='bottom', horizontalalignment='right', alpha=0.6, zorder=100)

    # Static legend — bottom right, no frame
    legend_items = [
        ('Train', '▲', COLORS['Train']),
        ('Metro', '■', COLORS['Metro']),
        ('Tram',  '◆', COLORS['Tram']),
        ('Bus',   '●', COLORS['Bus']),
    ]
    for i, (label, marker, color) in enumerate(legend_items):
        y = 0.22 - i * 0.04
        ax.text(0.98, y, f"{marker}  {label}", transform=ax.transAxes,
                fontsize=11, color=color, verticalalignment='center',
                horizontalalignment='right', alpha=0.85, zorder=100)

    # Animation state
    segment_vehicle_visits = [[] for _ in range(len(segments))]
    vehicles = []
    total_frames = FPS * DURATION
    rolling_window_seconds = ROLLING_WINDOW_MINUTES * 60
    observed_max_density = 0.0

    def update_frame(frame_num):
        nonlocal vehicles

        # Calculate current time
        current_seconds = start_seconds + (frame_num / total_frames) * duration_seconds
        current_time = f"{int(current_seconds // 3600):02d}:{int((current_seconds % 3600) // 60):02d}"
        frame_duration = duration_seconds / total_frames

        # Dynamic zoom
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
                    'id': trip_id, 'shape_id': shape_id,
                    'vehicle': route_info['vehicle'], 'route': route_info['geometry'],
                    'route_id': route_info['route_id'], 'start_time': current_seconds,
                    'end_abs': end_abs, 'sched': sched, 'current_segment': None
                })

        # Update vehicles — sub-stepped segment tracking + single draw position
        active_vehicles = []
        vehicles_by_type = {'Tram': [], 'Bus': [], 'Train': [], 'Metro': []}
        streaks_by_type  = {'Tram': [], 'Bus': [], 'Train': [], 'Metro': []}

        for vehicle in vehicles:
            expired = current_seconds > vehicle['end_abs']

            # --- Segment tracking: sub-stepped at SEGMENT_TRACKING_STEP real seconds ---
            track_from = vehicle.get('last_tracked', vehicle['start_time'])
            track_to   = vehicle['end_abs'] if expired else current_seconds

            if track_from < track_to:
                t = track_from + SEGMENT_TRACKING_STEP
                track_times = []
                while t < track_to:
                    track_times.append(t)
                    t += SEGMENT_TRACKING_STEP
                track_times.append(track_to)

                for t in track_times:
                    if vehicle['sched'] is not None:
                        prog_t = float(np.interp(t, vehicle['sched'][0], vehicle['sched'][1]))
                    else:
                        elapsed = t - vehicle['start_time']
                        dur = vehicle['end_abs'] - vehicle['start_time']
                        prog_t = elapsed / dur if dur > 0 else 0

                    pos_t = vehicle['route'].interpolate(prog_t, normalized=True)
                    old_seg = vehicle['current_segment']
                    new_seg = None
                    shape_id = vehicle['shape_id']
                    if shape_id in route_to_segments:
                        nearby = segment_spatial_index.query(pos_t)
                        for seg_idx in route_to_segments[shape_id]:
                            if seg_idx in nearby:
                                new_seg = seg_idx
                                break
                    if new_seg != old_seg:
                        if old_seg is not None:
                            for visit in reversed(segment_vehicle_visits[old_seg]):
                                if visit[0] == vehicle['id'] and visit[2] is None:
                                    visit[2] = t
                                    break
                        if new_seg is not None:
                            segment_vehicle_visits[new_seg].append([vehicle['id'], t, None])
                            vehicle['current_segment'] = new_seg

                vehicle['last_tracked'] = track_to

            if expired:
                continue

            # --- Drawing: position at current_seconds only ---
            if vehicle['sched'] is not None:
                progress = float(np.interp(current_seconds, vehicle['sched'][0], vehicle['sched'][1]))
            else:
                elapsed = current_seconds - vehicle['start_time']
                dur = vehicle['end_abs'] - vehicle['start_time']
                progress = elapsed / dur if dur > 0 else 0

            position = vehicle['route'].interpolate(progress, normalized=True)
            vehicle['position'] = position
            vehicle['progress'] = progress
            active_vehicles.append(vehicle)

            vtype = vehicle['vehicle']
            vehicles_by_type[vtype].append(position)
            route = vehicle['route']
            route_length = route.length
            streak_pts = [(position.x, position.y)]
            for i in range(1, 4):
                bp = progress - (i * STREAK_LENGTH / 3) / route_length
                if bp >= 0:
                    p = route.interpolate(bp, normalized=True)
                    streak_pts.append((p.x, p.y))
            if len(streak_pts) > 1:
                streaks_by_type[vtype].append(np.array(streak_pts))

        vehicles = active_vehicles

        # Brightness calculation
        segment_brightness = np.full(len(segments), BASE_BRIGHTNESS, dtype=np.float32)
        cutoff_time = current_seconds - rolling_window_seconds
        active_segments = set()
        for idx, visits in enumerate(segment_vehicle_visits):
            if visits:
                segment_vehicle_visits[idx] = [v for v in visits if v[2] is None or v[2] >= cutoff_time]
                if segment_vehicle_visits[idx]:
                    active_segments.add(idx)

        for idx in active_segments:
            total_vs = 0
            for _, entry_time, exit_time in segment_vehicle_visits[idx]:
                a = max(entry_time, cutoff_time)
                b = min(exit_time if exit_time else current_seconds, current_seconds)
                if b > a:
                    total_vs += b - a
            occupancy = total_vs / rolling_window_seconds
            length_km = segment_lengths[idx]
            density = occupancy / length_km if length_km > 0 else 0

            nonlocal observed_max_density
            if density > observed_max_density:
                observed_max_density = density

            ratio = density / density_for_max_brightness if density_for_max_brightness > 0 else 0
            segment_brightness[idx] = min(BASE_BRIGHTNESS + (MAX_BRIGHTNESS - BASE_BRIGHTNESS) * ratio, MAX_BRIGHTNESS)

        # Build per-type segment data for LineCollections
        bright_data = {vt: {'segs': [], 'main_c': [], 'glow_c': []} for vt in ['Tram', 'Bus', 'Train', 'Metro']}
        for idx in active_segments:
            brightness = segment_brightness[idx]
            if brightness <= BASE_BRIGHTNESS + 0.001 or segment_arrays[idx] is None:
                continue
            vtype = segment_vtypes[idx]
            nb = min(max(brightness, 0.0), 1.0)
            gradient = COLOR_GRADIENTS.get(vtype, COLOR_GRADIENTS['Tram'])
            # Normalize so BASE_BRIGHTNESS → dark_color (matching static base), MAX → bright_color
            gp = (nb - BASE_BRIGHTNESS) / (MAX_BRIGHTNESS - BASE_BRIGHTNESS)
            gp = max(0.0, gp) ** 0.5  # sqrt curve: line brightens faster (25% density → 50% color)
            main_color = interpolate_color(gradient['dark'], gradient['bright'], gp)
            mr, mg, mb = int(main_color[1:3],16)/255, int(main_color[3:5],16)/255, int(main_color[5:7],16)/255
            bh = gradient['bright']
            r, g, b = int(bh[1:3],16)/255, int(bh[3:5],16)/255, int(bh[5:7],16)/255
            bright_data[vtype]['segs'].append(segment_arrays[idx])
            bright_data[vtype]['main_c'].append([mr, mg, mb, 1.0])
            bright_data[vtype]['glow_c'].append([r, g, b, max(0.0, gp * GLOW_ALPHA)])

        # Update segment collections
        for vtype in ['Train', 'Bus', 'Tram', 'Metro']:
            d = bright_data[vtype]
            if d['segs']:
                seg_glow_lc[vtype].set_segments(d['segs'])
                seg_glow_lc[vtype].set_color(d['glow_c'])
                seg_main_lc[vtype].set_segments(d['segs'])
                seg_main_lc[vtype].set_color(d['main_c'])
            else:
                seg_glow_lc[vtype].set_segments([])
                seg_main_lc[vtype].set_segments([])

        # Update streaks and vehicle dots
        for vtype in ['Train', 'Bus', 'Tram', 'Metro']:
            streak_lc[vtype].set_segments(streaks_by_type[vtype] if streaks_by_type[vtype] else [])
            positions = vehicles_by_type[vtype]
            vehicle_sc[vtype].set_offsets(
                np.array([[p.x, p.y] for p in positions]) if positions else np.empty((0, 2))
            )

        # Update text
        title_text.set_text(f"Warsaw Public Transit - 27.04.2026 {current_time}")
        # count_text.set_text(f"Active vehicles: {len(vehicles)}")

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
