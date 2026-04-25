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
HOURS = 20  # hours of transit to show
# HOURS = 21  # hours of transit to show
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
ZOOM_IN_END_H    = 6.5      # hour when tightest zoom is reached
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
COLORS = {'Tram': '#FF7075', 'Bus': '#B46EFC', 'Train': '#6BC9C6', 'Metro': '#13c2fc'}
VEHICLE_SIZES = {'Tram': 16, 'Bus': 14, 'Train': 20, 'Metro': 26}
# VEHICLE_SIZES = 0.9 * pd.Series(VEHICLE_SIZES)  # Scale down for better proportions
LINE_WIDTHS = {'Tram': 1.5, 'Bus': 1.2, 'Train': 2.0, 'Metro': 2.8}
LINE_WIDTHS = {k: v * 0.9 for k, v in LINE_WIDTHS.items()}  # Scale down for better proportions
OUTLINE_COLORS = {'Tram': '#730011', 'Bus': '#430073', 'Train': '#007045', 'Metro': '#004b80'}
VEHICLE_MARKERS = {'Tram': 'D', 'Bus': 'o', 'Train': '^', 'Metro': 's'}

# Static base colors for transit lines
LINE_COLORS = {
    'Tram':  '#9e0f14',
    'Bus':   '#72009e',
    'Train': '#2a8a7a',
    'Metro': '#006096',
}

# Z-order layering: Metro (bottom) → Train → Bus → Tram (top)
# Lines all below vehicles; within lines/vehicles, same Metro→Tram order
Z_ORDERS = {
    'Metro': {'glow': 10, 'line': 11, 'streak': 12, 'vehicle': 50},
    'Train': {'glow': 20, 'line': 21, 'streak': 22, 'vehicle': 51},
    'Bus':   {'glow': 30, 'line': 31, 'streak': 32, 'vehicle': 52},
    'Tram':  {'glow': 40, 'line': 41, 'streak': 42, 'vehicle': 53},
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
    'forests':   {'fc': '#0a1a0a', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'parks':     {'fc': '#0a1a0a', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'meadow':     {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'leisure':     {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'leisure_relations':     {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'grass':     {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'cemeteries': {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'allotments': {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'water':     {'fc': '#0f2e45', 'ec': 'none',    'lw': 0,   'zorder': 2},
    'roads':     {'fc': 'none',    'ec': '#4a4a4a', 'lw': 0.3, 'zorder': 3},
}

# Road width tiers by highway class — field name is 'fclass' in Geofabrik OSM exports
ROAD_HIGHWAY_FIELD = 'fclass'
ROAD_TIERS = [
    ({'motorway', 'motorway_link', 'expressway'},        2.5),
    ({'trunk', 'trunk_link', 'primary', 'primary_link'}, 2.0),
    ({'secondary', 'secondary_link'},                    1.5),
]

STREAK_LENGTH = 200


def smooth_stop_times(times, progresses, max_dist_m, max_speed_mps=22.0):
    """Fix implausibly fast inter-stop segments (GTFS data errors) by interpolating times."""
    times = times.copy()
    n = len(times)
    for i in range(1, n - 1):
        seg_dist = (progresses[i] - progresses[i - 1]) * max_dist_m
        seg_time = times[i] - times[i - 1]
        if seg_time > 0 and seg_dist / seg_time > max_speed_mps:
            frac = (progresses[i] - progresses[i - 1]) / (progresses[i + 1] - progresses[i - 1])
            times[i] = times[i - 1] + frac * (times[i + 1] - times[i - 1])
    return times


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

    # Mark the first stop of each trip — works regardless of whether GTFS uses 0- or 1-indexed sequences
    min_seq = schedule.groupby('trip_id')['stop_sequence'].transform('min')
    schedule['is_first_stop'] = schedule['stop_sequence'] == min_seq

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
            dist = grp['shape_dist_traveled'].values.astype(np.float64)
            dist = dist - dist[0]  # shift so first stop is always at position 0
            max_dist = dist.max()
            if max_dist <= 0 or len(grp) < 2:
                continue
            offsets = grp['departure_time'].apply(_parse_time).values.astype(np.float64)
            progresses = dist / max_dist
            offsets = smooth_stop_times(offsets, progresses, max_dist * 1000)
            freq_templates[trip_id] = (offsets, progresses)

        # Match schedule trips: direct match first, then frequency-based (template__offset)
        st_filtered = st_raw[st_raw['trip_id'].isin(trip_ids_needed)]
        for trip_id, grp in st_filtered.groupby('trip_id'):
            grp = grp.sort_values('stop_sequence')
            dist = grp['shape_dist_traveled'].values.astype(np.float64)
            dist = dist - dist[0]  # shift so first stop is always at position 0
            max_dist = dist.max()
            if max_dist <= 0 or len(grp) < 2:
                continue
            times = grp['departure_time'].apply(_parse_time).values.astype(np.float64)
            progresses = dist / max_dist
            times = smooth_stop_times(times, progresses, max_dist * 1000)
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

    # Pre-extract segment data to avoid repeated DataFrame lookups (major speedup!)
    logger.info("Pre-extracting segment data for faster rendering...")
    segment_coords = []
    segment_vtypes = []

    for idx, segment in segments.iterrows():
        if hasattr(segment.geometry, 'coords'):
            x, y = segment.geometry.xy
            segment_coords.append((x, y))
            segment_vtypes.append(segment.get('vehicle_ty', 'Tram'))
        else:
            segment_coords.append(None)
            segment_vtypes.append('Tram')

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

        line_color = LINE_COLORS.get(vtype, '#3d0005')
        line_width = LINE_WIDTHS.get(vtype, 1.5)
        z_order = Z_ORDERS.get(vtype, Z_ORDERS['Tram'])

        lc = LineCollection(segments_by_type[vtype], colors=line_color,
                           linewidths=line_width, alpha=0.75, zorder=z_order['line'],
                           capstyle='round', joinstyle='round')
        ax.add_collection(lc)

        logger.info(f"  Added {len(segments_by_type[vtype])} {vtype} base segments")

    logger.info("Static base network added! Will only draw bright segments each frame.")

    # Pre-create all dynamic artists — updated each frame, never created/destroyed
    streak_lc = {}
    vehicle_sc = {}
    for vtype in ['Train', 'Bus', 'Tram', 'Metro']:
        z = Z_ORDERS[vtype]
        streak = LineCollection([], colors=COLORS[vtype], linewidths=3, alpha=0.25, capstyle='round', zorder=z['streak'])
        ax.add_collection(streak)
        streak_lc[vtype] = streak
        sc = ax.scatter(np.empty(0), np.empty(0), s=VEHICLE_SIZES[vtype],
                        color=COLORS[vtype], alpha=0.9, marker=VEHICLE_MARKERS[vtype],
                        edgecolors=OUTLINE_COLORS[vtype], linewidths=0.8, zorder=z['vehicle'])
        vehicle_sc[vtype] = sc

    title_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=24, color='white',
           verticalalignment='top', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), zorder=100)
    count_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=16, color='white',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), zorder=100)
    ax.text(0.98, 0.02, "© 2026 Jacek Gęborys", transform=ax.transAxes, fontsize=12, color='white',
           verticalalignment='bottom', horizontalalignment='right', alpha=0.6, zorder=100)

    # Static legend — bottom right, no frame
    legend_items = [
        ('SKM Train', '▲', COLORS['Train']),
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
    vehicles = []
    total_frames = FPS * DURATION

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
            schedule['is_first_stop']
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
                    'shape_id': shape_id,
                    'vehicle': route_info['vehicle'], 'route': route_info['geometry'],
                    'start_time': current_seconds,
                    'end_abs': end_abs, 'sched': sched,
                })

        # Update vehicles — sub-stepped segment tracking + single draw position
        active_vehicles = []
        vehicles_by_type = {'Tram': [], 'Bus': [], 'Train': [], 'Metro': []}
        streaks_by_type  = {'Tram': [], 'Bus': [], 'Train': [], 'Metro': []}

        for vehicle in vehicles:
            if current_seconds > vehicle['end_abs']:
                continue

            # --- Drawing: position at current_seconds ---
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
                                               '-g', str(FPS),  # keyframe every 1s → small playable fragments
                                               '-movflags', '+frag_keyframe+empty_moov+default_base_moof'])

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
    return 0


if __name__ == "__main__":
    if not ROUTE_LINES.exists() or not SCHEDULE.exists():
        logger.error("Files not found")
        exit(1)

    exit(create_animation())
