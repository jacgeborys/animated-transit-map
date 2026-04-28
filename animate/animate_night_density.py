"""
Warsaw Night Transit Animation (20:00 - 06:00)
===============================================
Identical pipeline to animate_full_density.py with three key differences:
  - Time window: 20:00 → 06:00 (10 hours, crosses midnight)
  - No dynamic zoom — static map extent throughout
  - Departure-seconds parsing handles GTFS 24+ hour times (night buses)
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from datetime import datetime, timedelta
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import logging

from config import ANALYSIS_DATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).parent.parent
ROUTE_LINES     = PROJECT_ROOT / "data" / "processed" / "route_lines_continuous.shp"
SCHEDULE        = PROJECT_ROOT / "data" / "processed" / "schedule_for_animation.csv"

_gtfs_dirs = sorted([
    d for d in (PROJECT_ROOT / "data" / "raw").glob("warsaw_gtfs_*")
    if "_with_" not in d.name
])
GTFS_STOP_TIMES = _gtfs_dirs[-1] / "stop_times.txt" if _gtfs_dirs else None

FPS           = 30
DURATION      = 180   # seconds of video
START_HOUR    = 20    # 20:00
END_HOUR      = 6     # 06:00 next day
HOURS         = (24 - START_HOUR) + END_HOUR  # 10

FFMPEG_PATH   = r"C:\ffmpeg\bin\ffmpeg.exe"
VEHICLE_FILTER = None   # 'Tram', 'Bus', 'Train', or None for all

if VEHICLE_FILTER:
    _vl = VEHICLE_FILTER.lower()
    JUNCTION_SEGMENTS = PROJECT_ROOT / "data" / "processed" / f"{_vl}_junction_segments.shp"
    OUTPUT            = PROJECT_ROOT / "animate" / "_output" / f"warsaw_{_vl}_night.mp4"
else:
    JUNCTION_SEGMENTS = PROJECT_ROOT / "data" / "processed" / "junction_segments.shp"
    OUTPUT            = PROJECT_ROOT / "animate" / "_output" / "warsaw_transit_night.mp4"

# Static map extent — no zoom
CENTRAL_WARSAW_X = 638000
CENTRAL_WARSAW_Y = 487000
FRAME_SIZE       = 22000

# Vehicle physics
VEHICLE_SPEEDS = {'Tram': 4.5, 'Bus': 5.5, 'Train': 15.0, 'Metro': 10.0}
DEFAULT_SPEED  = 4.7

# Visual
COLORS          = {'Tram': '#FF7075', 'Bus': '#B46EFC', 'Train': '#6BC9C6', 'Metro': '#13c2fc'}
VEHICLE_SIZES   = {'Tram': 16, 'Bus': 14, 'Train': 24, 'Metro': 26}
LINE_WIDTHS     = {k: v * 0.9 for k, v in {'Tram': 1.5, 'Bus': 1.2, 'Train': 2.2, 'Metro': 2.8}.items()}
OUTLINE_COLORS  = {'Tram': '#730011', 'Bus': '#430073', 'Train': '#007345', 'Metro': '#004b80'}
VEHICLE_MARKERS = {'Tram': 'D', 'Bus': 'o', 'Train': '^', 'Metro': 's'}
LINE_COLORS     = {'Tram': '#9e0f14', 'Bus': '#72009e', 'Train': '#2a8a7a', 'Metro': '#006096'}

Z_ORDERS = {
    'Metro': {'glow': 10, 'line': 11, 'streak': 12, 'vehicle': 50},
    'Train': {'glow': 20, 'line': 21, 'streak': 22, 'vehicle': 51},
    'Bus':   {'glow': 30, 'line': 31, 'streak': 32, 'vehicle': 52},
    'Tram':  {'glow': 40, 'line': 41, 'streak': 42, 'vehicle': 53},
}

OSM_DIR = Path(r"D:\QGIS\mapy_warszawy_misc\data\osm")
BACKGROUND_LAYERS = {
    'forests':            OSM_DIR / 'forests.gpkg',
    'parks':              OSM_DIR / 'parks.gpkg',
    'meadow':             OSM_DIR / 'meadow.gpkg',
    'leisure':            OSM_DIR / 'leisure.gpkg',
    'leisure_relations':  OSM_DIR / 'leisure_relations.gpkg',
    'grass':              OSM_DIR / 'grass.gpkg',
    'cemeteries':         OSM_DIR / 'cemeteries.gpkg',
    'allotments':         OSM_DIR / 'allotments.gpkg',
    'water':              OSM_DIR / 'water.gpkg',
    'roads':              OSM_DIR / 'roads.shp',
}
LAYER_STYLES = {
    'forests':           {'fc': '#0a1a0a', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'parks':             {'fc': '#0a1a0a', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'meadow':            {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'leisure':           {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'leisure_relations': {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'grass':             {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'cemeteries':        {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'allotments':        {'fc': '#091709', 'ec': 'none',    'lw': 0,   'zorder': 1},
    'water':             {'fc': '#0f2e45', 'ec': 'none',    'lw': 0,   'zorder': 2},
    'roads':             {'fc': 'none',    'ec': '#4a4a4a', 'lw': 0.3, 'zorder': 3},
}
ROAD_HIGHWAY_FIELD = 'fclass'
ROAD_TIERS = [
    ({'motorway', 'motorway_link', 'expressway'},        2.5),
    ({'trunk', 'trunk_link', 'primary', 'primary_link'}, 2.0),
    ({'secondary', 'secondary_link'},                    1.5),
]

STREAK_LENGTH = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_gtfs_secs(t: str) -> float | None:
    """Parse GTFS time string to seconds, correctly handling hours >= 24."""
    try:
        h, m, s = str(t).split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None


def smooth_stop_times(times, progresses, max_dist_m, max_speed_mps=22.0):
    times = times.copy()
    n = len(times)
    for i in range(1, n - 1):
        seg_dist = (progresses[i] - progresses[i - 1]) * max_dist_m
        seg_time = times[i] - times[i - 1]
        if seg_time > 0 and seg_dist / seg_time > max_speed_mps:
            frac = (progresses[i] - progresses[i - 1]) / (progresses[i + 1] - progresses[i - 1])
            times[i] = times[i - 1] + frac * (times[i + 1] - times[i - 1])
    return times


_base_date = datetime.strptime(ANALYSIS_DATE, "%Y%m%d")


def fmt_clock(seconds: float) -> str:
    """Convert absolute seconds to HH:MM string, wrapping past midnight."""
    h = int(seconds // 3600) % 24
    m = int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}"


def fmt_title(seconds: float) -> str:
    """Return 'DD.MM.YYYY  HH:MM', advancing the date past midnight."""
    date = _base_date + timedelta(days=1) if seconds >= 86400 else _base_date
    return f"{date.strftime('%d.%m.%Y')}  {fmt_clock(seconds)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_animation():
    start_seconds    = START_HOUR * 3600
    duration_seconds = HOURS * 3600
    night_span       = duration_seconds           # alias for bar math

    # --- Load segments ---
    if VEHICLE_FILTER:
        if not JUNCTION_SEGMENTS.exists():
            logger.error(f"Junction segments not found: {JUNCTION_SEGMENTS}")
            return 1
        segments = gpd.read_file(JUNCTION_SEGMENTS)
        logger.info(f"Loaded {len(segments)} junction segments")
    else:
        all_segments = []
        for vt in ['tram', 'bus', 'train', 'metro']:
            vf = PROJECT_ROOT / "data" / "processed" / f"{vt}_junction_segments.shp"
            if vf.exists():
                vs = gpd.read_file(vf)
                vs['vehicle_ty'] = vt.capitalize()
                all_segments.append(vs)
                logger.info(f"  Loaded {len(vs)} {vt} segments")
            else:
                logger.warning(f"  {vt}_junction_segments.shp not found, skipping")
        if not all_segments:
            logger.error("No junction segment files found!")
            return 1
        segments = gpd.GeoDataFrame(pd.concat(all_segments, ignore_index=True))
        logger.info(f"Combined: {len(segments)} segments")

    segments['route_ids_list'] = segments['route_ids'].apply(
        lambda x: x.split(',') if isinstance(x, str) else []
    )

    # --- Load routes ---
    routes = gpd.read_file(ROUTE_LINES)
    routes['route_id'] = routes['route_id'].astype(str)
    if VEHICLE_FILTER:
        routes = routes[routes['vehicle'] == VEHICLE_FILTER].copy()
    logger.info(f"Loaded {len(routes)} route shapes")

    # --- Load schedule ---
    schedule = pd.read_csv(SCHEDULE)
    schedule['route_id'] = schedule['route_id'].astype(str)
    if VEHICLE_FILTER:
        schedule = schedule[schedule['vehicle'] == VEHICLE_FILTER].copy()

    # Parse departure seconds — handles GTFS 24+ hour times (night buses, cross-midnight)
    schedule['departure_seconds'] = schedule['departure_time_str'].apply(parse_gtfs_secs)
    schedule = schedule.dropna(subset=['departure_seconds'])
    schedule['departure_seconds'] = schedule['departure_seconds'].astype(float)

    # Keep only trips that depart within the night window
    schedule = schedule[
        (schedule['departure_seconds'] >= start_seconds) &
        (schedule['departure_seconds'] <  start_seconds + duration_seconds)
    ]

    min_seq = schedule.groupby('trip_id')['stop_sequence'].transform('min')
    schedule['is_first_stop'] = schedule['stop_sequence'] == min_seq

    logger.info(
        f"Schedule: {len(schedule)} entries, {schedule['trip_id'].nunique()} trips "
        f"({fmt_clock(start_seconds)}–{fmt_clock(start_seconds + duration_seconds)})"
    )

    # --- Route lookup ---
    route_lookup = {
        row['shape_id']: {
            'geometry': row.geometry,
            'vehicle':  row['vehicle'],
            'route_id': row['route_id'],
        }
        for _, row in routes.iterrows()
    }

    # --- Stop-by-stop schedules ---
    trip_stop_schedule = {}
    if GTFS_STOP_TIMES and GTFS_STOP_TIMES.exists():
        logger.info(f"Loading stop times from {GTFS_STOP_TIMES.parent.name}...")
        trip_ids_needed = set(schedule['trip_id'].unique())
        st_raw = pd.read_csv(GTFS_STOP_TIMES, low_memory=False)

        def _parse_time(t):
            h, m, s = t.split(':')
            return int(h) * 3600 + int(m) * 60 + int(s)

        freq_templates = {}
        for trip_id, grp in st_raw.groupby('trip_id'):
            grp = grp.sort_values('stop_sequence')
            dist = grp['shape_dist_traveled'].values.astype(np.float64)
            dist = dist - dist[0]
            max_dist = dist.max()
            if max_dist <= 0 or len(grp) < 2 or np.isnan(max_dist):
                continue
            offsets = grp['departure_time'].apply(_parse_time).values.astype(np.float64)
            progresses = dist / max_dist
            offsets = smooth_stop_times(offsets, progresses, max_dist * 1000)
            freq_templates[trip_id] = (offsets, progresses)

        st_filtered = st_raw[st_raw['trip_id'].isin(trip_ids_needed)]
        for trip_id, grp in st_filtered.groupby('trip_id'):
            grp = grp.sort_values('stop_sequence')
            dist = grp['shape_dist_traveled'].values.astype(np.float64)
            dist = dist - dist[0]
            max_dist = dist.max()
            if max_dist <= 0 or len(grp) < 2 or np.isnan(max_dist):
                continue
            times = grp['departure_time'].apply(_parse_time).values.astype(np.float64)
            progresses = dist / max_dist
            times = smooth_stop_times(times, progresses, max_dist * 1000)
            trip_stop_schedule[trip_id] = (times, progresses)

        freq_resolved = 0
        for trip_id in trip_ids_needed:
            if trip_id in trip_stop_schedule:
                continue
            if '__' in trip_id:
                template_id, offset_str = trip_id.rsplit('__', 1)
                if template_id in freq_templates and offset_str.lstrip('-').isdigit():
                    offsets, progresses = freq_templates[template_id]
                    trip_stop_schedule[trip_id] = (offsets + float(offset_str), progresses)
                    freq_resolved += 1

        logger.info(f"  Direct: {len(trip_stop_schedule) - freq_resolved:,}  "
                    f"Freq-based: {freq_resolved:,}  "
                    f"Speed fallback: {len(trip_ids_needed) - len(trip_stop_schedule):,}")
    else:
        logger.warning("stop_times.txt not found — using speed-based interpolation")

    # --- Pre-extract segment geometry ---
    segment_coords = []
    segment_vtypes = []
    for _, segment in segments.iterrows():
        if hasattr(segment.geometry, 'coords'):
            x, y = segment.geometry.xy
            segment_coords.append((x, y))
        else:
            segment_coords.append(None)
        segment_vtypes.append(segment.get('vehicle_ty', 'Tram'))

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='#000000')
    ax.set_facecolor('#000000')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    half = FRAME_SIZE / 2
    ax.set_xlim(CENTRAL_WARSAW_X - half, CENTRAL_WARSAW_X + half)
    ax.set_ylim(CENTRAL_WARSAW_Y - half, CENTRAL_WARSAW_Y + half)

    # --- Background layers ---
    from shapely.geometry import box as shapely_box
    clip_box = shapely_box(
        CENTRAL_WARSAW_X - half, CENTRAL_WARSAW_Y - half,
        CENTRAL_WARSAW_X + half, CENTRAL_WARSAW_Y + half,
    )
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs='EPSG:2180')

    logger.info("Drawing background map layers...")
    for name, path in BACKGROUND_LAYERS.items():
        if not path.exists():
            logger.warning(f"  {name}: not found, skipping")
            continue
        try:
            bg = gpd.read_file(path)
            if bg.crs is None:
                bg = bg.set_crs('EPSG:4326')
            if bg.crs.to_epsg() != 2180:
                bg = bg.to_crs('EPSG:2180')
            if bg.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).any():
                bg['geometry'] = bg.geometry.buffer(0)
            bg = bg[bg.geometry.is_valid & ~bg.geometry.is_empty]
            bg = gpd.clip(bg, clip_gdf)
            if bg.empty:
                continue
            style = LAYER_STYLES[name]
            if name == 'roads' and ROAD_HIGHWAY_FIELD in bg.columns:
                classified = set()
                for classes, lw in ROAD_TIERS:
                    tier = bg[bg[ROAD_HIGHWAY_FIELD].isin(classes)]
                    if not tier.empty:
                        tier.plot(ax=ax, facecolor='none', edgecolor=style['ec'],
                                  linewidth=lw, zorder=style['zorder'])
                    classified |= classes
                rest = bg[~bg[ROAD_HIGHWAY_FIELD].isin(classified)]
                if not rest.empty:
                    rest.plot(ax=ax, facecolor='none', edgecolor=style['ec'],
                              linewidth=style['lw'], zorder=style['zorder'])
            else:
                bg.plot(ax=ax, facecolor=style['fc'], edgecolor=style['ec'],
                        linewidth=style['lw'], zorder=style['zorder'])
            logger.info(f"  {name}: {len(bg)} features")
        except Exception as e:
            logger.warning(f"  {name}: failed ({e}), skipping")

    # --- Ghost train lines ---
    from matplotlib.collections import LineCollection
    train_routes = routes[routes['vehicle'] == 'Train']
    ghost_lines = [
        np.column_stack(row.geometry.xy)
        for _, row in train_routes.iterrows()
        if hasattr(row.geometry, 'xy')
    ]
    if ghost_lines:
        ax.add_collection(LineCollection(
            ghost_lines, colors='#163d38', linewidths=LINE_WIDTHS['Train'],
            alpha=1.0, zorder=Z_ORDERS['Train']['glow'] - 1, capstyle='round', joinstyle='round',
        ))
        logger.info(f"  {len(ghost_lines)} ghost train lines")

    # --- Static base network ---
    segments_by_type = {'Tram': [], 'Bus': [], 'Train': [], 'Metro': []}
    for idx in range(len(segments)):
        if segment_coords[idx] is None:
            continue
        x, y = segment_coords[idx]
        segments_by_type[segment_vtypes[idx]].append(np.column_stack([x, y]))

    for vtype in ['Train', 'Bus', 'Tram', 'Metro']:
        if not segments_by_type[vtype]:
            continue
        ax.add_collection(LineCollection(
            segments_by_type[vtype], colors=LINE_COLORS[vtype],
            linewidths=LINE_WIDTHS[vtype], alpha=0.75,
            zorder=Z_ORDERS[vtype]['line'], capstyle='round', joinstyle='round',
        ))
        logger.info(f"  {len(segments_by_type[vtype])} {vtype} base segments")

    # --- Dynamic artists ---
    streak_lc  = {}
    vehicle_sc = {}
    for vtype in ['Train', 'Bus', 'Tram', 'Metro']:
        z = Z_ORDERS[vtype]
        streak_lc[vtype] = LineCollection(
            [], colors=COLORS[vtype], linewidths=3, alpha=0.25,
            capstyle='round', zorder=z['streak'],
        )
        ax.add_collection(streak_lc[vtype])
        vehicle_sc[vtype] = ax.scatter(
            np.empty(0), np.empty(0), s=VEHICLE_SIZES[vtype],
            color=COLORS[vtype], alpha=0.9, marker=VEHICLE_MARKERS[vtype],
            edgecolors=OUTLINE_COLORS[vtype], linewidths=1.0, zorder=z['vehicle'],
        )

    FONT = 'Segoe UI'
    title_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=24, color='white',
                         verticalalignment='top', fontweight='bold', fontfamily=FONT, zorder=100)
    ax.text(0.98, 0.97, "© 2026 Jacek Gęborys", transform=ax.transAxes, fontsize=12,
            color='white', verticalalignment='top', horizontalalignment='right',
            alpha=0.55, fontfamily=FONT, zorder=100)
    ax.text(0.98, 0.955, "Data: ZTM Warsaw via mkuran.pl  ·  BDOT10k", transform=ax.transAxes,
            fontsize=10, color='white', verticalalignment='top', horizontalalignment='right',
            alpha=0.45, fontfamily=FONT, zorder=100)

    for i, (label, marker, color) in enumerate([
        ('Train', '▲', COLORS['Train']),
        ('Metro', '■', COLORS['Metro']),
        ('Tram',  '♦', COLORS['Tram']),
        ('Bus',   '●', COLORS['Bus']),
    ]):
        ax.text(0.98, 0.22 - i * 0.04, f"{marker}  {label}", transform=ax.transAxes,
                fontsize=11, color=color, verticalalignment='center',
                horizontalalignment='right', alpha=0.85, fontfamily=FONT, zorder=100)

    # --- Night progress bar (20:00 → 06:00) ---
    from matplotlib.patches import Rectangle
    BAR_Y  = 0.038
    BAR_H  = 0.012
    BAR_X0 = 0.05
    BAR_X1 = 0.95
    BAR_W  = BAR_X1 - BAR_X0

    ax.add_patch(Rectangle((BAR_X0, BAR_Y - BAR_H / 2), BAR_W, BAR_H,
                            transform=ax.transAxes, color='#222222', alpha=0.75,
                            zorder=98, clip_on=False))

    bar_fill = Rectangle((BAR_X0, BAR_Y - BAR_H / 2), 0, BAR_H,
                          transform=ax.transAxes, color='#888888', alpha=0.85,
                          zorder=99, clip_on=False)
    ax.add_patch(bar_fill)

    # Hour ticks: 20:00, 22:00, 00:00, 02:00, 04:00, 06:00
    for step in range(0, HOURS + 1, 2):
        frac = step / HOURS
        x    = BAR_X0 + frac * BAR_W
        label = f"{(START_HOUR + step) % 24:02d}:00"
        ax.plot([x, x], [BAR_Y - BAR_H / 2, BAR_Y + BAR_H / 2 + 0.006],
                transform=ax.transAxes, color='white', lw=0.6, alpha=0.5,
                zorder=100, clip_on=False)
        ax.text(x, BAR_Y + BAR_H / 2 + 0.008, label,
                transform=ax.transAxes, fontsize=12, color='white', alpha=0.5,
                ha='center', va='bottom', fontfamily=FONT, zorder=100, clip_on=False)

    (bar_marker,) = ax.plot([], [], 'o', color='white', ms=5, alpha=0.95,
                            transform=ax.transAxes, zorder=101, clip_on=False)

    # --- Animation loop ---
    vehicles      = []
    total_frames  = FPS * DURATION

    def update_frame(frame_num):
        nonlocal vehicles

        current_seconds = start_seconds + (frame_num / total_frames) * duration_seconds
        frame_duration  = duration_seconds / total_frames

        # Spawn new vehicles
        new_trips = schedule[
            (schedule['departure_seconds'] >= current_seconds - frame_duration / 2) &
            (schedule['departure_seconds'] <= current_seconds + frame_duration / 2) &
            schedule['is_first_stop']
        ]
        for _, trip in new_trips.iterrows():
            shape_id = trip['shape_id']
            if shape_id not in route_lookup:
                continue
            route_info = route_lookup[shape_id]
            trip_id    = trip['trip_id']
            if trip_id in trip_stop_schedule:
                sched   = trip_stop_schedule[trip_id]
                end_abs = float(sched[0][-1])
            else:
                speed   = VEHICLE_SPEEDS.get(route_info['vehicle'], DEFAULT_SPEED)
                end_abs = current_seconds + route_info['geometry'].length / speed
                sched   = None
            vehicles.append({
                'shape_id':   shape_id,
                'vehicle':    route_info['vehicle'],
                'route':      route_info['geometry'],
                'start_time': current_seconds,
                'end_abs':    end_abs,
                'sched':      sched,
            })

        # Update positions
        active_vehicles  = []
        vehicles_by_type = {'Tram': [], 'Bus': [], 'Train': [], 'Metro': []}
        streaks_by_type  = {'Tram': [], 'Bus': [], 'Train': [], 'Metro': []}

        for vehicle in vehicles:
            if current_seconds > vehicle['end_abs']:
                continue
            if vehicle['sched'] is not None:
                progress = float(np.interp(
                    current_seconds, vehicle['sched'][0], vehicle['sched'][1]
                ))
            else:
                elapsed  = current_seconds - vehicle['start_time']
                dur      = vehicle['end_abs'] - vehicle['start_time']
                progress = elapsed / dur if dur > 0 else 0

            position = vehicle['route'].interpolate(progress, normalized=True)
            active_vehicles.append(vehicle)

            vtype        = vehicle['vehicle']
            route        = vehicle['route']
            route_length = route.length
            vehicles_by_type[vtype].append(position)

            streak_pts = [(position.x, position.y)]
            for i in range(1, 4):
                bp = progress - (i * STREAK_LENGTH / 3) / route_length
                if bp >= 0:
                    p = route.interpolate(bp, normalized=True)
                    streak_pts.append((p.x, p.y))
            if len(streak_pts) > 1:
                streaks_by_type[vtype].append(np.array(streak_pts))

        vehicles = active_vehicles

        for vtype in ['Train', 'Bus', 'Tram', 'Metro']:
            streak_lc[vtype].set_segments(
                streaks_by_type[vtype] if streaks_by_type[vtype] else []
            )
            positions = vehicles_by_type[vtype]
            vehicle_sc[vtype].set_offsets(
                np.array([[p.x, p.y] for p in positions]) if positions else np.empty((0, 2))
            )

        # Progress bar
        bar_frac = (current_seconds - start_seconds) / night_span
        cur_x    = BAR_X0 + bar_frac * BAR_W
        bar_fill.set_width(cur_x - BAR_X0)
        bar_marker.set_data([cur_x], [BAR_Y])

        # Title — wrap clock past midnight
        title_text.set_text(f"Warsaw Night Transit · {fmt_title(current_seconds)}")

        if frame_num % 60 == 0:
            logger.info(
                f"Frame {frame_num}/{total_frames} "
                f"({frame_num / total_frames * 100:.0f}%) "
                f"— {len(vehicles)} active vehicles"
            )

    # --- Render ---
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

    if not Path(FFMPEG_PATH).exists():
        logger.error(f"FFmpeg not found at {FFMPEG_PATH}")
        plt.close()
        return 1

    writer = animation.FFMpegWriter(
        fps=FPS, codec='libx264', bitrate=5000,
        metadata={'artist': 'Jacek Gęborys'},
        extra_args=['-pix_fmt', 'yuv420p',
                    '-g', str(FPS),
                    '-movflags', '+frag_keyframe+empty_moov+default_base_moof'],
    )

    logger.info(f"Rendering {total_frames} frames → {OUTPUT}")
    try:
        with writer.saving(fig, OUTPUT, dpi=100):
            for frame_num in range(total_frames):
                update_frame(frame_num)
                writer.grab_frame()
        logger.info("Done!")
    except KeyboardInterrupt:
        logger.info("Interrupted — partial video saved.")
        plt.close()
        return 0
    except Exception as e:
        logger.error(f"Render error: {e}")
        plt.close()
        return 1

    plt.close()
    return 0


if __name__ == "__main__":
    if not ROUTE_LINES.exists() or not SCHEDULE.exists():
        logger.error("Route lines or schedule not found — run route_line_builder.py first")
        exit(1)
    exit(create_animation())
