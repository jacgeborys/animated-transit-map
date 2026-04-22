"""
Route-Based Segment Animation - V2
===================================
Creates segments directly from actual route paths (no geometric intersection splitting).
Only splits where the GTFS data shows routes diverge/merge.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, LineString
from shapely.ops import linemerge, unary_union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Create magma colormap (dark to bright yellow-white)
magma_cmap = plt.get_cmap('magma')

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
ROUTE_LINES = PROJECT_ROOT / "data" / "processed" / "route_lines_continuous.shp"
SCHEDULE = PROJECT_ROOT / "data" / "processed" / "schedule_for_animation.csv"
OUTPUT = PROJECT_ROOT / "animate" / "_output" / "warsaw_segments_trasa_lazienkowska.mp4"

# Animation settings
FPS = 20
DURATION = 90
HOURS = 21
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# Map extent
CENTRAL_WARSAW_X = 638000
CENTRAL_WARSAW_Y = 487000
FRAME_SIZE = 25000

# Visual settings
COLORS = {'Tram': '#E63946', 'Bus': '#7209B7', 'Train': '#2A9D8F'}
VEHICLE_SIZES = {'Tram': 16, 'Bus': 12, 'Train': 24}
LINE_WIDTH = 1.5
GLOW_WIDTH = 2.8
GLOW_ALPHA = 0.4
BASE_BRIGHTNESS = 0.02  # Much dimmer starting point (was 0.05)
MAX_BRIGHTNESS = 1.0
OUTLINE_COLORS = {'Tram': '#FF7075', 'Bus': '#B46EFC', 'Train': '#6BC9C6'}
STREAK_LENGTH = 250

# Target routes
TARGET_ROUTES = ['138', '151', '143', '187', 'N25', '523', '188', '411', '182', '502', '525', '514', '385', '507', '141', '167']


def create_segments_from_routes(routes):
    """
    Split routes at convergence/divergence points to create node-to-node segments.
    This allows shared sections to accumulate brightness from multiple routes
    while individual sections only brighten from their own route.
    """
    logger.info("Creating node-to-node segments...")

    from shapely.geometry import Point, LineString, MultiPoint
    from shapely.ops import split, snap, linemerge

    # Step 1: Find all junction points (where routes converge/diverge)
    junction_points = []

    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            geom_i = routes.iloc[i].geometry
            geom_j = routes.iloc[j].geometry

            # Find intersection/overlap
            intersection = geom_i.intersection(geom_j.buffer(15))

            if intersection.is_empty:
                continue

            # Get endpoints of overlapping sections
            if intersection.geom_type == 'LineString':
                coords = list(intersection.coords)
                if len(coords) >= 2:
                    junction_points.append(Point(coords[0]))
                    junction_points.append(Point(coords[-1]))
            elif intersection.geom_type == 'MultiLineString':
                for line in intersection.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        junction_points.append(Point(coords[0]))
                        junction_points.append(Point(coords[-1]))

    # Remove duplicate points (within 20m)
    unique_junctions = []
    for pt in junction_points:
        is_duplicate = False
        for existing in unique_junctions:
            if pt.distance(existing) < 20:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_junctions.append(pt)

    logger.info(f"Found {len(unique_junctions)} junction points")

    # Step 2: Split each route at junction points
    all_segments = []

    for idx, route in routes.iterrows():
        route_geom = route.geometry
        route_id = route['route_id']
        shape_id = route['shape_id']

        # Find which junctions are on this route
        junctions_on_route = []
        for junction in unique_junctions:
            if route_geom.distance(junction) < 20:
                junctions_on_route.append(junction)

        if not junctions_on_route:
            # No junctions, use whole route
            all_segments.append({
                'geometry': route_geom,
                'route_id': route_id,
                'shape_id': shape_id,
                'route_ids': [route_id]
            })
            continue

        # Sort junctions by position along route
        junctions_with_dist = []
        for junction in junctions_on_route:
            dist = route_geom.project(junction)
            junctions_with_dist.append((dist, junction))
        junctions_with_dist.sort(key=lambda x: x[0])

        # Split route into segments between junctions
        current_start = 0

        for i, (junction_dist, junction_pt) in enumerate(junctions_with_dist):
            # Create segment from current_start to this junction
            if junction_dist - current_start > 50:  # Min 50m segment
                try:
                    segment_geom = substring(route_geom, current_start, junction_dist)
                    if segment_geom and hasattr(segment_geom, 'length') and segment_geom.length > 50:
                        all_segments.append({
                            'geometry': segment_geom,
                            'route_id': route_id,
                            'shape_id': shape_id,
                            'route_ids': [route_id]
                        })
                except:
                    pass
            current_start = junction_dist

        # Add final segment
        if route_geom.length - current_start > 50:
            try:
                segment_geom = substring(route_geom, current_start, route_geom.length)
                if segment_geom and hasattr(segment_geom, 'length') and segment_geom.length > 50:
                    all_segments.append({
                        'geometry': segment_geom,
                        'route_id': route_id,
                        'shape_id': shape_id,
                        'route_ids': [route_id]
                    })
            except:
                pass

    logger.info(f"Created {len(all_segments)} segments from {len(routes)} routes")

    # Step 3: Find which segments overlap
    logger.info("Finding overlapping segments...")
    for i in range(len(all_segments)):
        for j in range(i + 1, len(all_segments)):
            if all_segments[i]['route_id'] == all_segments[j]['route_id']:
                continue

            geom_i = all_segments[i]['geometry']
            geom_j = all_segments[j]['geometry']

            # Check for overlap
            overlap = geom_i.buffer(15).intersection(geom_j.buffer(15))
            if hasattr(overlap, 'length') and overlap.length > 100:
                if all_segments[j]['route_id'] not in all_segments[i]['route_ids']:
                    all_segments[i]['route_ids'].append(all_segments[j]['route_id'])
                if all_segments[i]['route_id'] not in all_segments[j]['route_ids']:
                    all_segments[j]['route_ids'].append(all_segments[i]['route_id'])

    # Log statistics
    overlap_counts = {}
    for seg in all_segments:
        num_routes = len(seg['route_ids'])
        overlap_counts[num_routes] = overlap_counts.get(num_routes, 0) + 1

    logger.info("Segment overlap distribution (before merging):")
    for num_routes in sorted(overlap_counts.keys()):
        logger.info(f"  {overlap_counts[num_routes]} segments with {num_routes} route(s)")

    # Merge overlapping segments into single entities
    logger.info("Merging overlapping segments...")
    merged_segments = []
    used_indices = set()

    for i in range(len(all_segments)):
        if i in used_indices:
            continue

        # Find all segments that overlap with this one
        group = [i]
        group_route_ids = set(all_segments[i]['route_ids'])

        for j in range(i + 1, len(all_segments)):
            if j in used_indices:
                continue

            # Check if j overlaps with any in the group
            if any(rid in all_segments[j]['route_ids'] for rid in group_route_ids):
                # Check geometric overlap
                geom_group = all_segments[i]['geometry']
                geom_j = all_segments[j]['geometry']
                overlap = geom_group.buffer(15).intersection(geom_j.buffer(15))

                if hasattr(overlap, 'length') and overlap.length > 100:
                    group.append(j)
                    group_route_ids.update(all_segments[j]['route_ids'])

        # Mark all in group as used
        for idx in group:
            used_indices.add(idx)

        # If group has multiple segments, merge them
        if len(group) > 1:
            # Use the longest geometry as the merged geometry
            longest_idx = max(group, key=lambda idx: all_segments[idx]['geometry'].length)
            merged_geom = all_segments[longest_idx]['geometry']

            merged_segments.append({
                'geometry': merged_geom,
                'route_ids': sorted(list(group_route_ids)),
                'shape_id': all_segments[longest_idx]['shape_id'],  # Keep one for mapping
                'route_id': ','.join(sorted(list(group_route_ids)))
            })
        else:
            # Single segment, keep as is
            merged_segments.append(all_segments[i])

    logger.info(f"Merged {len(all_segments)} segments into {len(merged_segments)} unique segments")

    return gpd.GeoDataFrame(merged_segments, crs=routes.crs)


def substring(geom, start_dist, end_dist):
    """Extract substring of LineString between start and end distances"""
    from shapely.geometry import LineString

    if start_dist < 0 or end_dist > geom.length:
        return None

    coords = []
    current_dist = 0

    for i in range(len(geom.coords) - 1):
        p1 = Point(geom.coords[i])
        p2 = Point(geom.coords[i + 1])
        segment_length = p1.distance(p2)

        if current_dist + segment_length >= start_dist and current_dist <= end_dist:
            # This segment is within our range
            if current_dist < start_dist:
                # Interpolate start point
                fraction = (start_dist - current_dist) / segment_length
                start_pt = geom.interpolate(start_dist)
                coords.append((start_pt.x, start_pt.y))
            else:
                coords.append(geom.coords[i])

            if current_dist + segment_length > end_dist:
                # Interpolate end point
                end_pt = geom.interpolate(end_dist)
                coords.append((end_pt.x, end_pt.y))
                break
            else:
                coords.append(geom.coords[i + 1])

        current_dist += segment_length

    if len(coords) >= 2:
        return LineString(coords)
    return None


def create_animation():
    """Create animation with simple route-based segments"""

    logger.info("Loading continuous routes...")
    routes = gpd.read_file(ROUTE_LINES)
    routes['route_id'] = routes['route_id'].astype(str)
    routes = routes[routes['route_id'].isin(TARGET_ROUTES)]
    logger.info(f"Loaded {len(routes)} route shapes")

    logger.info("Loading schedule...")
    schedule = pd.read_csv(SCHEDULE)
    schedule['route_id'] = schedule['route_id'].astype(str)
    schedule = schedule[schedule['route_id'].isin(TARGET_ROUTES)]

    # Debug trip counts
    logger.info("Trips per route:")
    for route_id in TARGET_ROUTES:
        count = schedule[schedule['route_id'] == route_id]['trip_id'].nunique()
        logger.info(f"  Route {route_id}: {count} trips")

    # Parse times
    schedule['departure_time'] = pd.to_datetime(
        schedule['departure_time_str'], format='%H:%M:%S', errors='coerce'
    ).dt.time
    schedule = schedule.dropna(subset=['departure_time'])
    schedule['departure_seconds'] = schedule['departure_time'].apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second
    )

    logger.info(f"Schedule: {len(schedule)} entries, {schedule['trip_id'].nunique()} trips")

    # Create simple segments
    segments = create_segments_from_routes(routes)

    # Map shape_ids to segment indices
    # For merged segments, multiple shape_ids map to the same segment
    route_to_segments = {}
    for idx, segment in segments.iterrows():
        # A merged segment serves multiple routes
        for route_id in segment['route_ids']:
            # Find all shape_ids for this route_id
            matching_shapes = routes[routes['route_id'] == route_id]['shape_id'].tolist()
            for shape_id in matching_shapes:
                if shape_id not in route_to_segments:
                    route_to_segments[shape_id] = []
                if idx not in route_to_segments[shape_id]:
                    route_to_segments[shape_id].append(idx)

    # Pre-compute overlapping segments for efficient brightening
    # For each segment, find all segments (including itself) that share the same space
    segment_overlaps = {}
    for idx in range(len(segments)):
        segment = segments.iloc[idx]
        overlapping = [idx]  # Always include itself

        # Find all other segments that share routes with this one
        for other_idx in range(len(segments)):
            if other_idx != idx:
                other_segment = segments.iloc[other_idx]
                # Check if they share any route_ids (meaning they overlap spatially)
                if any(rid in segment['route_ids'] for rid in other_segment['route_ids']):
                    overlapping.append(other_idx)

        segment_overlaps[idx] = overlapping

    logger.info(f"Segment overlap mapping created:")
    for idx, overlaps in segment_overlaps.items():
        if len(overlaps) > 1:
            logger.info(f"  Segment {idx}: overlaps with {len(overlaps)-1} others")

    # Base brightness should be 5% of the maximum brightness
    calculated_base_brightness = MAX_BRIGHTNESS * 0.05

    # Count expected trips per segment
    logger.info("Calculating expected trips per segment...")
    trips_per_route = {}
    for route_id in TARGET_ROUTES:
        count = schedule[schedule['route_id'] == route_id]['trip_id'].nunique()
        trips_per_route[route_id] = count
        logger.info(f"  Route {route_id}: {count} trips")

    # For each segment, calculate how many trips it will get
    # (sum of trips from all routes that use it)
    expected_trips_per_segment = np.zeros(len(segments))
    for idx, segment in segments.iterrows():
        trips_for_segment = sum(trips_per_route.get(rid, 0) for rid in segment['route_ids'])
        expected_trips_per_segment[idx] = trips_for_segment

    max_expected_trips = expected_trips_per_segment.max()
    min_expected_trips = expected_trips_per_segment[expected_trips_per_segment > 0].min()

    logger.info(f"Expected trips per segment: min={min_expected_trips:.0f}, max={max_expected_trips:.0f}")

    # Set brightness so busiest segment reaches ~0.95 at 95% of its expected trips
    # Each actual TRIP (not vehicle-frame) should increment brightness
    brightness_per_trip = (MAX_BRIGHTNESS - calculated_base_brightness) / (max_expected_trips * 0.95)

    logger.info(f"Brightness calculation:")
    logger.info(f"  Base brightness: {calculated_base_brightness:.3f}")
    logger.info(f"  Brightness per trip: {brightness_per_trip:.6f}")
    logger.info(f"  Busiest segment ({max_expected_trips:.0f} expected trips) will reach ~0.95 brightness")
    logger.info(f"  Quietest segment ({min_expected_trips:.0f} expected trips) will reach ~{calculated_base_brightness + min_expected_trips * brightness_per_trip:.2f} brightness")

    # Setup figure
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Set extent
    half_size = FRAME_SIZE / 2
    xmin, xmax = CENTRAL_WARSAW_X - half_size, CENTRAL_WARSAW_X + half_size
    ymin, ymax = CENTRAL_WARSAW_Y - half_size, CENTRAL_WARSAW_Y + half_size
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Animation state
    # Start ALL segments at 0 brightness - they jump to base on first trip
    # This ensures uniform visibility at start, with brightness varying only based on usage
    segment_brightness = np.zeros(len(segments))
    segment_trip_count = np.zeros(len(segments))  # Track UNIQUE trips per segment
    vehicles = []
    total_frames = FPS * DURATION

    # Build route lookup
    route_lookup = {}
    for idx, row in routes.iterrows():
        route_lookup[row['shape_id']] = {
            'geometry': row.geometry,
            'vehicle': row['vehicle'],
            'route_id': row['route_id']
        }

    def update_frame(frame_num):
        nonlocal vehicles

        ax.clear()
        ax.set_facecolor('#1a1a1a')
        ax.axis('off')

        # Calculate current time
        start_seconds = 4 * 3600
        duration_seconds = HOURS * 3600
        current_seconds = start_seconds + (frame_num / total_frames) * duration_seconds
        current_time = f"{int(current_seconds // 3600):02d}:{int((current_seconds % 3600) // 60):02d}"
        frame_duration = duration_seconds / total_frames

        # Add new vehicles
        new_trips = schedule[
            (schedule['departure_seconds'] >= current_seconds - frame_duration/2) &
            (schedule['departure_seconds'] <= current_seconds + frame_duration/2) &
            (schedule['stop_sequence'] == 1)
        ]

        for _, trip in new_trips.iterrows():
            shape_id = trip['shape_id']
            if shape_id in route_lookup:
                route_info = route_lookup[shape_id]
                route_length = route_info['geometry'].length
                trip_duration = route_length / 4.7

                vehicles.append({
                    'shape_id': shape_id,
                    'vehicle': route_info['vehicle'],
                    'route': route_info['geometry'],
                    'route_id': route_info['route_id'],
                    'start_time': current_seconds,
                    'duration': trip_duration
                })

        # Update vehicles and brighten their segments
        active_vehicles = []
        for vehicle in vehicles:
            elapsed = current_seconds - vehicle['start_time']
            if elapsed > vehicle['duration']:
                continue

            progress = elapsed / vehicle['duration']
            position = vehicle['route'].interpolate(progress, normalized=True)
            vehicle['position'] = position

            # Find which segment the vehicle is currently on
            shape_id = vehicle['shape_id']
            if shape_id in route_to_segments:
                # Check each segment of this route to see if vehicle is on it
                for seg_idx in route_to_segments[shape_id]:
                    segment = segments.iloc[seg_idx]

                    # Check if vehicle's position is on this segment
                    if position.distance(segment.geometry) < 20:  # Within 20m
                        # Check if we've already brightened this segment for this vehicle
                        if 'brightened_segments' not in vehicle:
                            vehicle['brightened_segments'] = set()

                        if seg_idx not in vehicle['brightened_segments']:
                            vehicle['brightened_segments'].add(seg_idx)

                            # Brighten ONLY this segment (not overlapping ones)
                            # Each route's vehicles will brighten their own segments
                            # Visual overlap happens because segments are drawn on top of each other
                            segment_trip_count[seg_idx] += 1

                            # On first trip, jump to base brightness
                            # Then accumulate based on trip count
                            if segment_brightness[seg_idx] < calculated_base_brightness:
                                segment_brightness[seg_idx] = calculated_base_brightness
                            elif segment_brightness[seg_idx] < MAX_BRIGHTNESS:
                                segment_brightness[seg_idx] = min(
                                    segment_brightness[seg_idx] + brightness_per_trip,
                                    MAX_BRIGHTNESS,
                                    1.0
                                )
                        break  # Found the segment, stop checking

            active_vehicles.append(vehicle)

        vehicles = active_vehicles

        # Draw segments
        for idx, segment in segments.iterrows():
            brightness = segment_brightness[idx]

            if brightness > 0 and hasattr(segment.geometry, 'coords'):
                # Use red color with brightness controlling opacity
                color = '#E63946'  # Red
                x, y = segment.geometry.xy

                # Normalize brightness to [0, 1] for alpha
                normalized_brightness = min(max(brightness, 0.0), 1.0)

                # Glow (with constant alpha)
                ax.plot(x, y, color=color, linewidth=GLOW_WIDTH,
                       alpha=normalized_brightness * GLOW_ALPHA,
                       solid_capstyle='round',
                       solid_joinstyle='round', zorder=1)

                # Main line (alpha = brightness)
                ax.plot(x, y, color=color, linewidth=LINE_WIDTH,
                       alpha=normalized_brightness, solid_capstyle='round',
                       solid_joinstyle='round', zorder=2)

        # Draw vehicles
        positions, streaks = [], []

        for vehicle in vehicles:
            if 'position' in vehicle:
                current_pos = vehicle['position']
                positions.append(current_pos)

                # Create streak
                elapsed = current_seconds - vehicle['start_time']
                progress = elapsed / vehicle['duration']
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
                    streaks.append(streak_coords)

        # Draw
        if positions:
            # Red color for vehicles
            vehicle_color = '#E63946'
            outline_color = '#FF7075'  # Slightly brighter red for outline
            size = VEHICLE_SIZES['Tram']

            # Streaks
            for streak_coords in streaks:
                xs, ys = [c[0] for c in streak_coords], [c[1] for c in streak_coords]
                ax.plot(xs, ys, color=vehicle_color, linewidth=3, alpha=0.3,
                       solid_capstyle='round', zorder=11)

            # Dots
            xs, ys = [p.x for p in positions], [p.y for p in positions]
            ax.scatter(xs, ys, s=size, color=vehicle_color, alpha=0.6,
                      edgecolors=outline_color, linewidths=0.5, zorder=12)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Text
        ax.text(0.02, 0.98, f"Lines {', '.join(TARGET_ROUTES)} - {current_time}",
               transform=ax.transAxes, fontsize=24, color='white',
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), zorder=100)

        ax.text(0.02, 0.92, f"Active trams: {len(vehicles)}",
               transform=ax.transAxes, fontsize=16, color='white',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), zorder=100)

        ax.text(0.98, 0.02, "© Jacek Gęborys",
               transform=ax.transAxes, fontsize=12, color='white',
               verticalalignment='bottom', horizontalalignment='right',
               alpha=0.6, zorder=100)

        if frame_num % 30 == 0:
            pct = frame_num / total_frames * 100

            # Find busiest and quietest segments
            busiest_idx = segment_trip_count.argmax()
            quietest_idx = segment_trip_count.argmin()

            busiest_trips = segment_trip_count[busiest_idx]
            quietest_trips = segment_trip_count[quietest_idx]

            busiest_brightness = segment_brightness[busiest_idx]
            quietest_brightness = segment_brightness[quietest_idx]

            logger.info(f"Frame {frame_num}/{total_frames} ({pct:.0f}%) - {len(vehicles)} active")
            logger.info(f"  Busiest segment: {busiest_trips:.0f} trips, brightness={busiest_brightness:.3f}")
            logger.info(f"  Quietest segment: {quietest_trips:.0f} trips, brightness={quietest_brightness:.3f}")
            logger.info(f"  Brightness range: {segment_brightness.min():.3f} - {segment_brightness.max():.3f}")

    # Create animation
    logger.info(f"Generating {total_frames} frames...")
    anim = animation.FuncAnimation(fig, update_frame, frames=total_frames,
                                  interval=1000/FPS, blit=False)

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving to {OUTPUT}...")
    plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

    if not Path(FFMPEG_PATH).exists():
        logger.error(f"FFmpeg not found")
        gif_output = OUTPUT.parent / OUTPUT.name.replace('.mp4', '.gif')
        writer = animation.PillowWriter(fps=FPS)
        anim.save(gif_output, writer=writer, dpi=100)
        logger.info(f"Saved as GIF: {gif_output}")
        plt.close()
        return

    writer = animation.FFMpegWriter(fps=FPS, codec='libx264', bitrate=5000,
                                   metadata={'artist': 'Jacek Gęborys'},
                                   extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(OUTPUT, writer=writer, dpi=100)
    plt.close()

    logger.info(f"Done! {OUTPUT}")


if __name__ == "__main__":
    if not ROUTE_LINES.exists() or not SCHEDULE.exists():
        logger.error("Files not found")
        exit(1)

    create_animation()