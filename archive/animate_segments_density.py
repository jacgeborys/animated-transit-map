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
OUTPUT = PROJECT_ROOT / "animate" / "_output" / "warsaw_segments_density.mp4"

# Animation settings
FPS = 30
DURATION = 60
HOURS = 21
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"

# Map extent
CENTRAL_WARSAW_X = 638000
CENTRAL_WARSAW_Y = 487000
FRAME_SIZE = 25000

# Visual settings
COLORS = {'Tram': '#E63946', 'Bus': '#7209B7', 'Train': '#2A9D8F'}
VEHICLE_SIZES = {'Tram': 16, 'Bus': 12, 'Train': 24}
LINE_WIDTH = 1.8
GLOW_WIDTH = 4.0  # Stronger glow
GLOW_ALPHA = 0.5  # Stronger glow alpha
BASE_BRIGHTNESS = 0.1  # Higher base so segments are always visible
MAX_BRIGHTNESS = 1.0
OUTLINE_COLORS = {'Tram': '#FF7075', 'Bus': '#B46EFC', 'Train': '#6BC9C6'}
STREAK_LENGTH = 250

# Brightness calibration: max brightness at what % of peak observed density
MAX_BRIGHTNESS_AT_PERCENTILE = 0.30  # 80% of peak density = max brightness

# Rolling window for density calculation (smoother, less flickering)
ROLLING_WINDOW_MINUTES = 30  # Count vehicles in past 10 minutes

# Target routes
TARGET_ROUTES = ['4', '13', '20', '23', '26']


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
                'route_id': ','.join(sorted(list(group_route_ids))),
                'length_km': merged_geom.length / 1000.0  # Store length in km
            })
        else:
            # Single segment, keep as is
            all_segments[i]['length_km'] = all_segments[i]['geometry'].length / 1000.0
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

    # Base brightness - segments always visible
    calculated_base_brightness = BASE_BRIGHTNESS

    # Build route lookup (needed for peak density estimation)
    route_lookup = {}
    for idx, row in routes.iterrows():
        route_lookup[row['shape_id']] = {
            'geometry': row.geometry,
            'vehicle': row['vehicle'],
            'route_id': row['route_id']
        }

    # For density-based visualization, pre-calculate expected peak density
    # Count max concurrent vehicles expected on any segment
    logger.info("Estimating peak density from schedule...")

    # Sample the schedule at multiple time points to find peak
    start_seconds = 4 * 3600
    duration_seconds = HOURS * 3600
    sample_times = np.linspace(start_seconds, start_seconds + duration_seconds, 100)

    max_expected_density = 0.0

    for sample_time in sample_times:
        # Count vehicles active at this time
        segment_vehicle_counts = [0] * len(segments)

        for _, trip in schedule.iterrows():
            if trip['stop_sequence'] != 1:
                continue

            departure = trip['departure_seconds']
            shape_id = trip['shape_id']

            if shape_id not in route_lookup:
                continue

            route_info = route_lookup[shape_id]
            route_length = route_info['geometry'].length
            trip_duration = route_length / 4.7

            # Is this trip active at sample_time?
            if departure <= sample_time <= departure + trip_duration:
                # Find which segment it's on (approximate)
                if shape_id in route_to_segments:
                    # Just count it on the first segment for estimation
                    seg_indices = route_to_segments[shape_id]
                    if seg_indices and len(seg_indices) > 0:
                        segment_vehicle_counts[seg_indices[0]] += 1

        # Calculate densities for this time point
        for idx, segment in segments.iterrows():
            if segment_vehicle_counts[idx] > 0:
                length_km = segment['length_km']
                density = segment_vehicle_counts[idx] / length_km if length_km > 0 else 0
                if density > max_expected_density:
                    max_expected_density = density

    # Set calibration point
    density_for_max_brightness = max_expected_density * MAX_BRIGHTNESS_AT_PERCENTILE

    logger.info(f"Density-based brightness:")
    logger.info(f"  Base brightness: {calculated_base_brightness:.3f}")
    logger.info(f"  Estimated peak density: {max_expected_density:.1f} vehicles/km")
    logger.info(f"  Max brightness at {MAX_BRIGHTNESS_AT_PERCENTILE*100:.0f}% of peak = {density_for_max_brightness:.1f} vehicles/km")

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
    # Track vehicle presence over time for accurate occupancy calculation
    # Each entry: (vehicle_id, entry_time, exit_time or None if still present)
    segment_vehicle_visits = [[] for _ in range(len(segments))]
    segment_max_density = np.zeros(len(segments))  # Track max density per segment for logging
    vehicles = []
    total_frames = FPS * DURATION
    rolling_window_seconds = ROLLING_WINDOW_MINUTES * 60

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
                    'id': trip['trip_id'],  # Unique ID for tracking
                    'shape_id': shape_id,
                    'vehicle': route_info['vehicle'],
                    'route': route_info['geometry'],
                    'route_id': route_info['route_id'],
                    'start_time': current_seconds,
                    'duration': trip_duration,
                    'current_segment': None  # Track which segment vehicle is on
                })

        # Update vehicles and track segment occupancy over time
        active_vehicles = []
        for vehicle in vehicles:
            elapsed = current_seconds - vehicle['start_time']
            if elapsed > vehicle['duration']:
                # Vehicle finished - mark exit time on its last segment
                if 'current_segment' in vehicle and vehicle['current_segment'] is not None:
                    seg_idx = vehicle['current_segment']
                    # Find this vehicle's last visit and mark exit time
                    visits = segment_vehicle_visits[seg_idx]
                    for visit in reversed(visits):
                        if visit[0] == vehicle['id'] and visit[2] is None:
                            visit[2] = current_seconds  # Mark exit time
                            break
                continue

            progress = elapsed / vehicle['duration']
            position = vehicle['route'].interpolate(progress, normalized=True)
            vehicle['position'] = position

            old_segment = vehicle.get('current_segment', None)

            # Find which segment the vehicle is currently on
            shape_id = vehicle['shape_id']
            new_segment = None

            if shape_id in route_to_segments:
                # Check each segment of this route to see if vehicle is on it
                for seg_idx in route_to_segments[shape_id]:
                    segment = segments.iloc[seg_idx]

                    # Check if vehicle's position is on this segment
                    if position.distance(segment.geometry) < 20:  # Within 20m
                        new_segment = seg_idx
                        break  # Found the segment, stop checking

            # Handle segment transitions
            if new_segment != old_segment:
                # Exit old segment
                if old_segment is not None:
                    visits = segment_vehicle_visits[old_segment]
                    for visit in reversed(visits):
                        if visit[0] == vehicle['id'] and visit[2] is None:
                            visit[2] = current_seconds  # Mark exit time
                            break

                # Enter new segment
                if new_segment is not None:
                    segment_vehicle_visits[new_segment].append([vehicle['id'], current_seconds, None])
                    vehicle['current_segment'] = new_segment

            active_vehicles.append(vehicle)

        vehicles = active_vehicles

        # Calculate brightness based on segment occupancy rate (not just count)
        segment_brightness = np.full(len(segments), calculated_base_brightness)

        for idx, segment in segments.iterrows():
            visits = segment_vehicle_visits[idx]

            # Remove old visits outside rolling window
            cutoff_time = current_seconds - rolling_window_seconds
            visits[:] = [v for v in visits if v[2] is None or v[2] >= cutoff_time]

            # Calculate total vehicle-seconds in the window
            total_vehicle_seconds = 0
            for vehicle_id, entry_time, exit_time in visits:
                # Clip to window boundaries
                actual_entry = max(entry_time, cutoff_time)
                actual_exit = min(exit_time if exit_time else current_seconds, current_seconds)

                if actual_exit > actual_entry:
                    total_vehicle_seconds += (actual_exit - actual_entry)

            if total_vehicle_seconds > 0:
                # Occupancy rate: average number of vehicles present in the window
                # For a single vehicle: vehicle_seconds / window_seconds = fraction of time present
                # For multiple vehicles: can exceed 1.0 (e.g., 2 simultaneous vehicles = 2.0)
                occupancy_rate = total_vehicle_seconds / rolling_window_seconds

                # Use occupancy rate directly as our "density" metric
                # This represents traffic flow/intensity regardless of segment length
                # 1 tram on short segment for brief time = same as 1 tram on long segment for long time
                density = occupancy_rate

                # Track max density per segment for logging
                if density > segment_max_density[idx]:
                    segment_max_density[idx] = density

                # Convert density to brightness using pre-calculated peak
                brightness_ratio = density / density_for_max_brightness
                brightness = calculated_base_brightness + (MAX_BRIGHTNESS - calculated_base_brightness) * brightness_ratio
                brightness = min(brightness, MAX_BRIGHTNESS)

                segment_brightness[idx] = brightness

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
        ax.text(0.02, 0.98, f"Lines {', '.join(TARGET_ROUTES)} - Vehicle Density - {current_time}",
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

            # Calculate current densities based on occupancy
            current_densities = np.zeros(len(segments))
            for idx, segment in segments.iterrows():
                visits = segment_vehicle_visits[idx]
                cutoff_time = current_seconds - rolling_window_seconds

                total_vehicle_seconds = 0
                for vehicle_id, entry_time, exit_time in visits:
                    if exit_time is None or exit_time >= cutoff_time:
                        actual_entry = max(entry_time, cutoff_time)
                        actual_exit = min(exit_time if exit_time else current_seconds, current_seconds)
                        if actual_exit > actual_entry:
                            total_vehicle_seconds += (actual_exit - actual_entry)

                occupancy_rate = total_vehicle_seconds / rolling_window_seconds
                length_km = segment['length_km']
                current_densities[idx] = occupancy_rate / length_km if length_km > 0 else 0

            if len(current_densities[current_densities > 0]) > 0:
                busiest_idx = current_densities.argmax()
                busiest_density = current_densities[busiest_idx]
                busiest_brightness = segment_brightness[busiest_idx]
                num_lit_segments = len(current_densities[current_densities > 0])
            else:
                busiest_density = 0
                busiest_brightness = calculated_base_brightness
                num_lit_segments = 0

            logger.info(f"Frame {frame_num}/{total_frames} ({pct:.0f}%) - {len(vehicles)} active")
            logger.info(f"  Busiest occupancy (10min avg): {busiest_density:.2f} vehicles, brightness={busiest_brightness:.3f}")
            logger.info(f"  Segments with activity: {num_lit_segments}/{len(segments)}")
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