"""
Junction-Based Segment Creator
================================
Extracts the successful segmentation logic from animate_segments_density.py
into a reusable preprocessing step.

Pipeline position: After route_line_builder.py, before animation/visualization

Inputs:
  - data/processed/route_lines_continuous.shp

Outputs:
  - data/processed/junction_segments.shp (segments with route_ids lists)
  - data/processed/junction_points.shp (debug: junction points for QGIS)

What it does:
  1. Finds junction points where routes converge/diverge
  2. Splits routes at these junctions
  3. Merges overlapping segments that share routes
  4. Outputs segments ready for animation or visualization
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from pathlib import Path
from tqdm import tqdm
import logging
from shapely.strtree import STRtree
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ROUTE_LINES = PROJECT_ROOT / "data" / "processed" / "route_lines_continuous.shp"
OUTPUT_SEGMENTS = PROJECT_ROOT / "data" / "processed" / "junction_segments.shp"
OUTPUT_JUNCTIONS = PROJECT_ROOT / "data" / "processed" / "junction_points.shp"
OUTPUT_JUNCTION_GROUPS = PROJECT_ROOT / "data" / "processed" / "junction_groups.shp"
OUTPUT_TRAM_SNAPPED = PROJECT_ROOT / "data" / "output" / "tram_routes_snapped.shp"

# PERFORMANCE CONFIG - Test with small area first!
TEST_MODE = False  # Set to False for full city processing
TEST_AREA_CENTER_X = 638000  # Central Warsaw X
TEST_AREA_CENTER_Y = 487000  # Central Warsaw Y
TEST_AREA_SIZE = 1000  # 2km x 2km test area
SKIP_INTERMEDIATE_VERTEX_SNAPPING = True  # Massive speedup, try True first


def substring(geom, start_dist, end_dist):
    """Extract substring of LineString between start and end distances"""
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


def create_segments_from_routes(routes, vehicle_filter=None, snap_distance=50,
                                output_junctions=None, output_junction_groups=None):
    """
    Build a node-to-node network for transit routes using junction groups as "black holes".

    Logic:
    1. Load existing junction_points and junction_groups
    2. For each route, find junctions within snap_distance (50m)
    3. Snap route to junction group centroids (junctions "suck in" nearby routes)
    4. Split routes at snapped points
    5. Create segments between consecutive junction groups

    This creates a proper network where routes connect at junction groups.

    Args:
        routes: GeoDataFrame of routes
        vehicle_filter: Vehicle type to filter ('Tram', 'Bus', 'Train', or None)
        snap_distance: Distance in meters to snap to junctions
        output_junctions: Path to save junction points (vehicle-specific)
        output_junction_groups: Path to save junction groups (vehicle-specific)
    """
    # Use provided paths or fall back to globals
    if output_junctions is None:
        output_junctions = output_junctions
    if output_junction_groups is None:
        output_junction_groups = output_junction_groups

    logger.info(f"Creating node-to-node network for {vehicle_filter}...")
    logger.info(f"Using junction group snap distance: {snap_distance}m")

    # Show performance settings
    if TEST_MODE:
        logger.info(f"⚡ TEST MODE: {TEST_AREA_SIZE}m x {TEST_AREA_SIZE}m area")
    if SKIP_INTERMEDIATE_VERTEX_SNAPPING:
        logger.info(f"⚡ SKIP_INTERMEDIATE_VERTEX_SNAPPING: Enabled (massive speedup!)")

    # Filter by vehicle type FIRST (critical for performance!)
    if vehicle_filter:
        routes_before = len(routes)
        routes = routes[routes['vehicle'] == vehicle_filter].copy()
        logger.info(f"🔍 PRE-FILTERED by vehicle type: {routes_before} → {len(routes)} {vehicle_filter} routes")
        logger.info(f"   (Junction finding will ONLY check {vehicle_filter}-to-{vehicle_filter} intersections)")

    if len(routes) == 0:
        logger.error(f"No routes found for vehicle type: {vehicle_filter}")
        return gpd.GeoDataFrame()

    # Ensure we're using projected CRS (meters)
    if routes.crs and routes.crs.is_geographic:
        logger.info(f"Converting from {routes.crs} to EPSG:2180 (Poland)...")
        routes = routes.to_crs('EPSG:2180')
    logger.info(f"Using CRS: {routes.crs}")

    # SPATIAL FILTER: Test mode - only process routes in small area
    if TEST_MODE:
        from shapely.geometry import box
        half_size = TEST_AREA_SIZE / 2
        test_bbox = box(
            TEST_AREA_CENTER_X - half_size,
            TEST_AREA_CENTER_Y - half_size,
            TEST_AREA_CENTER_X + half_size,
            TEST_AREA_CENTER_Y + half_size
        )
        logger.info(f"🔬 TEST MODE: Filtering to {TEST_AREA_SIZE}m x {TEST_AREA_SIZE}m area around center")
        routes_before = len(routes)
        routes = routes[routes.geometry.intersects(test_bbox)].copy()
        logger.info(f"   Filtered from {routes_before} to {len(routes)} routes in test area")

        if len(routes) == 0:
            logger.warning("No routes in test area!")
            return gpd.GeoDataFrame()

    # Step 1: Find all junction points (where routes converge/diverge)
    logger.info(f"Step 1/4: Finding junctions between {len(routes)} routes...")

    # Check if junctions already computed
    if output_junctions.exists():
        logger.info(f"⚡ Loading existing junction points from {output_junctions}...")
        junction_points = gpd.read_file(output_junctions)
        logger.info(f"✓ Loaded {len(junction_points)} junction points")
        logger.info(f"   (Delete {output_junctions} to regenerate)")
    else:
        # Build spatial index for fast neighbor queries
        logger.info("Building spatial index...")
        route_geometries = [route.geometry for _, route in routes.iterrows()]
        spatial_index = STRtree(route_geometries)

        junction_points_list = []
        junction_metadata = []  # Track which routes meet at each junction

        # Only check routes that are actually nearby (MUCH faster!)
        pairs_checked = 0
        pairs_with_intersection = 0

        for i in tqdm(range(len(routes)), desc="Finding junctions", unit="routes"):
            route_i = routes.iloc[i]
            geom_i = route_i.geometry

            # Use spatial index to find only nearby routes (within 100m buffer)
            nearby_indices = spatial_index.query(geom_i.buffer(100))

            for j in nearby_indices:
                # Only check each pair once (i < j) and skip self
                if j <= i:
                    continue

                pairs_checked += 1
                route_j = routes.iloc[j]
                geom_j = route_j.geometry

                # Quick bounding box check first (very fast)
                bounds_i = geom_i.bounds
                bounds_j = geom_j.bounds
                # Check if bounding boxes overlap (xmin, ymin, xmax, ymax)
                if (bounds_i[2] < bounds_j[0] or bounds_i[0] > bounds_j[2] or
                    bounds_i[3] < bounds_j[1] or bounds_i[1] > bounds_j[3]):
                    continue

                # Find where routes are nearby (within 15m = same street/track)
                # Use buffered intersection to handle GPS inaccuracies
                buffered_intersection = geom_i.intersection(geom_j.buffer(15, quad_segs=2))

                if not buffered_intersection.is_empty:
                    pairs_with_intersection += 1

                    # Extract junction points based on intersection type
                    junction_candidates = []

                    if buffered_intersection.geom_type == 'LineString':
                        coords = list(buffered_intersection.coords)
                        if len(coords) >= 2:
                            # Only add endpoints if they're not in the middle of a long parallel segment
                            # Check: are the endpoints actually divergence points?
                            start_pt = Point(coords[0])
                            end_pt = Point(coords[-1])

                            # Add start point only if it's near the actual route geometries
                            # (not just a buffer artifact)
                            if geom_i.distance(start_pt) < 20 and geom_j.distance(start_pt) < 20:
                                junction_candidates.append(start_pt)
                            if geom_i.distance(end_pt) < 20 and geom_j.distance(end_pt) < 20:
                                junction_candidates.append(end_pt)

                    elif buffered_intersection.geom_type == 'MultiLineString':
                        for line in buffered_intersection.geoms:
                            coords = list(line.coords)
                            if len(coords) >= 2:
                                start_pt = Point(coords[0])
                                end_pt = Point(coords[-1])

                                if geom_i.distance(start_pt) < 20 and geom_j.distance(start_pt) < 20:
                                    junction_candidates.append(start_pt)
                                if geom_i.distance(end_pt) < 20 and geom_j.distance(end_pt) < 20:
                                    junction_candidates.append(end_pt)

                    # Add validated junction points
                    for pt in junction_candidates:
                        junction_points_list.append(pt)
                        junction_metadata.append({
                            'route_1': route_i['route_id'],
                            'route_2': route_j['route_id'],
                            'point': pt
                        })

        logger.info(f"Checked {pairs_checked} nearby route pairs (instead of {(len(routes)*(len(routes)-1))//2} all pairs)")
        logger.info(f"Found intersections in {pairs_with_intersection} pairs")
        logger.info(f"Found {len(junction_points_list)} potential junction points")

        # CLUSTERING: Use DBSCAN to group nearby junction points
        if junction_points_list:
            logger.info(f"Clustering junction points with DBSCAN (eps={snap_distance}m)...")

            # Convert to coordinate array
            coords = np.array([[pt.x, pt.y] for pt in junction_points_list])

            # Use DBSCAN to cluster nearby points
            clustering = DBSCAN(eps=snap_distance, min_samples=1).fit(coords)
            labels = clustering.labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"DBSCAN clustering: {len(junction_points_list)} → {n_clusters} clusters ({100*(1-n_clusters/len(junction_points_list)):.1f}% reduction)")

            # Save ORIGINAL junction points (actual detected locations)
            output_junctions.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving {len(junction_points_list)} original junction points to {output_junctions}...")

            junction_points = gpd.GeoDataFrame({
                'geometry': junction_points_list,
                'cluster_id': labels
            }, crs=routes.crs)
            junction_points.to_file(output_junctions)
            logger.info(f"✓ Junction points saved (actual detected locations)")

            # Compute centroids for each cluster (will be saved as junction_groups)
            cluster_centroids = []
            cluster_ids = []
            for cluster_id in set(labels):
                if cluster_id == -1:  # Noise points (shouldn't happen with min_samples=1)
                    continue
                cluster_mask = labels == cluster_id
                cluster_coords = coords[cluster_mask]
                centroid = cluster_coords.mean(axis=0)
                cluster_centroids.append(Point(centroid))
                cluster_ids.append(cluster_id)

            logger.info(f"✓ Clustering complete: {len(cluster_centroids)} junction group centroids")

        else:
            logger.info("No junction points found")
            return gpd.GeoDataFrame()

    # Create or load junction groups (centroids)
    if not output_junction_groups.exists():
        # Check if we have pre-computed centroids from the clustering step
        if 'cluster_centroids' in locals() and 'cluster_ids' in locals():
            logger.info(f"\nCreating junction groups from pre-computed centroids...")
            junction_groups_data = []
            for cluster_id, centroid in zip(cluster_ids, cluster_centroids):
                cluster_points = junction_points[junction_points['cluster_id'] == cluster_id]
                junction_groups_data.append({
                    'cluster_id': cluster_id,
                    'num_points': len(cluster_points),
                    'geometry': centroid
                })
            junction_groups = gpd.GeoDataFrame(junction_groups_data, crs=junction_points.crs)
        else:
            # Fallback: compute centroids from junction_points
            logger.info(f"\nCreating junction groups from {len(junction_points)} junction points...")
            junction_groups_data = []
            for cluster_id in sorted(junction_points['cluster_id'].unique()):
                cluster_points = junction_points[junction_points['cluster_id'] == cluster_id]
                # Compute centroid of all points in this cluster
                coords = np.array([[pt.x, pt.y] for pt in cluster_points.geometry])
                centroid = Point(coords.mean(axis=0))
                junction_groups_data.append({
                    'cluster_id': cluster_id,
                    'num_points': len(cluster_points),
                    'geometry': centroid
                })
            junction_groups = gpd.GeoDataFrame(junction_groups_data, crs=junction_points.crs)

        # Save junction groups
        output_junction_groups.parent.mkdir(parents=True, exist_ok=True)
        junction_groups.to_file(output_junction_groups)
        logger.info(f"✓ Saved {len(junction_groups)} junction groups (centroids) to {output_junction_groups}")
    else:
        logger.info(f"⚡ Loading junction groups from {output_junction_groups}...")
        junction_groups = gpd.read_file(output_junction_groups)
        logger.info(f"✓ Loaded {len(junction_groups)} junction groups")

    # Create centroid lookup dictionary
    group_centroids = {row['cluster_id']: row['geometry'] for _, row in junction_groups.iterrows()}

    # Build spatial index for junction points for fast lookup
    logger.info("Building spatial index for junction points...")
    junction_point_tree = STRtree(list(junction_points.geometry))

    # Build spatial index for junction group centroids (for fast intermediate vertex snapping)
    logger.info("Building spatial index for junction group centroids...")
    centroid_list = list(junction_groups.geometry)
    centroid_tree = STRtree(centroid_list)
    centroid_to_cluster = {i: row['cluster_id'] for i, (_, row) in enumerate(junction_groups.iterrows())}

    # Step 2: Find junctions near routes and split at junctions (ORIGINAL WORKING APPROACH)
    logger.info(f"\nStep 2/4: Snapping routes to junction centroids...")

    from collections import defaultdict
    from shapely.geometry import box

    # Track segments by their junction endpoints (to merge and sum trip counts)
    segment_connections = defaultdict(lambda: {'route_ids': [], 'trip_counts': [], 'geometries': []})

    for idx, route in tqdm(routes.iterrows(), total=len(routes), desc="Processing routes", unit="routes"):
        route_geom = route.geometry
        route_id = route['route_id']
        vehicle_type = route['vehicle']
        trip_count = route.get('trip_count', 1)

        # Find junction centroids near this route using SPATIAL INDEX
        route_bounds = route_geom.bounds
        route_length = route_geom.length

        # Use spatial index to find nearby centroids (MUCH faster!)
        search_bbox = box(
            route_bounds[0] - snap_distance,
            route_bounds[1] - snap_distance,
            route_bounds[2] + snap_distance,
            route_bounds[3] + snap_distance
        )
        nearby_centroid_indices = centroid_tree.query(search_bbox)

        junctions_to_snap = []

        # Only check nearby centroids (not all of them!)
        for centroid_idx in nearby_centroid_indices:
            centroid = centroid_list[centroid_idx]
            cluster_id = centroid_to_cluster[centroid_idx]

            # Check actual distance
            dist = route_geom.distance(centroid)
            if dist <= snap_distance:
                # Project centroid onto route to find where to split
                snap_point = route_geom.project(centroid)
                junctions_to_snap.append((snap_point, cluster_id, centroid))

        if len(junctions_to_snap) < 2:
            continue

        # Sort by position along route
        junctions_to_snap.sort(key=lambda x: x[0])

        # Create segments between consecutive junctions
        for i in range(len(junctions_to_snap) - 1):
            dist_1, cluster_1, centroid_1 = junctions_to_snap[i]
            dist_2, cluster_2, centroid_2 = junctions_to_snap[i + 1]

            # Extract segment
            segment_geom = substring(route_geom, dist_1, dist_2)

            if segment_geom and segment_geom.length > 10:
                # Snap endpoints to junction centroids
                coords = list(segment_geom.coords)
                coords[0] = (centroid_1.x, centroid_1.y)
                coords[-1] = (centroid_2.x, centroid_2.y)
                snapped_segment = LineString(coords)

                # Create key for this junction pair (order doesn't matter)
                key = tuple(sorted([cluster_1, cluster_2]))

                # Store this segment
                segment_connections[key]['route_ids'].append(route_id)
                segment_connections[key]['trip_counts'].append(trip_count)
                segment_connections[key]['geometries'].append(snapped_segment)

    logger.info(f"✓ Found {len(segment_connections)} unique street segments")

    # Step 3: Merge segments and sum trip counts
    logger.info(f"\nStep 3/4: Merging routes and summing trip counts...")

    segments = []
    for key, data in tqdm(segment_connections.items(), desc="Merging segments", unit="segments"):
        route_ids = list(set(data['route_ids']))  # Unique routes
        total_trips = sum(data['trip_counts'])  # SUM all trips!
        best_geom = max(data['geometries'], key=lambda g: g.length)  # Use longest geometry

        segments.append({
            'geometry': best_geom,
            'route_ids': ','.join(sorted(route_ids)),
            'num_routes': len(route_ids),
            'trip_count': total_trips,  # TOTAL trips per day for this segment
            'vehicle_type': vehicle_type,
            'length_km': best_geom.length / 1000.0
        })

    logger.info(f"✓ Created {len(segments)} merged segments with total trip counts")

    # Step 4: Statistics
    logger.info(f"\nStep 4/4: Street segment statistics...")

    route_counts = {}
    for seg in segments:
        num_routes = seg['num_routes']
        route_counts[num_routes] = route_counts.get(num_routes, 0) + 1

    logger.info("Segment distribution by number of routes:")
    for num_routes in sorted(route_counts.keys()):
        logger.info(f"  {route_counts[num_routes]} segments with {num_routes} route(s)")

    total_length_km = sum(seg['length_km'] for seg in segments)
    total_trips = sum(seg['trip_count'] for seg in segments)
    avg_trips = total_trips / len(segments) if segments else 0

    logger.info(f"\nTotal network length: {total_length_km:.1f} km")
    logger.info(f"Average segment length: {total_length_km / len(segments):.2f} km")
    logger.info(f"Total trip count: {total_trips}")
    logger.info(f"Average trips per segment: {avg_trips:.1f}")

    # Find busiest segments
    if segments:
        top_segments = sorted(segments, key=lambda s: s['trip_count'], reverse=True)[:5]
        logger.info(f"\nBusiest segments:")
        for i, seg in enumerate(top_segments, 1):
            logger.info(f"  {i}. {seg['trip_count']} trips, {seg['num_routes']} routes, {seg['length_km']:.2f}km")

    return gpd.GeoDataFrame(segments, crs=routes.crs)


def create_tram_map_snapped(routes, junction_groups_path, snap_distance=50):
    """
    Extract tram routes and snap them to junction group centroids.

    Args:
        routes: GeoDataFrame of all routes
        junction_groups_path: Path to junction_groups.shp
        snap_distance: Distance in meters to snap endpoints to junction centroids

    Returns:
        GeoDataFrame of snapped tram routes
    """
    logger.info("\n" + "=" * 60)
    logger.info("Creating Tram Map Snapped to Junction Groups")
    logger.info("=" * 60)

    # Filter for tram routes
    tram_routes = routes[routes['vehicle'] == 'Tram'].copy()
    logger.info(f"\nFiltered to {len(tram_routes)} tram routes")

    if len(tram_routes) == 0:
        logger.error("No tram routes found!")
        return gpd.GeoDataFrame()

    # Load junction groups
    if not junction_groups_path.exists():
        logger.error(f"Junction groups file not found: {junction_groups_path}")
        return gpd.GeoDataFrame()

    junction_groups = gpd.read_file(junction_groups_path)
    logger.info(f"Loaded {len(junction_groups)} junction groups")

    # Compute centroids from geometry
    junction_centroids = junction_groups.geometry.centroid.tolist()
    logger.info(f"Snapping tram routes to junction centroids (distance: {snap_distance}m)...")

    def snap_line_to_junctions(line_geom, junction_centroids, snap_distance):
        """Snap line endpoints to nearby junction centroids."""
        if line_geom.is_empty or not isinstance(line_geom, LineString):
            return line_geom

        coords = list(line_geom.coords)

        # Snap start point
        start_point = Point(coords[0])
        for junction_centroid in junction_centroids:
            if start_point.distance(junction_centroid) < snap_distance:
                coords[0] = junction_centroid.coords[0]
                break

        # Snap end point
        end_point = Point(coords[-1])
        for junction_centroid in junction_centroids:
            if end_point.distance(junction_centroid) < snap_distance:
                coords[-1] = junction_centroid.coords[0]
                break

        return LineString(coords)

    # Apply snapping
    snapped_count = 0
    for idx in tqdm(range(len(tram_routes)), desc="Snapping routes", unit="routes"):
        original_geom = tram_routes.iloc[idx].geometry
        snapped_geom = snap_line_to_junctions(original_geom, junction_centroids, snap_distance)

        # Check if geometry was actually modified
        if not original_geom.equals(snapped_geom):
            snapped_count += 1

        tram_routes.iloc[idx, tram_routes.columns.get_loc('geometry')] = snapped_geom

    logger.info(f"✓ Snapped {snapped_count} of {len(tram_routes)} tram routes")

    # Statistics
    total_length_km = tram_routes.geometry.length.sum() / 1000
    avg_trip_count = tram_routes['trip_count'].mean()
    total_trips = tram_routes['trip_count'].sum()

    logger.info(f"\nTram Network Summary:")
    logger.info(f"  Routes: {len(tram_routes)}")
    logger.info(f"  Total length: {total_length_km:.2f} km")
    logger.info(f"  Total trips: {total_trips}")
    logger.info(f"  Average trips per route: {avg_trip_count:.1f}")

    return tram_routes


def main():
    """Create junction-based segments from route lines"""

    logger.info("=" * 60)
    logger.info("Node-to-Node Network Builder")
    logger.info("=" * 60)

    # Note: Junction files are now created per vehicle type, so no pre-check needed
    logger.info("\n💡 Creating separate junction networks for each vehicle type...")

    if not ROUTE_LINES.exists():
        logger.error(f"❌ Route lines not found: {ROUTE_LINES}")
        logger.error("Run core/route_line_builder.py first")
        return 1

    logger.info(f"\nLoading routes from {ROUTE_LINES}...")
    routes = gpd.read_file(ROUTE_LINES)
    routes['route_id'] = routes['route_id'].astype(str)
    logger.info(f"✓ Loaded {len(routes)} route shapes")

    # Show vehicle type distribution
    vehicle_counts = routes['vehicle'].value_counts()
    logger.info("\nVehicle type distribution:")
    for vehicle, count in vehicle_counts.items():
        logger.info(f"  {vehicle}: {count} routes")

    # Create segments for EACH vehicle type separately (keeps networks independent)
    logger.info("\n🚊🚌🚆 Building separate networks for each vehicle type...")

    all_segments = []
    vehicle_types = ['Bus', 'Train'] # 'Tram' ,
    created_files = []

    for vehicle_type in vehicle_types:
        # Check if this vehicle type exists in the data
        if vehicle_type not in routes['vehicle'].values:
            logger.info(f"⏭️  No {vehicle_type} routes found, skipping...")
            continue

        # Create vehicle-specific output paths
        vehicle_lower = vehicle_type.lower()
        vehicle_junctions = PROJECT_ROOT / "data" / "processed" / f"{vehicle_lower}_junction_points.shp"
        vehicle_junction_groups = PROJECT_ROOT / "data" / "processed" / f"{vehicle_lower}_junction_groups.shp"
        vehicle_segments_path = PROJECT_ROOT / "data" / "processed" / f"{vehicle_lower}_junction_segments.shp"

        # Check if segments already exist
        if vehicle_segments_path.exists():
            logger.info(f"✅ {vehicle_type} segments already exist, skipping...")
            logger.info(f"   (Delete {vehicle_segments_path.name} to regenerate)")
            # Load existing segments for the combined file
            existing_segments = gpd.read_file(vehicle_segments_path)
            all_segments.append(existing_segments)
            created_files.extend([vehicle_junctions, vehicle_junction_groups, vehicle_segments_path])
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {vehicle_type.upper()} network...")
        logger.info(f"{'='*60}")

        # Vehicle-specific snap distances
        snap_distances = {'Tram': 50, 'Bus': 2, 'Train': 50}
        snap_dist = snap_distances.get(vehicle_type, 50)
        logger.info(f"Using snap distance: {snap_dist}m for {vehicle_type}")

        # Create segments with vehicle-specific paths
        vehicle_segments = create_segments_from_routes(
            routes,
            vehicle_filter=vehicle_type,
            snap_distance=snap_dist,
            output_junctions=vehicle_junctions,
            output_junction_groups=vehicle_junction_groups
        )

        if len(vehicle_segments) > 0:
            # Save vehicle-specific segments
            vehicle_segments_path.parent.mkdir(parents=True, exist_ok=True)
            vehicle_segments.to_file(vehicle_segments_path)
            logger.info(f"✓ Saved {len(vehicle_segments)} {vehicle_type} segments to {vehicle_segments_path.name}")

            all_segments.append(vehicle_segments)
            created_files.extend([vehicle_junctions, vehicle_junction_groups, vehicle_segments_path])
        else:
            logger.warning(f"⚠️  No segments created for {vehicle_type}")

    if len(all_segments) == 0:
        logger.error("❌ No segments created for any vehicle type!")
        return 1

    # Combine all segments into one GeoDataFrame for convenience
    segments = gpd.GeoDataFrame(pd.concat(all_segments, ignore_index=True))
    logger.info(f"\n✓ Total segments across all vehicle types: {len(segments)}")

    # Save combined segments
    OUTPUT_SEGMENTS.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n💾 Saving combined segments to {OUTPUT_SEGMENTS}...")
    segments.to_file(OUTPUT_SEGMENTS)
    created_files.append(OUTPUT_SEGMENTS)
    logger.info("✓ Segments saved")

    # Create tram map snapped to junction groups
    logger.info("\n🚊 Creating tram map snapped to junction groups...")
    tram_junction_groups = PROJECT_ROOT / "data" / "processed" / "tram_junction_groups.shp"
    if tram_junction_groups.exists():
        tram_routes_snapped = create_tram_map_snapped(routes, tram_junction_groups)

        if len(tram_routes_snapped) > 0:
            OUTPUT_TRAM_SNAPPED.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"\n💾 Saving snapped tram routes to {OUTPUT_TRAM_SNAPPED}...")
            tram_routes_snapped.to_file(OUTPUT_TRAM_SNAPPED)
            logger.info("✓ Tram routes saved")
            created_files.append(OUTPUT_TRAM_SNAPPED)
    else:
        logger.info("⏭️  Skipping tram map (no tram junction groups found)")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Done! Network segments ready")
    logger.info("=" * 60)
    logger.info("\nFiles created:")
    logger.info(f"  📏 Combined segments:  {OUTPUT_SEGMENTS}")
    logger.info("\nVehicle-specific files:")
    for vehicle_type in vehicle_types:
        vehicle_lower = vehicle_type.lower()
        junctions = PROJECT_ROOT / "data" / "processed" / f"{vehicle_lower}_junction_points.shp"
        groups = PROJECT_ROOT / "data" / "processed" / f"{vehicle_lower}_junction_groups.shp"
        segs = PROJECT_ROOT / "data" / "processed" / f"{vehicle_lower}_junction_segments.shp"
        if segs.exists():
            logger.info(f"\n  {vehicle_type}:")
            logger.info(f"    📍 Junction points:  {junctions.name}")
            logger.info(f"    🔗 Junction groups:  {groups.name}")
            logger.info(f"    📏 Segments:         {segs.name}")
    if OUTPUT_TRAM_SNAPPED.exists():
        logger.info(f"\n  🚊 Tram routes (snap): {OUTPUT_TRAM_SNAPPED.name}")
    logger.info("\nNext steps:")
    logger.info("  1. Open files in QGIS to preview each vehicle network separately")
    logger.info("  2. Check {vehicle}_junction_groups.shp for each vehicle type")
    logger.info("  3. Run animate/animate_full_density.py to create animation")

    return 0


if __name__ == "__main__":
    exit(main())
