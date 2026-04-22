"""
Simple Junction Finder
======================
Just finds where routes intersect and groups them - no segment creation.
"""
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
from tqdm import tqdm
import logging
from shapely.strtree import STRtree
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ROUTE_LINES = PROJECT_ROOT / "data" / "processed" / "route_lines_continuous.shp"
OUTPUT_JUNCTIONS = PROJECT_ROOT / "data" / "processed" / "junction_groups.shp"


def find_junctions(routes, vehicle_filter='Tram'):
    """
    Find where routes intersect and cluster them into junction groups.
    Returns junction points with metadata.
    """
    logger.info(f"Finding junctions for {vehicle_filter}...")

    # Filter by vehicle type
    if vehicle_filter:
        routes = routes[routes['vehicle'] == vehicle_filter].copy()
        logger.info(f"Filtered to {len(routes)} {vehicle_filter} routes")

    # Ensure projected CRS
    if routes.crs and routes.crs.is_geographic:
        logger.info(f"Converting to EPSG:2180...")
        routes = routes.to_crs('EPSG:2180')
    logger.info(f"Using CRS: {routes.crs}")

    # Build spatial index
    logger.info("Building spatial index...")
    route_geometries = [route.geometry for _, route in routes.iterrows()]
    spatial_index = STRtree(route_geometries)

    junction_points = []
    junction_metadata = []

    # Find intersections
    pairs_checked = 0
    pairs_with_intersection = 0

    for i in tqdm(range(len(routes)), desc="Finding junctions", unit="routes"):
        route_i = routes.iloc[i]
        geom_i = route_i.geometry

        # Find nearby routes using spatial index
        nearby_indices = spatial_index.query(geom_i.buffer(100))

        for j in nearby_indices:
            if j <= i:  # Only check each pair once
                continue

            pairs_checked += 1
            route_j = routes.iloc[j]
            geom_j = route_j.geometry

            # Find intersection with buffer (15m tolerance)
            intersection = geom_i.intersection(geom_j.buffer(15, quad_segs=2))

            if not intersection.is_empty:
                pairs_with_intersection += 1

                # Extract junction points from intersection geometry
                if intersection.geom_type == 'LineString':
                    coords = list(intersection.coords)
                    if len(coords) >= 2:
                        # Add start and end of overlapping section
                        junction_points.append(Point(coords[0]))
                        junction_points.append(Point(coords[-1]))
                        junction_metadata.append({
                            'route_1': route_i['route_id'],
                            'route_2': route_j['route_id'],
                            'trips_1': route_i.get('trip_count', 0),
                            'trips_2': route_j.get('trip_count', 0)
                        })
                        junction_metadata.append({
                            'route_1': route_i['route_id'],
                            'route_2': route_j['route_id'],
                            'trips_1': route_i.get('trip_count', 0),
                            'trips_2': route_j.get('trip_count', 0)
                        })

                elif intersection.geom_type == 'MultiLineString':
                    for line in intersection.geoms:
                        coords = list(line.coords)
                        if len(coords) >= 2:
                            junction_points.append(Point(coords[0]))
                            junction_points.append(Point(coords[-1]))
                            junction_metadata.append({
                                'route_1': route_i['route_id'],
                                'route_2': route_j['route_id'],
                                'trips_1': route_i.get('trip_count', 0),
                                'trips_2': route_j.get('trip_count', 0)
                            })
                            junction_metadata.append({
                                'route_1': route_i['route_id'],
                                'route_2': route_j['route_id'],
                                'trips_1': route_i.get('trip_count', 0),
                                'trips_2': route_j.get('trip_count', 0)
                            })

    logger.info(f"Checked {pairs_checked} route pairs")
    logger.info(f"Found {pairs_with_intersection} pairs with intersections")
    logger.info(f"Found {len(junction_points)} junction points")

    if not junction_points:
        logger.warning("No junctions found!")
        return gpd.GeoDataFrame()

    # Cluster junctions within 30m using DBSCAN
    logger.info("Clustering junctions within 30m...")
    coords = np.array([[pt.x, pt.y] for pt in junction_points])

    clustering = DBSCAN(eps=30, min_samples=1, metric='euclidean')
    labels = clustering.fit_predict(coords)

    n_clusters = len(set(labels))
    logger.info(f"Clustered into {n_clusters} junction groups")

    # Create junction groups (centroids)
    junction_groups = []

    for cluster_id in tqdm(range(n_clusters), desc="Creating junction groups", unit="groups"):
        cluster_mask = labels == cluster_id
        cluster_coords = coords[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]

        # Centroid
        centroid = Point(cluster_coords.mean(axis=0))

        # Collect all routes at this junction
        routes_at_junction = set()
        total_trips = 0

        for idx in cluster_indices:
            meta = junction_metadata[idx]
            routes_at_junction.add(meta['route_1'])
            routes_at_junction.add(meta['route_2'])
            total_trips += meta['trips_1'] + meta['trips_2']

        junction_groups.append({
            'geometry': centroid,
            'group_id': cluster_id,
            'num_points': len(cluster_indices),
            'num_routes': len(routes_at_junction),
            'route_ids': ','.join(sorted(routes_at_junction)),
            'total_trips': total_trips,
            'x': centroid.x,
            'y': centroid.y
        })

    logger.info(f"Created {len(junction_groups)} junction groups")

    # Statistics
    single_junctions = sum(1 for jg in junction_groups if jg['num_points'] == 1)
    logger.info(f"  - {single_junctions} single-point junctions")
    logger.info(f"  - {len(junction_groups) - single_junctions} clustered junctions")

    max_points = max(jg['num_points'] for jg in junction_groups)
    max_routes = max(jg['num_routes'] for jg in junction_groups)
    logger.info(f"  - Max {max_points} points merged into one junction")
    logger.info(f"  - Max {max_routes} routes at one junction")

    return gpd.GeoDataFrame(junction_groups, crs=routes.crs)


def main():
    """Find junctions and save them"""

    logger.info("=" * 60)
    logger.info("Simple Junction Finder")
    logger.info("=" * 60)

    if not ROUTE_LINES.exists():
        logger.error(f"Route lines not found: {ROUTE_LINES}")
        logger.error("Run core/route_line_builder.py first")
        return 1

    logger.info(f"\nLoading routes from {ROUTE_LINES}...")
    routes = gpd.read_file(ROUTE_LINES)
    routes['route_id'] = routes['route_id'].astype(str)
    logger.info(f"✓ Loaded {len(routes)} route shapes")

    # Show vehicle distribution
    vehicle_counts = routes['vehicle'].value_counts()
    logger.info("\nVehicle distribution:")
    for vehicle, count in vehicle_counts.items():
        logger.info(f"  {vehicle}: {count} routes")

    # Find junctions (trams only for testing)
    logger.info("\n🚊 Finding junctions for TRAMS only")
    junctions = find_junctions(routes, vehicle_filter='Tram')

    if len(junctions) == 0:
        logger.error("No junctions found!")
        return 1

    # Save
    OUTPUT_JUNCTIONS.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n💾 Saving junction groups to {OUTPUT_JUNCTIONS}...")
    junctions.to_file(OUTPUT_JUNCTIONS)
    logger.info("✓ Saved")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Done!")
    logger.info("=" * 60)
    logger.info(f"\nFiles:")
    logger.info(f"  📍 Junction groups: {OUTPUT_JUNCTIONS}")
    logger.info(f"  🚊 Route shapes:    {ROUTE_LINES}")
    logger.info("\nOpen both in QGIS to visualize:")
    logger.info("  - Route lines (red)")
    logger.info("  - Junction groups (cyan dots)")

    return 0


if __name__ == "__main__":
    exit(main())
