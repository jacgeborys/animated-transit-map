"""
Line Segmentation Module

Splits transit route lines at intersection points to create street segments
that can accumulate brightness from multiple overlapping routes.
"""
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import split, linemerge, unary_union
from pathlib import Path
from typing import List, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class LineSegmenter:
    """Segment transit lines at intersection points"""
    
    def __init__(self, route_lines_gdf: gpd.GeoDataFrame):
        """
        Initialize segmenter
        
        Args:
            route_lines_gdf: GeoDataFrame with continuous route lines
        """
        self.route_lines = route_lines_gdf.copy()
        self.segments = None
        self.intersection_points = None
        
    def find_intersections(self, tolerance: float = 1.0) -> MultiPoint:
        """
        Find intersection points between all routes

        Args:
            tolerance: Distance tolerance for considering lines as intersecting (meters)

        Returns:
            MultiPoint of intersection locations
        """
        logger.info("Finding intersection points...")

        intersections = []

        # Check each pair of routes
        for i in tqdm(range(len(self.route_lines)), desc="Finding intersections"):
            line1 = self.route_lines.iloc[i].geometry

            for j in range(i + 1, len(self.route_lines)):
                line2 = self.route_lines.iloc[j].geometry

                # Only consider actual crossings, not parallel routes
                if line1.intersects(line2):
                    intersection = line1.intersection(line2)

                    # Extract points from intersection
                    if intersection.geom_type == 'Point':
                        intersections.append(intersection)
                    elif intersection.geom_type == 'MultiPoint':
                        intersections.extend(list(intersection.geoms))
                    # Skip LineString overlaps - these are parallel routes, not crossings

        logger.info(f"Found {len(intersections)} raw intersection points")

        # Remove duplicates by clustering nearby points
        if intersections:
            unique_intersections = self._cluster_points(intersections, tolerance * 10)
            self.intersection_points = MultiPoint(unique_intersections)
            logger.info(f"Clustered to {len(unique_intersections)} unique intersections")
            return self.intersection_points

        return MultiPoint([])

    def _cluster_points(self, points: List[Point], tolerance: float) -> List[Point]:
        """
        Cluster nearby points to remove duplicates

        Args:
            points: List of Point objects
            tolerance: Distance to consider points as same

        Returns:
            List of unique points (cluster centers)
        """
        if not points:
            return []

        # Simple clustering: merge points within tolerance
        unique = []
        used = set()

        for i, p1 in enumerate(points):
            if i in used:
                continue

            cluster = [p1]
            used.add(i)

            for j, p2 in enumerate(points[i+1:], start=i+1):
                if j not in used and p1.distance(p2) < tolerance:
                    cluster.append(p2)
                    used.add(j)

            # Use cluster centroid
            coords = [p.coords[0] for p in cluster]
            centroid_x = np.mean([c[0] for c in coords])
            centroid_y = np.mean([c[1] for c in coords])
            unique.append(Point(centroid_x, centroid_y))

        return unique

    def split_at_intersections(self) -> gpd.GeoDataFrame:
        """
        Split route lines at intersection points

        Returns:
            GeoDataFrame with line segments
        """
        if self.intersection_points is None or len(self.intersection_points.geoms) == 0:
            logger.warning("No intersection points found, returning original lines")
            return self.route_lines

        logger.info("Splitting lines at intersections...")

        segments = []

        for idx, row in tqdm(self.route_lines.iterrows(),
                            total=len(self.route_lines),
                            desc="Splitting lines"):
            line = row.geometry

            # Find intersection points on this line
            points_on_line = []
            for point in self.intersection_points.geoms:
                if line.distance(point) < 10:  # Within 10m
                    # Project point onto line to get exact position
                    distance = line.project(point)
                    projected = line.interpolate(distance)
                    points_on_line.append((distance, projected))

            if not points_on_line:
                # No intersections, keep whole line as segment
                segments.append({
                    'geometry': line,
                    'shape_id': row['shape_id'],
                    'route_id': row['route_id'],
                    'vehicle': row['vehicle'],
                    'trip_count': row['trip_count'],
                    'length': line.length
                })
                continue

            # Sort points by distance along line
            points_on_line.sort(key=lambda x: x[0])

            # Add start and end points
            coords = list(line.coords)
            all_distances = [0] + [p[0] for p in points_on_line] + [line.length]

            # Create segments between consecutive points
            for i in range(len(all_distances) - 1):
                start_dist = all_distances[i]
                end_dist = all_distances[i + 1]

                if end_dist - start_dist < 5:  # Skip very short segments
                    continue

                # Extract segment
                start_point = line.interpolate(start_dist)
                end_point = line.interpolate(end_dist)

                # Get all points between start and end
                segment_coords = []
                for coord in coords:
                    p = Point(coord)
                    d = line.project(p)
                    if start_dist <= d <= end_dist:
                        segment_coords.append(coord)

                if len(segment_coords) >= 2:
                    segment_line = LineString(segment_coords)

                    segments.append({
                        'geometry': segment_line,
                        'shape_id': row['shape_id'],
                        'route_id': row['route_id'],
                        'vehicle': row['vehicle'],
                        'trip_count': row['trip_count'],
                        'length': segment_line.length
                    })

        logger.info(f"Created {len(segments)} segments from {len(self.route_lines)} routes")

        self.segments = gpd.GeoDataFrame(segments, crs=self.route_lines.crs)
        return self.segments

    def merge_overlapping_segments(self,
                                   buffer_distance: float = 4.0,
                                   direction_tolerance: float = 10.0) -> gpd.GeoDataFrame:
        """
        Merge segments from different routes that use the same street

        Args:
            buffer_distance: Distance to consider segments as overlapping (meters)
            direction_tolerance: Max bearing difference for matching (degrees)

        Returns:
            GeoDataFrame with merged street segments
        """
        if self.segments is None:
            raise ValueError("Run split_at_intersections() first")

        logger.info("Merging overlapping segments into street segments...")

        # Calculate bearings
        self.segments['bearing'] = self.segments.geometry.apply(self._calculate_bearing)

        # Sort by length (process longer segments first)
        segments = self.segments.sort_values('length', ascending=False).reset_index(drop=True)

        # Track which segments have been merged
        merged_into = {}  # segment_idx -> street_segment_idx
        street_segments = []

        for idx in tqdm(range(len(segments)), desc="Merging segments"):
            if idx in merged_into:
                continue

            segment = segments.iloc[idx]
            buffer = segment.geometry.buffer(buffer_distance)

            # Find overlapping segments
            matching = [idx]
            trip_counts = [segment['trip_count']]

            for other_idx in range(idx + 1, len(segments)):
                if other_idx in merged_into:
                    continue

                other = segments.iloc[other_idx]

                # Check spatial overlap
                if not buffer.intersects(other.geometry):
                    continue

                # Check bearing similarity
                bearing_diff = abs(segment['bearing'] - other['bearing']) % 360
                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff

                if bearing_diff <= direction_tolerance:
                    matching.append(other_idx)
                    trip_counts.append(other['trip_count'])
                    merged_into[other_idx] = len(street_segments)

            # Create merged street segment
            street_segments.append({
                'geometry': segment.geometry,  # Use geometry of longest segment
                'vehicle': segment['vehicle'],
                'trip_count': sum(trip_counts),  # Sum all trips using this street
                'route_count': len(matching),  # How many routes use this street
                'length': segment.geometry.length,
                'bearing': segment['bearing']
            })

        logger.info(f"Merged into {len(street_segments)} unique street segments")

        merged_gdf = gpd.GeoDataFrame(street_segments, crs=self.segments.crs)
        return merged_gdf

    @staticmethod
    def _calculate_bearing(line: LineString) -> float:
        """Calculate bearing of a line in degrees"""
        if len(line.coords) < 2:
            return 0.0

        start, end = line.coords[0], line.coords[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 360


if __name__ == "__main__":
    from pathlib import Path

    # Test the segmenter
    PROJECT_ROOT = Path("D:/QGIS/gtfs_schedules_city")
    ROUTE_LINES = PROJECT_ROOT / "data/processed/route_lines_continuous.shp"
    OUTPUT = PROJECT_ROOT / "data/processed/street_segments.shp"

    logger.info("Loading route lines...")
    routes = gpd.read_file(ROUTE_LINES)

    # Create segmenter
    segmenter = LineSegmenter(routes)

    # Find intersections
    intersections = segmenter.find_intersections(tolerance=1.0)

    # Split at intersections
    segments = segmenter.split_at_intersections()

    # Merge overlapping segments
    street_segments = segmenter.merge_overlapping_segments(
        buffer_distance=4.0,
        direction_tolerance=10.0
    )

    # Save
    logger.info(f"Saving to {OUTPUT}")
    street_segments.to_file(OUTPUT)

    logger.info("Done!")
    logger.info(f"Original routes: {len(routes)}")
    logger.info(f"Split segments: {len(segments)}")
    logger.info(f"Street segments: {len(street_segments)}")
    logger.info(f"Max trips on any segment: {street_segments['trip_count'].max()}")