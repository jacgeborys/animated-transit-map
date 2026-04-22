"""
Generate street segments (intersection to intersection) for animation and mapping

This script uses the segmenter to create logical street segments from continuous routes.
Output can be used for both static maps and animation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.segmenter import LineSegmenter
import geopandas as gpd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent
ROUTE_LINES = PROJECT_ROOT / "data" / "processed" / "route_lines_continuous.shp"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "street_segments.shp"

def main():
    """Generate street segments from route lines"""

    if not ROUTE_LINES.exists():
        logger.error(f"Route lines not found: {ROUTE_LINES}")
        logger.error("Run: python core/5_route_line_builder.py first")
        return 1

    logger.info("="*70)
    logger.info("Street Segment Generator")
    logger.info("="*70)

    logger.info(f"Loading route lines from {ROUTE_LINES}...")
    routes = gpd.read_file(ROUTE_LINES)
    logger.info(f"Loaded {len(routes)} route shapes")

    # Create segmenter
    segmenter = LineSegmenter(routes)

    # Find intersections
    logger.info("\nStep 1: Finding intersection points...")
    intersections = segmenter.find_intersections(tolerance=1.0)

    if len(intersections.geoms) == 0:
        logger.warning("No intersections found! Using original routes.")
        street_segments = routes
    else:
        # Split at intersections
        logger.info("\nStep 2: Splitting routes at intersections...")
        segments = segmenter.split_at_intersections()

        # Merge overlapping segments
        logger.info("\nStep 3: Merging overlapping segments...")
        street_segments = segmenter.merge_overlapping_segments(
            buffer_distance=15.0,  # 15m tolerance for overlap
            direction_tolerance=15.0  # 15 degree tolerance
        )

    # Add route_ids as list for animation compatibility
    if 'route_id' in street_segments.columns:
        logger.info("\nBuilding route_ids list...")
        street_segments['route_ids'] = street_segments['route_id'].apply(
            lambda x: [str(x)] if isinstance(x, (str, int, float)) else
                     [str(r) for r in str(x).split(',')] if isinstance(x, str) else []
        )

    # Calculate length in km
    if 'length_km' not in street_segments.columns:
        street_segments['length_km'] = street_segments.geometry.length / 1000.0

    # Save
    logger.info(f"\nSaving to {OUTPUT}...")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    street_segments.to_file(OUTPUT)

    # Statistics
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    logger.info(f"Original routes: {len(routes)}")
    logger.info(f"Street segments: {len(street_segments)}")
    logger.info(f"Average segment length: {street_segments['length'].mean():.0f}m")

    if 'route_count' in street_segments.columns:
        logger.info(f"\nRoute overlap distribution:")
        overlap = street_segments['route_count'].value_counts().sort_index()
        for count, freq in overlap.items():
            logger.info(f"  {freq} segments used by {count} route(s)")

    if 'trip_count' in street_segments.columns:
        logger.info(f"\nTrip count statistics:")
        logger.info(f"  Min: {street_segments['trip_count'].min()}")
        logger.info(f"  Max: {street_segments['trip_count'].max()}")
        logger.info(f"  Mean: {street_segments['trip_count'].mean():.0f}")

    logger.info(f"\nOutput saved to: {OUTPUT}")
    logger.info("\nYou can now use this for:")
    logger.info("  - Static maps in QGIS (style by trip_count)")
    logger.info("  - Animation (set USE_STREET_SEGMENTS=True)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
