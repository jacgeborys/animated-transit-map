"""
Complete GTFS Processing Pipeline for Warsaw Transit Animation

This script runs the full pipeline from GTFS download to street segments.
Use this as your main entry point.
"""
import sys
from pathlib import Path
import logging
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.gtfs_downloader import GTFSDownloader
from core.gtfs_parser import GTFSParser
from core.route_line_builder import RouteLineBuilder
from visualization.segmenter import LineSegmenter
from config import ANALYSIS_DATE, PROCESSED_DIR
import geopandas as gpd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(skip_download=False, skip_segments=False):
    """
    Run the complete pipeline

    Args:
        skip_download: Skip GTFS download and use existing data
        skip_segments: Skip segment creation (use existing street_segments.shp)
    """

    logger.info("="*70)
    logger.info("WARSAW TRANSIT PROCESSING PIPELINE")
    logger.info("="*70)

    # ========================================================================
    # STEP 1: Download GTFS Data
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Download GTFS Data")
    logger.info("="*70)

    downloader = GTFSDownloader()

    if skip_download:
        logger.info("Skipping download, using existing data")
        data_dir = downloader.get_latest_data_dir()
    else:
        logger.info("Downloading latest GTFS data...")
        data_dir = downloader.download_and_extract()

    logger.info(f"Using GTFS data from: {data_dir}")

    # ========================================================================
    # STEP 2: Build Continuous Route Lines
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Build Continuous Route Lines")
    logger.info("="*70)

    route_builder = RouteLineBuilder(data_dir)
    route_builder.load_data(ANALYSIS_DATE)
    route_lines = route_builder.build_route_lines()

    route_lines_path = PROCESSED_DIR / "route_lines_continuous.shp"
    logger.info(f"Saving route lines to {route_lines_path}")
    route_lines_path.parent.mkdir(parents=True, exist_ok=True)
    route_lines.to_file(route_lines_path)

    logger.info(f"Created {len(route_lines)} continuous route lines")

    # ========================================================================
    # STEP 3: Generate Schedule for Animation
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Generate Schedule for Animation")
    logger.info("="*70)

    schedule_path = PROCESSED_DIR / "schedule_for_animation.csv"
    route_builder.save_schedule(schedule_path)
    logger.info(f"Schedule saved to {schedule_path}")

    # ========================================================================
    # STEP 4: Create Street Segments (Intersection-Based)
    # ========================================================================
    if not skip_segments:
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Create Street Segments (Intersection to Intersection)")
        logger.info("="*70)

        logger.info("Creating segmenter...")
        segmenter = LineSegmenter(route_lines)

        logger.info("Finding intersection points...")
        intersections = segmenter.find_intersections(tolerance=1.0)

        if len(intersections.geoms) == 0:
            logger.warning("No intersections found! Using original routes as segments.")
            street_segments = route_lines
        else:
            logger.info("Splitting routes at intersections...")
            segments = segmenter.split_at_intersections()

            logger.info("Merging overlapping segments...")
            street_segments = segmenter.merge_overlapping_segments(
                buffer_distance=15.0,
                direction_tolerance=15.0
            )

        # Add route_ids as list for animation
        if 'route_id' in street_segments.columns:
            street_segments['route_ids'] = street_segments['route_id'].apply(
                lambda x: [str(x)] if isinstance(x, (str, int, float)) else
                         [str(r) for r in str(x).split(',')] if isinstance(x, str) else []
            )

        # Calculate length in km
        if 'length_km' not in street_segments.columns:
            street_segments['length_km'] = street_segments.geometry.length / 1000.0

        # Save street segments
        segments_path = PROCESSED_DIR / "street_segments.shp"
        logger.info(f"Saving street segments to {segments_path}")
        street_segments.to_file(segments_path)

        logger.info(f"Created {len(street_segments)} street segments")

        if 'route_count' in street_segments.columns:
            logger.info(f"\nSegment statistics:")
            logger.info(f"  Average length: {street_segments['length'].mean():.0f}m")
            logger.info(f"  Max routes on one segment: {street_segments['route_count'].max()}")
    else:
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Skipping segment creation (using existing)")
        logger.info("="*70)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info("\nGenerated files:")
    logger.info(f"  1. Route lines: {route_lines_path}")
    logger.info(f"  2. Schedule: {schedule_path}")

    segments_path = PROCESSED_DIR / "street_segments.shp"
    if segments_path.exists():
        logger.info(f"  3. Street segments: {segments_path}")

    logger.info("\nNext steps:")
    logger.info("  - For animation: python animate/animate_full_density.py")
    logger.info("  - For static maps: Load street_segments.shp in QGIS")
    logger.info("  - For capacity analysis: python x_capacity_calculator.py")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete GTFS processing pipeline"
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip GTFS download and use existing data'
    )
    parser.add_argument(
        '--skip-segments',
        action='store_true',
        help='Skip segment creation (use existing street_segments.shp)'
    )

    args = parser.parse_args()

    sys.exit(main(
        skip_download=args.skip_download,
        skip_segments=args.skip_segments
    ))
