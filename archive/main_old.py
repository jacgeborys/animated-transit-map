"""
Main pipeline for Warsaw GTFS transit frequency analysis

This script:
1. Downloads latest GTFS data
2. Parses schedule data and filters by date/time
3. Converts points to line segments
4. Aggregates parallel segments to calculate transit frequencies
"""
import argparse
import logging
from pathlib import Path
from datetime import datetime

from config import (
    ANALYSIS_DATE,
    PROCESSED_DIR,
    OUTPUT_DIR,
    PEAK_HOURS
)
from core.gtfs_downloader import GTFSDownloader
from core.gtfs_parser import GTFSParser
# Files 3 and 4 archived - using new simplified pipeline
# Use run_pipeline.py instead for the new workflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(
    target_date: str = ANALYSIS_DATE,
    download_fresh: bool = False,
    skip_download: bool = False
):
    """
    Run the complete GTFS processing pipeline
    
    Args:
        target_date: Date to analyze in YYYYMMDD format
        download_fresh: Force download of fresh GTFS data
        skip_download: Skip download and use existing data
    """
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("Warsaw GTFS Transit Frequency Analysis Pipeline")
    logger.info("="*70)
    logger.info(f"Analysis date: {target_date}")
    logger.info(f"Peak hours: {PEAK_HOURS}")
    
    # Step 1: Download/Get GTFS data
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Getting GTFS data")
    logger.info("="*70)
    
    downloader = GTFSDownloader()
    
    if skip_download:
        logger.info("Skipping download, using existing data")
        data_dir = downloader.get_latest_data_dir()
    elif download_fresh:
        logger.info("Downloading fresh GTFS data")
        data_dir = downloader.download_and_extract()
    else:
        logger.info("Using latest existing data (or downloading if none exists)")
        data_dir = downloader.get_latest_data_dir()
    
    # Step 2: Parse and filter schedule data
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Parsing schedule data")
    logger.info("="*70)
    
    parser = GTFSParser(data_dir)
    shapes_with_counts = parser.process(target_date)
    
    logger.info(f"\nParsing complete:")
    logger.info(f"  - Shape points: {len(shapes_with_counts):,}")
    logger.info(f"  - Unique shapes: {shapes_with_counts['shape_id'].nunique():,}")
    logger.info(f"  - Total trips: {shapes_with_counts['trip_count'].sum():,.0f}")
    
    # Step 3: Convert to line segments
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Creating line segments")
    logger.info("="*70)
    
    processor = GeometryProcessor(shapes_with_counts)
    segments = processor.points_to_segments()
    
    segments_path = PROCESSED_DIR / "individual_segments.shp"
    processor.save(segments_path)
    
    logger.info(f"\nSegments created:")
    logger.info(f"  - Total segments: {len(segments):,}")
    logger.info(f"  - Saved to: {segments_path}")
    
    # Step 4: Aggregate parallel segments
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Aggregating parallel segments")
    logger.info("="*70)
    
    aggregator = SegmentAggregator(segments)
    results = aggregator.aggregate_all()
    aggregator.save_all(OUTPUT_DIR)
    
    logger.info(f"\nAggregation complete:")
    for vehicle, gdf in results.items():
        logger.info(
            f"  - {vehicle}: {len(gdf):,} segments, "
            f"{gdf['trip_sum'].sum():,.0f} total trips"
        )
    logger.info(f"  - Saved to: {OUTPUT_DIR}")
    
    # Summary
    elapsed = datetime.now() - start_time
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Total processing time: {elapsed}")
    logger.info(f"\nOutput files:")
    logger.info(f"  - Individual segments: {segments_path}")
    for vehicle in results.keys():
        output_path = OUTPUT_DIR / f"aggregated_{vehicle.lower()}.shp"
        logger.info(f"  - Aggregated {vehicle}: {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Warsaw GTFS data to analyze transit frequencies"
    )
    parser.add_argument(
        '--date',
        type=str,
        default=ANALYSIS_DATE,
        help='Analysis date in YYYYMMDD format (default: from config)'
    )
    parser.add_argument(
        '--download-fresh',
        action='store_true',
        help='Force download of fresh GTFS data'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download and use existing data'
    )
    
    args = parser.parse_args()
    
    main(
        target_date=args.date,
        download_fresh=args.download_fresh,
        skip_download=args.skip_download
    )
