"""
Module for converting GTFS points to line segments with geometry processing
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WGS84, POLAND_CRS, MIN_SEGMENT_LENGTH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeometryProcessor:
    """Convert GTFS shape points to line segments with metadata"""
    
    def __init__(self, shapes_df: pd.DataFrame):
        self.shapes_df = shapes_df
        self.segments_gdf = None
        
    @staticmethod
    def calculate_bearing(line: LineString) -> float:
        """
        Calculate bearing (direction) of a line segment in degrees
        
        Args:
            line: LineString geometry
            
        Returns:
            Bearing in degrees (0-360), or None if invalid
        """
        if not isinstance(line, LineString) or len(line.coords) < 2:
            return None
        
        start, end = line.coords[0], line.coords[-1]
        
        # Calculate angle in degrees
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalize to 0-360
        return angle % 360
    
    def create_segments_from_group(self, group: gpd.GeoDataFrame) -> List[Dict]:
        """
        Create line segments from a group of ordered points
        
        Args:
            group: GeoDataFrame with points for a single shape_id
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        points = list(group.geometry)
        
        if len(points) < 2:
            return segments
        
        # Get metadata from first point (same for all points in group)
        shape_id = group['shape_id'].iloc[0]
        trip_count = group['trip_count'].iloc[0]
        vehicle = group['vehicle'].iloc[0]
        
        # Create segments between consecutive points
        for i in range(len(points) - 1):
            segment = LineString([points[i], points[i + 1]])
            
            segments.append({
                'geometry': segment,
                'shape_id': shape_id,
                'trip_count': trip_count,
                'vehicle': vehicle,
                'length': segment.length,
                'bearing': self.calculate_bearing(segment)
            })
        
        return segments
    
    def points_to_segments(self) -> gpd.GeoDataFrame:
        """
        Convert shape points to line segments
        
        Returns:
            GeoDataFrame of line segments with attributes
        """
        logger.info("Creating point geometries...")
        
        # Create GeoDataFrame from points
        gdf = gpd.GeoDataFrame(
            self.shapes_df,
            geometry=[
                Point(lon, lat) 
                for lon, lat in zip(
                    self.shapes_df['shape_pt_lon'],
                    self.shapes_df['shape_pt_lat']
                )
            ],
            crs=WGS84
        )
        
        # Reproject to metric CRS for accurate distance calculations
        logger.info(f"Reprojecting to {POLAND_CRS}...")
        gdf = gdf.to_crs(POLAND_CRS)
        
        # Sort by shape and sequence
        logger.info("Sorting and grouping points...")
        gdf = gdf.sort_values(['shape_id', 'shape_pt_sequence'])
        
        # Create segments for each shape
        logger.info("Creating line segments...")
        all_segments = []
        
        for shape_id, group in gdf.groupby('shape_id'):
            segments = self.create_segments_from_group(group)
            all_segments.extend(segments)
        
        # Create GeoDataFrame from segments
        segments_gdf = gpd.GeoDataFrame(
            all_segments,
            geometry='geometry',
            crs=POLAND_CRS
        )
        
        # Filter out very short segments
        initial_count = len(segments_gdf)
        segments_gdf = segments_gdf[segments_gdf['length'] > MIN_SEGMENT_LENGTH]
        
        logger.info(
            f"Created {len(segments_gdf)} segments "
            f"(filtered {initial_count - len(segments_gdf)} segments < {MIN_SEGMENT_LENGTH}m)"
        )
        logger.info(f"Unique shapes: {segments_gdf['shape_id'].nunique()}")
        logger.info(f"Vehicle types: {segments_gdf['vehicle'].unique().tolist()}")
        
        self.segments_gdf = segments_gdf
        return segments_gdf
    
    def save(self, output_path: Path):
        """
        Save segments to shapefile
        
        Args:
            output_path: Path to save shapefile
        """
        if self.segments_gdf is None:
            raise ValueError("No segments to save. Run points_to_segments() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving segments to {output_path}")
        self.segments_gdf.to_file(output_path)
        logger.info("Segments saved successfully")


if __name__ == "__main__":
    from gtfs_downloader import GTFSDownloader
    from gtfs_parser import GTFSParser
    from config import ANALYSIS_DATE, PROCESSED_DIR
    
    # Get and parse data
    downloader = GTFSDownloader()
    data_dir = downloader.get_latest_data_dir()
    
    parser = GTFSParser(data_dir)
    shapes_with_counts = parser.process(ANALYSIS_DATE)
    
    # Convert to segments
    processor = GeometryProcessor(shapes_with_counts)
    segments = processor.points_to_segments()
    
    # Save
    output_path = PROCESSED_DIR / "individual_segments.shp"
    processor.save(output_path)
    
    print(f"\nSegments saved to: {output_path}")
