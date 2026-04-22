"""
Module for aggregating parallel transit segments into unified street-level transit frequencies
"""
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
from pathlib import Path
from typing import Set
from tqdm import tqdm
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BUFFER_DISTANCE,
    DIRECTION_TOLERANCE,
    SEGMENT_LENGTH,
    POLAND_CRS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SegmentAggregator:
    """Aggregate parallel transit segments to calculate combined frequencies"""
    
    def __init__(self, segments_gdf: gpd.GeoDataFrame):
        self.segments_gdf = segments_gdf.copy()
        self.aggregated = {}
        
    @staticmethod
    def split_linestring(linestring: LineString, segment_length: float) -> list:
        """
        Split a LineString into smaller segments of approximately equal length
        
        Args:
            linestring: LineString to split
            segment_length: Target length for each segment
            
        Returns:
            List of LineString segments
        """
        total_length = linestring.length
        num_segments = max(1, int(round(total_length / segment_length)))
        
        points = [
            linestring.interpolate(float(n) / num_segments, normalized=True)
            for n in range(num_segments + 1)
        ]
        
        return [
            LineString([points[n], points[n + 1]])
            for n in range(num_segments)
        ]
    
    def prepare_segments(self) -> gpd.GeoDataFrame:
        """
        Prepare segments for aggregation by splitting long segments
        
        Returns:
            GeoDataFrame with split segments
        """
        logger.info("Preparing segments for aggregation...")
        
        # Filter valid segments
        self.segments_gdf = self.segments_gdf[
            (self.segments_gdf['trip_count'] > 0) &
            (self.segments_gdf['vehicle'] != 'Unknown')
        ].copy()
        
        logger.info(f"Starting with {len(self.segments_gdf)} valid segments")
        
        # Split long segments
        logger.info(f"Splitting segments longer than {SEGMENT_LENGTH}m...")
        split_segments = []
        
        for _, row in tqdm(self.segments_gdf.iterrows(), 
                          total=len(self.segments_gdf),
                          desc="Splitting segments"):
            
            if row.geometry.length > SEGMENT_LENGTH:
                # Split into smaller segments
                for segment in self.split_linestring(row.geometry, SEGMENT_LENGTH):
                    new_row = row.copy()
                    new_row['geometry'] = segment
                    new_row['length'] = segment.length
                    new_row['bearing'] = self._recalculate_bearing(segment)
                    split_segments.append(new_row)
            else:
                split_segments.append(row)
        
        split_gdf = gpd.GeoDataFrame(split_segments, crs=POLAND_CRS)
        
        logger.info(
            f"After splitting: {len(split_gdf)} segments "
            f"(from {len(self.segments_gdf)} original)"
        )
        
        return split_gdf
    
    @staticmethod
    def _recalculate_bearing(line: LineString) -> float:
        """Recalculate bearing for a line segment"""
        if len(line.coords) < 2:
            return None
        start, end = line.coords[0], line.coords[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return np.degrees(np.arctan2(dy, dx)) % 360
    
    @staticmethod
    def bearing_difference(bearing1: float, bearing2: float) -> float:
        """
        Calculate minimum angle difference between two bearings
        
        Args:
            bearing1: First bearing in degrees (0-360)
            bearing2: Second bearing in degrees (0-360)
            
        Returns:
            Minimum angle difference (0-180)
        """
        diff = abs(bearing1 - bearing2) % 360
        return min(diff, 360 - diff)
    
    def aggregate_by_vehicle(self, vehicle_type: str, segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Aggregate segments for a specific vehicle type
        
        Args:
            vehicle_type: Type of vehicle to process
            segments: GeoDataFrame of segments
            
        Returns:
            Aggregated GeoDataFrame
        """
        logger.info(f"Aggregating {vehicle_type} segments...")
        
        # Filter by vehicle type
        vehicle_segments = segments[segments['vehicle'] == vehicle_type].copy()
        
        if len(vehicle_segments) == 0:
            logger.warning(f"No segments found for {vehicle_type}")
            return gpd.GeoDataFrame(columns=segments.columns, crs=POLAND_CRS)
        
        # Sort by length (descending) to process longer segments first
        vehicle_segments = vehicle_segments.sort_values('length', ascending=False)
        vehicle_segments = vehicle_segments.reset_index(drop=True)
        
        # Initialize trip_sum column
        vehicle_segments['trip_sum'] = vehicle_segments['trip_count']
        
        # Create spatial index for efficient querying
        spatial_index = vehicle_segments.sindex
        
        # Track processed shape_ids for each segment
        processed_shapes: dict = {}
        trip_sum_updates = {}
        
        logger.info(f"Processing {len(vehicle_segments)} {vehicle_type} segments...")
        
        for idx in tqdm(range(len(vehicle_segments)), desc=f"Aggregating {vehicle_type}"):
            if idx in processed_shapes:
                continue
            
            current_segment = vehicle_segments.iloc[idx]
            
            # Skip if already processed or no trips
            if current_segment['trip_sum'] == 0:
                continue
            
            # Create buffer for spatial query
            buffer = current_segment.geometry.buffer(BUFFER_DISTANCE)
            
            # Find potentially overlapping segments
            possible_matches = list(spatial_index.intersection(buffer.bounds))
            
            # Initialize set of processed shape_ids for this segment
            processed_shapes[idx] = {current_segment['shape_id']}
            
            # Check each potential match
            for match_idx in possible_matches:
                if match_idx == idx:
                    continue
                
                match_segment = vehicle_segments.iloc[match_idx]
                
                # Skip if already counted or same shape or no trips
                if (match_idx in processed_shapes or
                    match_segment['trip_sum'] == 0 or
                    match_segment['shape_id'] in processed_shapes[idx]):
                    continue
                
                # Check if bearings are similar (same direction)
                bearing_diff = self.bearing_difference(
                    current_segment['bearing'],
                    match_segment['bearing']
                )
                
                if bearing_diff <= DIRECTION_TOLERANCE:
                    # Add this segment's trips to current segment
                    processed_shapes[idx].add(match_segment['shape_id'])
                    trip_sum_updates[idx] = trip_sum_updates.get(
                        idx, current_segment['trip_sum']
                    ) + match_segment['trip_sum']
                    trip_sum_updates[match_idx] = 0  # Mark as processed
        
        # Apply updates
        for idx, trip_sum in trip_sum_updates.items():
            if idx < len(vehicle_segments):
                vehicle_segments.at[idx, 'trip_sum'] = trip_sum
        
        # Keep only segments with trips
        aggregated = vehicle_segments[vehicle_segments['trip_sum'] > 0].copy()
        
        logger.info(
            f"Aggregated {vehicle_type}: "
            f"{len(vehicle_segments)} -> {len(aggregated)} segments"
        )
        
        return aggregated
    
    def aggregate_all(self) -> dict:
        """
        Aggregate segments for all vehicle types
        
        Returns:
            Dictionary mapping vehicle type to aggregated GeoDataFrame
        """
        # Prepare segments
        prepared = self.prepare_segments()
        
        # Get unique vehicle types
        vehicle_types = prepared['vehicle'].unique()
        logger.info(f"Processing vehicle types: {vehicle_types.tolist()}")
        
        # Aggregate each vehicle type
        results = {}
        for vehicle in vehicle_types:
            if vehicle != 'Unknown':
                aggregated = self.aggregate_by_vehicle(vehicle, prepared)
                results[vehicle] = aggregated
        
        self.aggregated = results
        return results
    
    def save_all(self, output_dir: Path):
        """
        Save aggregated segments for all vehicle types
        
        Args:
            output_dir: Directory to save shapefiles
        """
        if not self.aggregated:
            raise ValueError("No aggregated data. Run aggregate_all() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for vehicle, gdf in self.aggregated.items():
            output_path = output_dir / f"aggregated_{vehicle.lower()}.shp"
            logger.info(f"Saving {vehicle} segments to {output_path}")
            gdf.to_file(output_path)
            logger.info(f"Saved {len(gdf)} {vehicle} segments")


if __name__ == "__main__":
    from config import PROCESSED_DIR, OUTPUT_DIR
    
    # Load segments
    segments_path = PROCESSED_DIR / "individual_segments.shp"
    logger.info(f"Loading segments from {segments_path}")
    segments = gpd.read_file(segments_path)
    
    # Aggregate
    aggregator = SegmentAggregator(segments)
    results = aggregator.aggregate_all()
    
    # Save
    aggregator.save_all(OUTPUT_DIR)
    
    print("\nAggregation complete!")
    for vehicle, gdf in results.items():
        print(f"{vehicle}: {len(gdf)} segments, {gdf['trip_sum'].sum():.0f} total trips")
