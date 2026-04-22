"""
Route line builder for animation

Creates continuous, unbroken transit lines by route for animation purposes.
Unlike the aggregated segments, these preserve the complete route geometry.
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import POLAND_CRS, WGS84
from core.gtfs_parser import GTFSParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RouteLineBuilder:
    """Build continuous lines for each route (for animation)"""
    
    def __init__(self, gtfs_data_dir: Path):
        self.data_dir = Path(gtfs_data_dir)
        self.parser = GTFSParser(self.data_dir)
        self.route_lines = None
        self.stops_gdf = None
        
    def load_data(self, target_date: str):
        """
        Load GTFS data including shapes, routes, trips, stops
        
        Args:
            target_date: Date in YYYYMMDD format
        """
        logger.info("Loading GTFS data for route lines...")
        
        # Load basic files
        self.parser.load_data()

        # Filter by date and time
        service_id = self.parser.filter_by_date(target_date)
        self.parser.prepare_trips(service_id)
        self.parser.expand_metro_frequencies(target_date)  # inject synthetic metro departures
        # self.parser.filter_by_time()  # COMMENTED OUT for full day animation
        
        # Load stops
        stops_df = pd.read_csv(self.data_dir / "stops.txt")
        
        # Create stops GeoDataFrame
        self.stops_gdf = gpd.GeoDataFrame(
            stops_df,
            geometry=[Point(lon, lat) for lon, lat in zip(stops_df.stop_lon, stops_df.stop_lat)],
            crs=WGS84
        )
        self.stops_gdf = self.stops_gdf.to_crs(POLAND_CRS)
        
        logger.info(f"Loaded {len(self.stops_gdf)} stops")
    
    def build_route_lines(self) -> gpd.GeoDataFrame:
        """
        Build continuous lines for each unique route
        
        Returns:
            GeoDataFrame with one continuous line per route
        """
        logger.info("Building continuous route lines...")
        
        # Get shapes data
        shapes_df = self.parser.shapes
        
        # Create points
        points_gdf = gpd.GeoDataFrame(
            shapes_df,
            geometry=[
                Point(lon, lat) 
                for lon, lat in zip(shapes_df.shape_pt_lon, shapes_df.shape_pt_lat)
            ],
            crs=WGS84
        )
        points_gdf = points_gdf.to_crs(POLAND_CRS)
        
        # Sort by shape and sequence
        points_gdf = points_gdf.sort_values(['shape_id', 'shape_pt_sequence'])
        
        # Build lines per shape_id
        route_lines = []
        
        for shape_id, group in points_gdf.groupby('shape_id'):
            # Create continuous line from all points
            points = list(group.geometry)
            
            if len(points) < 2:
                continue
            
            line = LineString(points)
            
            # Get trip info for this shape
            trips = self.parser.trips[self.parser.trips['shape_id'] == shape_id]
            
            if len(trips) == 0:
                continue
            
            # Get route info
            route_id = trips['route_id'].iloc[0]
            vehicle = trips['vehicle'].iloc[0]
            trip_count = len(trips)
            
            route_lines.append({
                'geometry': line,
                'shape_id': shape_id,
                'route_id': route_id,
                'vehicle': vehicle,
                'trip_count': trip_count,
                'length': line.length
            })
        
        # Create GeoDataFrame
        route_lines_gdf = gpd.GeoDataFrame(route_lines, crs=POLAND_CRS)
        
        logger.info(f"Created {len(route_lines_gdf)} continuous route lines")
        logger.info(f"Vehicle breakdown: {route_lines_gdf['vehicle'].value_counts().to_dict()}")
        
        self.route_lines = route_lines_gdf
        return route_lines_gdf
    
    def save_route_lines(self, output_path: Path):
        """
        Save route lines to shapefile
        
        Args:
            output_path: Where to save the shapefile
        """
        if self.route_lines is None:
            raise ValueError("No route lines to save. Run build_route_lines() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving route lines to {output_path}")
        self.route_lines.to_file(output_path)
        
    def get_route_schedule(self, route_id: str = None) -> pd.DataFrame:
        """
        Get schedule data for animation (when vehicles depart each stop)
        
        Args:
            route_id: Optional - filter to specific route
            
        Returns:
            DataFrame with trip_id, stop_id, arrival_time, departure_time
        """
        logger.info("Extracting schedule data...")
        
        # Get valid trips
        valid_trip_ids = self.parser.stop_times['trip_id'].unique()
        trips = self.parser.trips[self.parser.trips['trip_id'].isin(valid_trip_ids)]
        
        if route_id:
            trips = trips[trips['route_id'] == route_id]
        
        # Merge stop times with trip info
        schedule = pd.merge(
            self.parser.stop_times[['trip_id', 'stop_id', 'arrival_time', 'departure_time', 'stop_sequence']],
            trips[['trip_id', 'route_id', 'shape_id', 'vehicle']],
            on='trip_id'
        )
        
        # The arrival_time and departure_time are already strings from the CSV
        # We just rename them for clarity
        schedule['arrival_time_str'] = schedule['arrival_time']
        schedule['departure_time_str'] = schedule['departure_time']

        logger.info(f"Generated schedule for {len(schedule)} stop events")

        return schedule

    def save_schedule(self, output_path: Path, route_id: str = None):
        """
        Save schedule data to CSV

        Args:
            output_path: Where to save the CSV
            route_id: Optional - filter to specific route
        """
        schedule = self.get_route_schedule(route_id)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving schedule to {output_path}")

        schedule.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse
    from config import ANALYSIS_DATE, PROCESSED_DIR
    from gtfs_downloader import GTFSDownloader

    parser = argparse.ArgumentParser(
        description="Build continuous route lines for animation"
    )
    parser.add_argument(
        '--date',
        type=str,
        default=ANALYSIS_DATE,
        help='Analysis date in YYYYMMDD format'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROCESSED_DIR,
        help='Output directory'
    )

    args = parser.parse_args()

    # Get GTFS data
    downloader = GTFSDownloader()
    data_dir = downloader.get_latest_data_dir()

    # Build route lines
    builder = RouteLineBuilder(data_dir)
    builder.load_data(args.date)
    route_lines = builder.build_route_lines()

    # Save outputs
    builder.save_route_lines(args.output_dir / "route_lines_continuous.shp")
    builder.save_schedule(args.output_dir / "schedule_for_animation.csv")

    logger.info("\nRoute lines built successfully!")
    logger.info(f"Continuous lines: {len(route_lines)}")
    logger.info(f"Total length: {route_lines['length'].sum()/1000:.1f} km")