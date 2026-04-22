"""
Module for adding capacity estimates to aggregated transit segments

This script reads already-aggregated shapefiles and adds capacity calculations
without needing to rerun the entire aggregation process.
"""
import geopandas as gpd
from pathlib import Path
import logging

from config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CapacityCalculator:
    """Add vehicle capacity estimates to transit segments"""
    
    # Base capacity estimates for Warsaw fleet (passengers: seated + standing)
    VEHICLE_CAPACITY = {
        'Tram': {
            'default': 200,  # Average Warsaw tram capacity
            'modern_low_floor': 240,  # Pesa 120Na, 128N (newer models)
            'standard': 180,  # Konstal 105Na, older models
            'notes': 'Mix of old Konstal and modern Pesa trams'
        },
        'Bus': {
            'default': 100,  # Weighted average
            'articulated': 120,  # 18m buses (Solaris Urbino 18)
            'standard': 90,  # 12m buses (Solaris Urbino 12)
            'midi': 60,  # 9-10m buses (Mercedes Conecto)
            'notes': 'Mix of standard and articulated, mostly Solaris fleet'
        },
        'Train': {
            'default': 600,  # Average commuter train
            'SKM': 800,  # Szybka Kolej Miejska (6-car sets)
            'WKD': 400,  # Warszawska Kolej Dojazdowa (2-3 car sets)
            'notes': 'Primarily SKM and WKD services'
        }
    }
    
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        
    def estimate_capacity(self, vehicle_type: str, route_id: str = None) -> int:
        """
        Estimate vehicle capacity based on type and route characteristics
        
        Args:
            vehicle_type: Type of vehicle (Bus, Tram, Train)
            route_id: Optional route identifier for route-specific adjustments
            
        Returns:
            Estimated passenger capacity (seated + standing)
        """
        base_capacity = self.VEHICLE_CAPACITY.get(vehicle_type, {}).get('default', 100)
        
        # Route-specific adjustments for buses
        if vehicle_type == 'Bus' and route_id:
            route_id = str(route_id)
            
            # Express buses (100-199 series, 500-599 series) - usually articulated
            if route_id.startswith(('1', '5')) and len(route_id) == 3:
                return self.VEHICLE_CAPACITY['Bus']['articulated']
            
            # Night buses (N-series) - often standard buses
            elif route_id.startswith('N'):
                return self.VEHICLE_CAPACITY['Bus']['standard']
            
            # Zone/suburban buses (700-899 series) - often standard
            elif route_id.startswith(('7', '8')):
                return self.VEHICLE_CAPACITY['Bus']['standard']
            
            # High-frequency core routes (single/double digits) - mix, use default
            # Local routes (200-699) - mix of standard and articulated, use default
        
        # Tram capacity - Warsaw is upgrading fleet, use slightly higher estimate
        # for higher-frequency routes (assumption: newer trams on busy routes)
        if vehicle_type == 'Tram':
            # Could add route-specific logic here if needed
            # For now, use optimistic default as Warsaw has many modern trams
            return self.VEHICLE_CAPACITY['Tram']['default']
        
        return base_capacity
    
    def add_capacity_to_shapefile(self, shapefile_path: Path) -> gpd.GeoDataFrame:
        """
        Add capacity calculations to an existing shapefile
        
        Args:
            shapefile_path: Path to aggregated shapefile
            
        Returns:
            GeoDataFrame with capacity columns added
        """
        logger.info(f"Processing {shapefile_path.name}...")
        
        # Read existing shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Get vehicle type from filename or data
        vehicle_type = gdf['vehicle'].iloc[0] if 'vehicle' in gdf.columns else None
        
        if not vehicle_type:
            logger.warning(f"No vehicle type found in {shapefile_path.name}")
            return gdf
        
        # Calculate capacity per trip
        # NOTE: Column names limited to 10 chars for shapefile compatibility
        logger.info(f"Estimating capacity for {vehicle_type} vehicles...")

        if 'route_id' in gdf.columns:
            gdf['veh_cap'] = gdf['route_id'].apply(
                lambda x: self.estimate_capacity(vehicle_type, x)
            )
        else:
            # No route_id available, use default for vehicle type
            default_capacity = self.estimate_capacity(vehicle_type)
            gdf['veh_cap'] = default_capacity
            logger.info(f"Using default capacity: {default_capacity} passengers")

        # Calculate total capacity (trips × capacity per vehicle)
        gdf['total_cap'] = gdf['trip_sum'] * gdf['veh_cap']

        # Calculate capacity per hour (peak hours = 10 hours total: 6-11 AM + 14-19 PM)
        gdf['cap_hr'] = gdf['total_cap'] / 10.0

        logger.info(f"Capacity statistics for {vehicle_type}:")
        logger.info(f"  Average capacity per vehicle: {gdf['veh_cap'].mean():.0f} passengers")
        logger.info(f"  Total capacity offered: {gdf['total_cap'].sum():,.0f} passenger-trips")
        logger.info(f"  Average capacity per hour: {gdf['cap_hr'].mean():.0f} passengers/hour")

        return gdf

    def process_all(self, save_with_suffix: str = "_capacity"):
        """
        Process all aggregated shapefiles in output directory

        Args:
            save_with_suffix: Suffix to add to output filenames (or empty string to overwrite)
        """
        logger.info("="*70)
        logger.info("Adding Capacity Calculations to Aggregated Segments")
        logger.info("="*70)

        # Find all aggregated shapefiles
        shapefiles = list(self.output_dir.glob("aggregated_*.shp"))

        if not shapefiles:
            logger.error(f"No aggregated shapefiles found in {self.output_dir}")
            return

        logger.info(f"Found {len(shapefiles)} shapefiles to process")

        results = {}

        for shapefile in shapefiles:
            # Add capacity data
            gdf = self.add_capacity_to_shapefile(shapefile)

            # Save to new file or overwrite
            if save_with_suffix:
                # Save with suffix (e.g., aggregated_bus_capacity.shp)
                output_name = shapefile.stem + save_with_suffix + shapefile.suffix
                output_path = self.output_dir / output_name
            else:
                # Overwrite original
                output_path = shapefile

            logger.info(f"Saving to {output_path.name}...")
            gdf.to_file(output_path)

            results[shapefile.stem] = {
                'segments': len(gdf),
                'total_capacity': gdf['total_cap'].sum(),
                'avg_capacity_per_hour': gdf['cap_hr'].mean()
            }

        # Summary
        logger.info("\n" + "="*70)
        logger.info("CAPACITY CALCULATION COMPLETE")
        logger.info("="*70)

        for name, stats in results.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Segments: {stats['segments']:,}")
            logger.info(f"  Total capacity: {stats['total_capacity']:,.0f} passenger-trips")
            logger.info(f"  Avg capacity/hour: {stats['avg_capacity_per_hour']:,.0f} passengers/hour")

        logger.info(f"\nNew shapefiles saved with '{save_with_suffix}' suffix")
        logger.info("\nNew attributes added to shapefiles:")
        logger.info("  - veh_cap: Estimated vehicle capacity (passengers)")
        logger.info("  - total_cap: Total capacity offered (trip_sum × veh_cap)")
        logger.info("  - cap_hr: Average capacity per hour during peak periods")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add capacity estimates to aggregated transit segments"
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite original files instead of creating new ones with _capacity suffix'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help='Directory containing aggregated shapefiles'
    )

    args = parser.parse_args()

    calculator = CapacityCalculator(args.output_dir)

    if args.overwrite:
        calculator.process_all(save_with_suffix="")
    else:
        calculator.process_all(save_with_suffix="_capacity")