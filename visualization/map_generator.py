"""
Map generator for Warsaw transit frequency visualization

Creates professional cartograms with variable-width lines representing
transit frequency or capacity.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
import geopandas as gpd
import contextily as ctx
from pathlib import Path
import numpy as np
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))

from visualization.config_viz import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransitMapGenerator:
    """Generate beautiful transit frequency maps"""
    
    def __init__(self, data_dir: Path = OUTPUT_DIR, output_dir: Path = MAPS_DIR):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.vehicle_data = {}
        self.fig = None
        self.ax = None
        
    def load_data(self, use_capacity: bool = True):
        """
        Load transit line data
        
        Args:
            use_capacity: Load *_capacity.shp files if True, otherwise regular aggregated files
        """
        logger.info("Loading transit data...")
        
        vehicles = ['bus', 'tram', 'train']
        
        for vehicle in vehicles:
            if use_capacity:
                filename = f"aggregated_{vehicle}_capacity.shp"
            else:
                filename = f"aggregated_{vehicle}.shp"
            
            filepath = self.data_dir / filename
            
            if filepath.exists():
                gdf = gpd.read_file(filepath)
                self.vehicle_data[vehicle] = gdf
                logger.info(f"Loaded {len(gdf)} {vehicle} segments")
            else:
                logger.warning(f"File not found: {filepath}")
        
        if not self.vehicle_data:
            raise ValueError(f"No data files found in {self.data_dir}")
    
    def calculate_line_widths(self, gdf: gpd.GeoDataFrame, vehicle_type: str) -> np.ndarray:
        """
        Calculate line widths based on frequency/capacity
        
        Args:
            gdf: GeoDataFrame with transit segments
            vehicle_type: Type of vehicle for style lookup
            
        Returns:
            Array of line widths in points
        """
        attribute = WIDTH_ATTRIBUTE
        
        if attribute not in gdf.columns:
            logger.warning(f"Attribute '{attribute}' not found, using default widths")
            return np.full(len(gdf), STYLES[vehicle_type]['min_width'])
        
        values = gdf[attribute].values
        
        # Normalize to 0-1 range
        min_val = values.min()
        max_val = values.max()
        
        if max_val == min_val:
            normalized = np.ones_like(values)
        else:
            normalized = (values - min_val) / (max_val - min_val)
        
        # Scale to line width range
        min_width = STYLES[vehicle_type]['min_width']
        max_width = STYLES[vehicle_type]['max_width']
        
        widths = min_width + normalized * (max_width - min_width)
        
        return widths
    
    def setup_figure(self):
        """Setup matplotlib figure and axis"""
        logger.info("Setting up figure...")
        
        width = MAP_SIZE['width']
        height = MAP_SIZE['height']
        dpi = MAP_SIZE['dpi']
        
        self.fig, self.ax = plt.subplots(
            figsize=(width, height),
            dpi=dpi,
            facecolor=BACKGROUND['color']
        )
        
        self.ax.set_facecolor(BACKGROUND['color'])
        self.ax.set_aspect('equal')
        
        # Remove axis spines for cleaner look
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
    
    def add_basemap(self):
        """Add OpenStreetMap basemap background"""
        if not BACKGROUND.get('use_osm_basemap', False):
            return
        
        logger.info("Adding basemap...")
        
        try:
            ctx.add_basemap(
                self.ax,
                crs='EPSG:2180',
                source=ctx.providers.CartoDB.Positron,  # Light, clean basemap
                alpha=0.3,  # Very subtle
                attribution=False
            )
        except Exception as e:
            logger.warning(f"Could not add basemap: {e}")
    
    def plot_vehicle_type(self, vehicle_type: str):
        """
        Plot lines for a specific vehicle type
        
        Args:
            vehicle_type: Type of vehicle to plot (bus, tram, train)
        """
        if vehicle_type not in self.vehicle_data:
            logger.warning(f"No data for {vehicle_type}")
            return
        
        gdf = self.vehicle_data[vehicle_type]
        style = STYLES[vehicle_type]
        
        logger.info(f"Plotting {len(gdf)} {vehicle_type} segments...")
        
        # Calculate line widths
        widths = self.calculate_line_widths(gdf, vehicle_type)
        
        # Plot outline first (white stroke for contrast)
        gdf.plot(
            ax=self.ax,
            linewidth=widths + style['outline_width'],
            color=style['outline_color'],
            capstyle=style['cap_style'],
            joinstyle=style['join_style'],
            alpha=0.6,
            zorder=10 if vehicle_type == 'tram' else (9 if vehicle_type == 'train' else 8)
        )
        
        # Plot main line on top
        gdf.plot(
            ax=self.ax,
            linewidth=widths,
            color=style['color'],
            capstyle=style['cap_style'],
            joinstyle=style['join_style'],
            alpha=style['alpha'],
            zorder=11 if vehicle_type == 'tram' else (10 if vehicle_type == 'train' else 9)
        )
    
    def set_extent(self):
        """Set map extent/bounds"""
        if USE_AUTO_EXTENT:
            # Calculate bounds from all data
            all_bounds = []
            
            for gdf in self.vehicle_data.values():
                bounds = gdf.total_bounds
                all_bounds.append(bounds)
            
            if all_bounds:
                all_bounds = np.array(all_bounds)
                xmin = all_bounds[:, 0].min() - EXTENT_BUFFER
                ymin = all_bounds[:, 1].min() - EXTENT_BUFFER
                xmax = all_bounds[:, 2].max() + EXTENT_BUFFER
                ymax = all_bounds[:, 3].max() + EXTENT_BUFFER
                
                self.ax.set_xlim(xmin, xmax)
                self.ax.set_ylim(ymin, ymax)
        else:
            # Use predefined extent
            self.ax.set_xlim(MAP_EXTENT['xmin'], MAP_EXTENT['xmax'])
            self.ax.set_ylim(MAP_EXTENT['ymin'], MAP_EXTENT['ymax'])
    
    def add_legend(self):
        """Add legend to the map"""
        if not SHOW_LEGEND:
            return
        
        logger.info("Adding legend...")
        
        # Create legend elements
        legend_elements = []
        
        for vehicle_type in ['tram', 'bus', 'train']:
            if vehicle_type in self.vehicle_data:
                style = STYLES[vehicle_type]
                
                # Create representative line
                legend_elements.append(
                    Line2D(
                        [0], [0],
                        color=style['color'],
                        linewidth=3,
                        label=style['label'],
                        alpha=style['alpha']
                    )
                )
        
        # Add legend
        legend = self.ax.legend(
            handles=legend_elements,
            loc='upper right',
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=10,
            title='Vehicle Type',
            title_fontsize=11
        )
        
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
    
    def add_scale_bar(self):
        """Add scale bar to the map"""
        if not SHOW_SCALE_BAR:
            return
        
        logger.info("Adding scale bar...")
        
        scalebar = ScaleBar(
            dx=1,  # 1 meter per unit (EPSG:2180 is in meters)
            units='m',
            location='lower left',
            length_fraction=0.15,
            width_fraction=0.01,
            box_alpha=0.7,
            color='black',
            font_properties={'size': 9}
        )
        
        self.ax.add_artist(scalebar)
    
    def add_title(self):
        """Add title and subtitle to the map"""
        logger.info("Adding title...")
        
        # Main title
        self.fig.suptitle(
            MAP_TITLE,
            fontsize=18,
            fontweight='bold',
            y=0.98
        )
        
        # Subtitle
        if MAP_SUBTITLE:
            self.ax.text(
                0.5, 1.02,
                MAP_SUBTITLE,
                transform=self.ax.transAxes,
                fontsize=12,
                ha='center',
                va='bottom',
                style='italic',
                color='#555555'
            )
    
    def add_attribution(self):
        """Add data attribution"""
        attribution_text = (
            f"Data: ZTM Warszawa GTFS | "
            f"Width represents: {WIDTH_ATTRIBUTE.replace('_', ' ').title()} | "
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}"
        )
        
        self.fig.text(
            0.99, 0.01,
            attribution_text,
            fontsize=7,
            ha='right',
            va='bottom',
            color='#666666',
            style='italic'
        )
    
    def generate_map(self, vehicles_to_plot: list = None, use_capacity: bool = True):
        """
        Generate the complete map
        
        Args:
            vehicles_to_plot: List of vehicles to plot (default: all available)
            use_capacity: Use capacity-enriched data if available
        """
        logger.info("="*70)
        logger.info("Generating Warsaw Transit Frequency Map")
        logger.info("="*70)
        
        # Load data
        self.load_data(use_capacity=use_capacity)
        
        # Setup
        self.setup_figure()
        self.set_extent()
        
        # Add basemap first (background)
        self.add_basemap()
        
        # Plot vehicle types
        if vehicles_to_plot is None:
            vehicles_to_plot = ['train', 'bus', 'tram']  # Order matters for layering
        
        for vehicle_type in vehicles_to_plot:
            self.plot_vehicle_type(vehicle_type)
        
        # Add map elements
        self.add_legend()
        self.add_scale_bar()
        self.add_title()
        self.add_attribution()
        
        # Adjust layout
        plt.tight_layout()
    
    def save_map(self, filename_base: str = "warsaw_transit_map"):
        """
        Save the map in multiple formats
        
        Args:
            filename_base: Base filename (without extension)
        """
        if self.fig is None:
            raise ValueError("No map to save. Run generate_map() first.")
        
        logger.info("Saving map...")
        
        for fmt in OUTPUT_FORMATS:
            output_path = self.output_dir / f"{filename_base}.{fmt}"
            
            logger.info(f"Saving {fmt.upper()} to {output_path}")
            
            self.fig.savefig(
                output_path,
                format=fmt,
                dpi=MAP_SIZE['dpi'],
                bbox_inches='tight',
                facecolor=self.fig.get_facecolor(),
                edgecolor='none'
            )
        
        logger.info(f"Map saved to {self.output_dir}")
    
    def show(self):
        """Display the map (for interactive use)"""
        if self.fig is None:
            raise ValueError("No map to show. Run generate_map() first.")
        
        plt.show()


if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(
        description="Generate Warsaw transit frequency map"
    )
    parser.add_argument(
        '--vehicles',
        nargs='+',
        choices=['bus', 'tram', 'train'],
        help='Vehicle types to include (default: all)'
    )
    parser.add_argument(
        '--no-capacity',
        action='store_true',
        help='Use regular aggregated files instead of capacity-enriched files'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='warsaw_transit_map',
        help='Output filename base (without extension)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the map interactively'
    )
    
    args = parser.parse_args()
    
    # Generate map
    generator = TransitMapGenerator()
    generator.generate_map(
        vehicles_to_plot=args.vehicles,
        use_capacity=not args.no_capacity
    )
    
    # Save
    generator.save_map(filename_base=args.output_name)
    
    # Show if requested
    if args.show:
        generator.show()
    
    logger.info("Done!")
