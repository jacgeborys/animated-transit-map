"""
Configuration for map visualization and cartogram generation
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
MAPS_DIR = PROJECT_ROOT / "out" / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

# Map extent (Warsaw bounding box in EPSG:2180)
# Approximate Warsaw city boundaries
MAP_EXTENT = {
    'xmin': 640000,
    'xmax': 680000,
    'ymin': 470000,
    'ymax': 505000
}

# Or use auto-extent from data
USE_AUTO_EXTENT = True
EXTENT_BUFFER = 2000  # meters to add around data extent

# Map size (A3 landscape for professional output)
MAP_SIZE = {
    'width': 16.5,  # inches (A3 = 16.5" x 11.7")
    'height': 11.7,
    'dpi': 300
}

# Style configuration
STYLES = {
    'tram': {
        'color': '#b0252f',  # Red
        'outline_color': '#FFFFFF',  # White outline
        'outline_width': 0.8,  # Subtle outline
        'min_width': 0.5,  # Minimum line width (mm)
        'max_width': 4.0,  # Maximum line width (mm)
        'cap_style': 'round',  # Round line endings
        'join_style': 'round',
        'alpha': 0.85,  # Slight transparency
        'label': 'Tram Lines'
    },
    'bus': {
        'color': '#7209B7',  # Purple
        'outline_color': '#FFFFFF',  # White outline
        'outline_width': 0.8,
        'min_width': 0.3,
        'max_width': 3.0,
        'cap_style': 'round',
        'join_style': 'round',
        'alpha': 0.85,
        'label': 'Bus Lines'
    },
    'train': {
        'color': '#2A9D8F',  # Teal/green
        'outline_color': '#FFFFFF',
        'outline_width': 1.0,
        'min_width': 0.8,
        'max_width': 5.0,
        'cap_style': 'round',
        'join_style': 'round',
        'alpha': 0.90,
        'label': 'Train Lines'
    }
}

# Background/basemap
BACKGROUND = {
    'color': '#F8F9FA',  # Very light gray
    'use_osm_basemap': True,  # Use OpenStreetMap background
    'osm_style': 'cartodb-positron',  # Light, clean style
    # Alternative: 'stamen-toner-lite', 'cartodb-positron', 'openstreetmap'
}

# Which attribute to use for line width scaling
WIDTH_ATTRIBUTE = 'trip_sum'  # Options: 'trip_sum', 'total_cap', 'cap_per_hr'

# Title and labels
MAP_TITLE = "Warsaw Public Transit Frequency"
MAP_SUBTITLE = "Peak hours service (6-11 AM, 14-19 PM)"
SHOW_LEGEND = True
SHOW_SCALE_BAR = True
SHOW_NORTH_ARROW = True

# Output format
OUTPUT_FORMATS = ['png', 'pdf', 'svg']  # Generate multiple formats
