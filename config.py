"""
Configuration file for Warsaw GTFS data processing pipeline
"""
from pathlib import Path
from datetime import time

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# GTFS data source
GTFS_URL = "https://mkuran.pl/gtfs/warsaw.zip"

# Processing parameters
ANALYSIS_DATE = "20260119"  # Format: YYYYMMDD - will be updated to latest available

# Peak hours for analysis (two time ranges)
PEAK_HOURS = [
    (time(6, 0), time(11, 0)),   # Morning peak
    (time(14, 0), time(19, 0)),  # Evening peak
]

PEAK_HOURS = [
    (time(4, 0), time(23, 0)),   # Morning peak
    # (time(14, 0), time(19, 0)),  # Evening peak
]


# Geometric parameters
SEGMENT_LENGTH = 10  # meters - for splitting lines
BUFFER_DISTANCE = 4  # meters - for matching parallel segments
DIRECTION_TOLERANCE = 10  # degrees - for matching segments by direction
MIN_SEGMENT_LENGTH = 10  # meters - filter out very short segments

# Coordinate reference systems
WGS84 = "EPSG:4326"
POLAND_CRS = "EPSG:2180"  # ETRS89 / Poland CS92

# Vehicle type classification rules
VEHICLE_TYPE_RULES = {
    'tram': lambda route_id: route_id.isdigit() and len(route_id) <= 2,
    'bus': lambda route_id: route_id.isdigit() and len(route_id) == 3,
    'train': lambda route_id: route_id.startswith(('S', 'R')),
}
