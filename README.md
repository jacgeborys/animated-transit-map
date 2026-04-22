# Warsaw GTFS Transit Frequency Analysis

A modern Python pipeline for analyzing Warsaw's public transit frequencies using GTFS schedule data. The system automatically downloads the latest GTFS data, processes it to identify peak-hour transit patterns, and generates geospatial outputs showing transit frequencies across the network.

## Features

- **Automatic data download**: Fetches latest Warsaw GTFS data from mkuran.pl
- **Schedule filtering**: Analyzes specific dates and time periods (configurable peak hours)
- **Vehicle classification**: Separates trams, buses, and trains
- **Geometry processing**: Converts GTFS points to line segments with accurate bearings
- **Parallel segment aggregation**: Merges overlapping routes to calculate combined frequencies
- **Capacity estimation**: Adds passenger capacity calculations to segments
- **Static map generation**: Creates professional transit frequency cartograms
- **Animation support**: Generate animated visualizations of transit movement
- **Clean architecture**: Modular design with separate concerns for each processing step

## Quick Start (5 minutes)

### 1. Installation

```bash
cd D:\QGIS\gtfs_schedules_city
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python main.py
```

This will:
- Download latest Warsaw GTFS data (~30 seconds)
- Process schedules for the configured date (default: January 19, 2026)
- Create line segments from GTFS shapes
- Aggregate parallel routes into frequency maps
- Output separate shapefiles for buses, trams, and trains

**Total time**: ~2-5 minutes depending on your machine

### 3. Load Results in QGIS

1. Open QGIS
2. Add Vector Layer → Navigate to `data/output/`
3. Load `aggregated_bus.shp`, `aggregated_tram.shp`, `aggregated_train.shp`
4. Style by `trip_sum` attribute to show frequency

**Pro tip**: Use graduated symbology with `trip_sum` field. Higher values = more frequent service.

## Project Structure

```
gtfs_schedules_city/
├── core/                          # Core pipeline modules
│   ├── 1_gtfs_downloader.py      # Download and extract GTFS data
│   ├── 2_gtfs_parser.py          # Parse and filter schedule data
│   ├── 3_geometry_processor.py   # Convert points to line segments
│   ├── 4_segment_aggregator.py   # Aggregate parallel segments
│   └── 5_route_line_builder.py   # Build continuous route lines
├── visualization/                 # Static map generation
│   ├── map_generator.py          # Generate transit frequency maps
│   ├── segmenter.py              # Split lines at intersections
│   ├── config_viz.py             # Visualization configuration
│   └── requirements_viz.txt      # Visualization dependencies
├── animate/                       # Animation scripts
│   ├── animate_segments.py       # Segmented animation
│   ├── animate_density_full.py   # Full city density animation
│   ├── animate_segments_density.py  # Density with segments
│   └── _output/                  # Animation output videos
├── data/                          # Data storage
│   ├── raw/                      # Downloaded GTFS data
│   ├── processed/                # Intermediate outputs
│   └── output/                   # Final aggregated outputs
├── out/                           # Other outputs
│   └── maps/                     # Static maps & diagnostics
├── x_capacity_calculator.py       # Add capacity estimates
├── x_create_segment_diagnostic.py # Diagnostic tools
├── config.py                      # Main configuration
├── main.py                        # Main entry point
└── requirements.txt               # Main dependencies
```

## Command Line Options

```bash
# Analyze a specific date
python main.py --date 20260119

# Force download of fresh GTFS data
python main.py --download-fresh

# Use existing data without checking for updates
python main.py --skip-download
```

## Configuration

Edit `config.py` to customize:

```python
# Analysis date
ANALYSIS_DATE = "20260119"  # YYYYMMDD format

# Peak hours (modify as needed)
PEAK_HOURS = [
    (time(6, 0), time(11, 0)),   # Morning peak
    (time(14, 0), time(19, 0)),  # Evening peak
]

# Geometric parameters
SEGMENT_LENGTH = 10          # Split segments to this length (meters)
BUFFER_DISTANCE = 4          # Buffer for matching parallel segments (meters)
DIRECTION_TOLERANCE = 10     # Max bearing difference for matching (degrees)
```

## Output Files

### Individual Segments
- **Location**: `data/processed/individual_segments.shp`
- **Description**: All transit line segments before aggregation
- **Attributes**:
  - `shape_id`: Original GTFS shape identifier
  - `trip_count`: Number of trips on this segment
  - `vehicle`: Vehicle type (Bus, Tram, or Train)
  - `length`: Segment length in meters
  - `bearing`: Direction in degrees (0-360)

### Aggregated Segments
- **Location**: `data/output/aggregated_{vehicle}.shp`
- **Description**: Merged segments showing combined frequencies
- **Attributes**:
  - `trip_sum`: Total trips from all merged parallel routes
  - `vehicle`: Vehicle type
  - `length`: Segment length in meters
  - `bearing`: Direction in degrees

### Understanding trip_sum

**trip_sum attribute**: Total number of transit trips during peak hours on that segment

- 100+ trips = Very high frequency (every few minutes)
- 50-100 trips = High frequency (every 5-10 minutes)
- 20-50 trips = Moderate frequency
- <20 trips = Low frequency

This accounts for ALL routes using that street segment, so busy corridors show combined frequency.

## How It Works

### 1. Data Download (`core/1_gtfs_downloader.py`)
- Fetches latest Warsaw GTFS data from https://mkuran.pl/gtfs/warsaw.zip
- Extracts to timestamped folder in `data/raw/`
- Can reuse existing data or force fresh download

### 2. Schedule Parsing (`core/2_gtfs_parser.py`)
- Loads GTFS files: `shapes.txt`, `trips.txt`, `stop_times.txt`, `calendar.txt`
- Filters trips by date (finds correct service_id)
- Filters stop times by peak hours
- Classifies vehicles based on route_id patterns:
  - 1-2 digits → Tram
  - 3 digits → Bus
  - Starts with S/R → Train
- Aggregates trip counts per shape

### 3. Geometry Processing (`core/3_geometry_processor.py`)
- Converts GTFS shape points to LineString segments
- Calculates bearing (direction) for each segment
- Reprojects to EPSG:2180 (Poland CS92) for metric calculations
- Filters out very short segments (<10m)

### 4. Segment Aggregation (`core/4_segment_aggregator.py`)
- Splits long segments into ~10m pieces for better matching
- Uses spatial indexing for efficient overlap detection
- Merges segments that are:
  - Within 4m of each other (buffer distance)
  - Traveling in same direction (±10° bearing tolerance)
  - From different shape_ids (avoids double-counting)
- Processes each vehicle type separately
- Outputs combined frequency maps

## Capacity Calculator

Add passenger capacity estimates to your aggregated shapefiles without rerunning the entire pipeline.

### Usage

```bash
python x_capacity_calculator.py
```

This adds new attributes:
- `veh_cap` - Estimated vehicle capacity (passengers: seated + standing)
- `total_cap` - Total capacity offered (trip_sum × veh_cap)
- `cap_hr` - Average capacity per hour during peak periods

### Capacity Estimates

**Buses (avg: 100 passengers)**
- Articulated (18m): 120 passengers
- Standard (12m): 90 passengers
- Default: 100 passengers

**Trams (avg: 200 passengers)**
- Modern low-floor: 240 passengers
- Standard: 180 passengers
- Default: 200 passengers

**Trains (avg: 600 passengers)**
- SKM (commuter rail): 800 passengers
- WKD (light rail): 400 passengers
- Default: 600 passengers

### Visualize

Load the `*_capacity.shp` files in QGIS and style by:
- `total_cap` - Shows total capacity corridors (best for planning)
- `cap_per_hr` - Shows hourly capacity (useful for comparisons)
- `trip_sum` - Still shows frequency (number of vehicles)

## Map Visualization

Generate beautiful, publication-ready transit frequency maps with variable-width lines.

### Quick Start

1. Install visualization dependencies:
```bash
pip install -r visualization/requirements_viz.txt
```

2. Generate map:
```bash
python visualization/map_generator.py
```

Outputs:
- `out/maps/warsaw_transit_map.png` - High-res raster (300 DPI)
- `out/maps/warsaw_transit_map.pdf` - Vector for printing
- `out/maps/warsaw_transit_map.svg` - Vector for editing

### Features

- Variable line width based on frequency or capacity
- Round line caps for smooth appearance
- White outlines for contrast against background
- Color-coded by vehicle type:
  - Red: Trams
  - Purple: Buses
  - Teal: Trains
- Optional OSM basemap for context
- Professional cartography with legend, scale bar, title

### Customization

```bash
# Show only specific vehicles
python visualization/map_generator.py --vehicles tram bus

# Use regular aggregated data (without capacity)
python visualization/map_generator.py --no-capacity

# Custom output name
python visualization/map_generator.py --output-name "warsaw_buses_2026"

# Preview interactively
python visualization/map_generator.py --show
```

Edit `visualization/config_viz.py` to customize colors, line widths, map size, and more.

## Animation

Create animated visualizations showing transit vehicles moving through the city.

### Prerequisites

Generate route lines and schedule data:
```bash
python core/5_route_line_builder.py
```

### Generate Animation

```bash
python animate/animate_segments.py
```

Outputs: `animate/_output/warsaw_segments.mp4`

### Features

- Lines split at intersections (street segments)
- Overlapping routes brighten same segments
- Brightness accumulates as vehicles pass
- Moving vehicle dots on top
- Central Warsaw focus (25km frame)

### Configuration

Edit animation scripts in `animate/` to customize:
- FPS and duration
- Map extent
- Visual settings (line width, colors, glow effects)
- Vehicle sizes

## Use Cases

- **Transit planning**: Identify high-frequency corridors
- **Transport analysis**: Compare modal coverage across the network
- **Data visualization**: Create transit frequency maps in QGIS
- **Research**: Analyze temporal patterns in public transport
- **Infrastructure planning**: Understand current service distribution
- **Presentations**: Generate professional maps and animations

## Technical Details

### Coordinate Systems
- **Input**: WGS84 (EPSG:4326) - from GTFS data
- **Processing**: Poland CS92 (EPSG:2180) - for accurate metric measurements
- **Output**: Poland CS92 (EPSG:2180) - shapefiles

### Vehicle Classification Logic
Based on Warsaw's route numbering:
- Routes 1-99: Trams
- Routes 100-999: Buses
- Routes starting with S/R: Trains (SKM, WKD)

### Performance
- Processes ~100,000+ GTFS shape points in minutes
- Spatial indexing enables efficient parallel segment matching
- Progress bars show real-time processing status

## Dependencies

Main pipeline:
- **pandas**: Data manipulation
- **geopandas**: Geospatial data processing
- **shapely**: Geometric operations
- **numpy**: Numerical calculations
- **requests**: HTTP downloads
- **tqdm**: Progress bars

Visualization (optional):
- **matplotlib**: Map generation
- **contextily**: Basemap tiles
- **matplotlib-scalebar**: Scale bars

Animation (optional):
- **matplotlib.animation**: Video generation
- **ffmpeg**: Video encoding

## Troubleshooting

**No data for specified date**: Check that the date exists in the GTFS calendar. Weekday service typically uses service_id ending in '_2'.

**Missing segments**: Verify that peak hours in `config.py` match your analysis needs. No trips during those hours = no segments.

**Spatial operations slow**: Ensure you have enough RAM. The spatial index requires memory proportional to segment count.

**Download fails**: Check internet connection and verify GTFS_URL in `config.py` is accessible.

**Lines too thin/thick in maps**: Adjust `min_width` and `max_width` in `visualization/config_viz.py`.

**Basemap not appearing**: Check internet connection (downloads tiles) or try different basemap provider in config.

**Animation too slow**: Reduce FPS or duration in animation scripts.

## License

This project processes publicly available GTFS data from Warsaw's public transit system.

## Acknowledgments

- GTFS data provided by mkuran.pl
- Warsaw public transit data from ZTM Warszawa
