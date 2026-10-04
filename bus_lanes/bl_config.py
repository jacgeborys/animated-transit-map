"""Shared settings for the bus-lane need map (bus_lanes/)."""
from pathlib import Path

BL_DIR = Path(__file__).parent
PROJECT_ROOT = BL_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = BL_DIR / "_data"      # intermediate files (gitignored)
OUTPUT_DIR = BL_DIR / "_output"  # final maps + tables

POLAND_CRS = "EPSG:2180"

# (south, west, north, east) — Warsaw + a margin for suburban routes entering the city
BBOX = (52.08, 20.82, 52.39, 21.30)

# Analysis date: None = pick a random regular Wednesday covered by the feed.
# Set e.g. "20261014" to pin it (the chosen date is printed and written to outputs).
ANALYSIS_DATE = None
RANDOM_SEED = None               # set an int for a reproducible random pick

# Peak windows used for "buses per hour" (per direction); see PM_* at the bottom
PEAK_START_H = 7
PEAK_END_H = 9                   # exclusive → 07:00–08:59

# Map matching (GTFS shape → OSM way)
SAMPLE_STEP_M = 10               # densify shapes every N metres
MATCH_MAX_DIST_M = 15            # max distance from sample point to street centreline
MATCH_MAX_ANGLE_DEG = 35         # max bearing difference between shape and street

# Classification thresholds: peak buses per hour in ONE direction
NEED_HIGH = 30                   # ≥ 30 buses/h (one every 2 min) — bus lane clearly needed
NEED_MED = 15                    # ≥ 15 buses/h (one every 4 min) — bus lane worth considering
PM_START_H = 15                  # afternoon peak 15:00–16:59; buses/h = max(AM, PM) / 2
PM_END_H = 17

# Which bus-lane sources count (after the Oct 2026 community OSM edits, OSM is the verified source;
# "city" = mapa.um.warszawa.pl 2021 layer, "manual" = manual_bus_lanes.csv — still computed for reference)
LANE_SOURCES = ("osm",)
# Lane stretches (connected, same direction) shorter than this are bus-stop bays / terminus bits, not bus lanes
MIN_LANE_STRETCH_M = 80
