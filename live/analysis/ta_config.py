"""
Settings for the tram time-loss analysis (live GPS vs GTFS timetable).

Everything that is an ASSUMPTION (not measured) is marked as such — change it here and re-run
priority_scenarios.py / report.py; the measured tables do not depend on it.
"""

from pathlib import Path

DATE = "20261124"                       # positions_<DATE>.csv in live/data/

ROOT       = Path(__file__).resolve().parents[2]
LIVE_DATA  = ROOT / "live" / "data"
RAW_DIR    = ROOT / "data" / "raw"
HERE       = Path(__file__).resolve().parent
CACHE      = HERE / "_cache"
OUT        = HERE / "_output"
OSM_ROADS  = Path(r"D:\QGIS\mapy_warszawy_misc\data\osm\roads.shp")
CRS        = 2180                       # PL-1992, metres

# ---------------------------------------------------------------- matching GPS -> scheduled trip
OFF_SHAPE_M   = 40      # fix must be this close to the trip's shape
EARLY_S       = 300     # a trip may be observed up to 5 min before its schedule ...
LATE_S        = 1500    # ... and up to 25 min after
BACKTRACK_M   = 30      # fixes moving backwards along the trip by more than this are dropped

# ---------------------------------------------------------------- classifying stand-stills
STAND_KMH     = 5.0     # interval speed below this = standing
MAX_DT        = 90      # s; longer gaps between fixes break the trace
STOP_BEFORE_M = 50      # standing within [stop - 50 m, stop + 25 m] counts as "at the platform"
STOP_AFTER_M  = 25
TERMINUS_M    = 150     # first / last 150 m of a trip = terminus loop (layover, regulation), excluded
ANOMALY_MIN   = 8       # single stand-stills longer than this = breakdown / held / incident, reported apart

# ---------------------------------------------------------------- places (junctions)
CELL_M        = 40      # stand-stills are pooled across lines on this grid, per travel heading
HEADINGS      = 8       # 45° heading sectors -> each direction of a junction approach is separate
SAMPLE_M      = 10      # step when counting how many vehicles passed a cell
MIN_PASSES    = 60      # a place needs at least this many passing trams in the day to be ranked

PEAK_HOURS    = [(7, 9), (15, 18)]

# ---------------------------------------------------------------- fleet  (measured inputs: tramwar.pl, read 26.11.2026)
# fleet-number range -> type, places (nominal capacity as given on tramwar.pl), length m, runs as 2-car set
# NOTE: tramwar.pl mixes capacity norms between types (some 5 pax/m², some older norms) — treat as ±15 %.
# 105N-family sets: the GPS reports one car number; capacity = 2 cars x 97 (5 pax/m², tw105na.html).
FLEET = [
    # lo,   hi,   type,          places, length_m, set
    (1001, 2073, "105N (set)",   194,   27.0,  True),
    (2074, 2135, "105N2k/2000",  200,   27.9,  True),   # 123+132 on the page (older norm) scaled to ~5 pax/m²
    (2136, 2165, "123N (set)",   194,   28.6,  True),
    (3001, 3030, "116Na",        215,   24.05, False),
    (3101, 3115, "120N",         211,   31.82, False),
    (3116, 3295, "120Na Swing",  201,   30.12, False),
    (3501, 3506, "120Na DUO",    200,   30.12, False),
    (3601, 3650, "128N Jazz",    215,   29.7,  False),
    (3801, 3830, "134N Jazz",    132,   19.3,  False),
    (4001, 4020, "142N",         162,   23.42, False),
    (4101, 4118, "141N",         240,   32.9,  False),
    (4201, 4285, "140N",         240,   32.9,  False),
]

# ---------------------------------------------------------------- ASSUMPTIONS (scenario inputs)
LOAD_FACTOR      = {"peak": 0.50, "offpeak": 0.25}   # share of nominal places occupied — NOT measured
PRIORITY_EFFECT  = 0.70     # places.py only: share of stand-still time a working priority would remove
TARGET_STOP_RATE = 0.10     # scenario: every signal brought to the level of the good ones on the Wilanów line (2-12 %)
ACCEL_PENALTY_S  = 8        # extra seconds a stop-and-go costs beyond standing still (braking + accelerating)
VALUE_OF_TIME_PLN_H = 25    # passenger value of time, PLN per hour (order of magnitude, for illustration)
