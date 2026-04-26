"""
Extra Feeds GTFS Merger
========================
Downloads and merges Koleje Mazowieckie (from polish_trains.zip) and
WKD (from wkd.zip) into a combined directory alongside ZTM Warsaw GTFS data.

Pipeline position: run once after gtfs_downloader.py.
Pass the returned combined_dir to RouteLineBuilder instead of the raw ZTM dir.

Usage:
    from core.km_gtfs_merger import merge_km_into_ztm
    combined_dir = merge_km_into_ztm(ztm_dir)
"""

import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import logging

import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import POLISH_TRAINS_URL, WKD_URL, RAW_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KM_PREFIX  = 'km_'
WKD_PREFIX = 'wkd_'

COMBINED_SUFFIX = '_with_km_wkd'


# ---------------------------------------------------------------------------
# Generic download helper
# ---------------------------------------------------------------------------

def _download_and_extract(url: str, name: str, output_dir: Path) -> Path:
    """Download a zip from url, extract to a timestamped dir, return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = output_dir / f'{name}_{timestamp}.zip'
    extract_dir = output_dir / f'{name}_{timestamp}'

    logger.info(f'Downloading {name} from {url}')
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            downloaded += len(chunk)
            f.write(chunk)
            if total_size:
                print(f'\rDownload progress: {downloaded / total_size * 100:.1f}%', end='')
    print()

    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()

    logger.info(f'{name} GTFS extracted to {extract_dir}')
    return extract_dir


def _get_latest_dir(prefix: str, raw_dir: Path, url: str) -> Path:
    """Return the most recent directory with given prefix, downloading if none exists."""
    raw_dir = Path(raw_dir)
    dirs = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda x: x.name,
        reverse=True,
    )
    if not dirs:
        return _download_and_extract(url, prefix.rstrip('_'), raw_dir)
    logger.info(f'Using existing {prefix} data from {dirs[0]}')
    return dirs[0]


# Public download helpers
def download_polish_trains(output_dir: Path = RAW_DIR) -> Path:
    return _download_and_extract(POLISH_TRAINS_URL, 'polish_trains', output_dir)

def download_wkd(output_dir: Path = RAW_DIR) -> Path:
    return _download_and_extract(WKD_URL, 'wkd', output_dir)

def get_latest_polish_trains_dir(raw_dir: Path = RAW_DIR) -> Path:
    return _get_latest_dir('polish_trains_', raw_dir, POLISH_TRAINS_URL)

def get_latest_wkd_dir(raw_dir: Path = RAW_DIR) -> Path:
    return _get_latest_dir('wkd_', raw_dir, WKD_URL)


# ---------------------------------------------------------------------------
# Calendar helper
# ---------------------------------------------------------------------------

def _expand_calendar_txt(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Convert calendar.txt (day-of-week patterns) to calendar_dates.txt format."""
    day_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    rows = []
    for _, row in calendar_df.iterrows():
        start = datetime.strptime(str(int(row['start_date'])), '%Y%m%d')
        end   = datetime.strptime(str(int(row['end_date'])),   '%Y%m%d')
        current = start
        while current <= end:
            if int(row[day_cols[current.weekday()]]) == 1:
                rows.append({
                    'service_id':     row['service_id'],
                    'date':           int(current.strftime('%Y%m%d')),
                    'exception_type': 1,
                })
            current += timedelta(days=1)
    return pd.DataFrame(rows)


def _load_calendar_dates(gtfs_dir: Path, service_ids: set) -> pd.DataFrame:
    """Load calendar entries for the given service_ids, handling both calendar formats."""
    if (gtfs_dir / 'calendar_dates.txt').exists():
        df = pd.read_csv(gtfs_dir / 'calendar_dates.txt', dtype=str)
        df['date']           = df['date'].astype(int)
        df['exception_type'] = df['exception_type'].astype(int)
        result = df[df['service_id'].isin(service_ids)].copy()
        logger.info(f'  Calendar entries from calendar_dates.txt: {len(result)}')
    elif (gtfs_dir / 'calendar.txt').exists():
        cal = pd.read_csv(gtfs_dir / 'calendar.txt', dtype=str)
        result = _expand_calendar_txt(cal[cal['service_id'].isin(service_ids)])
        logger.info(f'  Calendar entries expanded from calendar.txt: {len(result)}')
    else:
        raise FileNotFoundError(f'Neither calendar.txt nor calendar_dates.txt found in {gtfs_dir}')
    return result


# ---------------------------------------------------------------------------
# KM extraction (from polish_trains.zip, filter by agency)
# ---------------------------------------------------------------------------

def _find_km_agency_id(trains_dir: Path) -> str:
    agency = pd.read_csv(trains_dir / 'agency.txt', dtype=str)
    km_row = agency[agency['agency_name'].str.contains('Mazowieckie', case=False, na=False)]
    if km_row.empty:
        raise ValueError(
            f'Could not find Koleje Mazowieckie in {trains_dir}/agency.txt. '
            f'Available agencies: {agency["agency_name"].tolist()}'
        )
    agency_id = str(km_row.iloc[0]['agency_id'])
    logger.info(f'Found KM agency_id: "{agency_id}"')
    return agency_id


def extract_km_data(trains_dir: Path) -> dict:
    """
    Extract KM-only data from polish_trains directory and apply km_ prefix to all IDs.
    Route IDs are kept as-is (KM_R1, KM_R2, ...) since they are already distinctive.
    """
    trains_dir = Path(trains_dir)
    logger.info('Extracting KM data from polish_trains...')
    km_agency_id = _find_km_agency_id(trains_dir)

    routes    = pd.read_csv(trains_dir / 'routes.txt', dtype=str)
    km_route_ids = set(routes.loc[routes['agency_id'] == km_agency_id, 'route_id'])
    logger.info(f'  KM routes ({len(km_route_ids)}): {sorted(km_route_ids)}')

    trips     = pd.read_csv(trains_dir / 'trips.txt', dtype=str)
    km_trips  = trips[trips['route_id'].isin(km_route_ids)].copy()
    km_trip_ids    = set(km_trips['trip_id'])
    km_shape_ids   = set(km_trips['shape_id'].dropna())
    km_service_ids = set(km_trips['service_id'])
    logger.info(f'  KM trips: {len(km_trips)}, shapes: {len(km_shape_ids)}')

    shapes         = pd.read_csv(trains_dir / 'shapes.txt', dtype=str)
    km_shapes      = shapes[shapes['shape_id'].isin(km_shape_ids)].copy()

    stop_times     = pd.read_csv(trains_dir / 'stop_times.txt', dtype=str)
    km_stop_times  = stop_times[stop_times['trip_id'].isin(km_trip_ids)].copy()
    km_stop_ids    = set(km_stop_times['stop_id'])

    stops          = pd.read_csv(trains_dir / 'stops.txt', dtype=str)
    km_stops       = stops[stops['stop_id'].isin(km_stop_ids)].copy()

    km_cal_dates   = _load_calendar_dates(trains_dir, km_service_ids)

    # Apply km_ prefix (route_id left as-is)
    km_trips['trip_id']      = KM_PREFIX + km_trips['trip_id']
    km_trips['shape_id']     = KM_PREFIX + km_trips['shape_id']
    km_trips['service_id']   = KM_PREFIX + km_trips['service_id']
    km_shapes['shape_id']    = KM_PREFIX + km_shapes['shape_id']
    km_stop_times['trip_id'] = KM_PREFIX + km_stop_times['trip_id']
    km_stop_times['stop_id'] = KM_PREFIX + km_stop_times['stop_id']
    km_stops['stop_id']      = KM_PREFIX + km_stops['stop_id']
    km_cal_dates['service_id'] = KM_PREFIX + km_cal_dates['service_id'].astype(str)

    return {
        'trips': km_trips, 'shapes': km_shapes,
        'stop_times': km_stop_times, 'stops': km_stops,
        'calendar_dates': km_cal_dates,
    }


# ---------------------------------------------------------------------------
# WKD extraction (standalone feed — take all routes, prefix everything)
# ---------------------------------------------------------------------------

def extract_wkd_data(wkd_dir: Path) -> dict:
    """
    Extract all WKD data and apply wkd_ prefix to all IDs including route_ids
    (WKD route IDs may be simple short strings that could collide with ZTM).
    """
    wkd_dir = Path(wkd_dir)
    logger.info('Extracting WKD data...')

    trips          = pd.read_csv(wkd_dir / 'trips.txt', dtype=str)
    wkd_trip_ids   = set(trips['trip_id'])
    wkd_shape_ids  = set(trips['shape_id'].dropna())
    wkd_service_ids= set(trips['service_id'])
    wkd_route_ids  = set(trips['route_id'])
    logger.info(f'  WKD trips: {len(trips)}, routes: {sorted(wkd_route_ids)}')

    shapes         = pd.read_csv(wkd_dir / 'shapes.txt', dtype=str)
    wkd_shapes     = shapes[shapes['shape_id'].isin(wkd_shape_ids)].copy()

    stop_times     = pd.read_csv(wkd_dir / 'stop_times.txt', dtype=str)
    wkd_stop_times = stop_times[stop_times['trip_id'].isin(wkd_trip_ids)].copy()
    wkd_stop_ids   = set(wkd_stop_times['stop_id'])

    stops          = pd.read_csv(wkd_dir / 'stops.txt', dtype=str)
    wkd_stops      = stops[stops['stop_id'].isin(wkd_stop_ids)].copy()

    wkd_cal_dates  = _load_calendar_dates(wkd_dir, wkd_service_ids)

    # Apply wkd_ prefix to everything including route_id
    trips['trip_id']          = WKD_PREFIX + trips['trip_id']
    trips['shape_id']         = WKD_PREFIX + trips['shape_id']
    trips['service_id']       = WKD_PREFIX + trips['service_id']
    trips['route_id']         = WKD_PREFIX + trips['route_id']
    wkd_shapes['shape_id']   = WKD_PREFIX + wkd_shapes['shape_id']
    wkd_stop_times['trip_id']= WKD_PREFIX + wkd_stop_times['trip_id']
    wkd_stop_times['stop_id']= WKD_PREFIX + wkd_stop_times['stop_id']
    wkd_stops['stop_id']     = WKD_PREFIX + wkd_stops['stop_id']
    wkd_cal_dates['service_id'] = WKD_PREFIX + wkd_cal_dates['service_id'].astype(str)

    return {
        'trips': trips, 'shapes': wkd_shapes,
        'stop_times': wkd_stop_times, 'stops': wkd_stops,
        'calendar_dates': wkd_cal_dates,
    }


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def _append_to_combined(combined_dir: Path, data: dict, label: str):
    """Append a feed's DataFrames into the combined directory files."""
    file_map = {
        'trips':          'trips.txt',
        'shapes':         'shapes.txt',
        'stop_times':     'stop_times.txt',
        'stops':          'stops.txt',
        'calendar_dates': 'calendar_dates.txt',
    }
    for key, filename in file_map.items():
        target  = combined_dir / filename
        base_df = pd.read_csv(target, dtype=str)
        new_df  = data[key].astype(str)
        common  = [c for c in base_df.columns if c in new_df.columns]
        merged  = pd.concat([base_df, new_df[common]], ignore_index=True)
        merged.to_csv(target, index=False)
        logger.info(f'  {filename}: +{len(new_df)} {label} rows → {len(merged)} total')


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def merge_km_into_ztm(ztm_dir: Path, trains_dir: Path = None, wkd_dir: Path = None) -> Path:
    """
    Merge KM and WKD GTFS data into a new combined directory alongside ZTM data.

    The original ZTM directory is never modified.
    If the combined directory already exists, it is returned immediately.
    Delete the combined dir to force regeneration.

    Args:
        ztm_dir:    Extracted ZTM GTFS directory
        trains_dir: Optional path to existing polish_trains directory (auto-downloaded if None)
        wkd_dir:    Optional path to existing wkd directory (auto-downloaded if None)

    Returns:
        Path to combined directory ready for use as GTFS data_dir.
    """
    ztm_dir      = Path(ztm_dir)
    combined_dir = ztm_dir.parent / (ztm_dir.name + COMBINED_SUFFIX)

    if combined_dir.exists():
        logger.info(f'Combined directory already exists, reusing: {combined_dir}')
        return combined_dir

    if trains_dir is None:
        trains_dir = get_latest_polish_trains_dir()
    if wkd_dir is None:
        wkd_dir = get_latest_wkd_dir()

    logger.info(f'Creating combined GTFS at {combined_dir}')
    combined_dir.mkdir()

    # Seed with ZTM files
    for f in ztm_dir.glob('*.txt'):
        shutil.copy(f, combined_dir / f.name)

    # Append KM
    logger.info('Merging KM data...')
    _append_to_combined(combined_dir, extract_km_data(trains_dir), 'KM')

    # Append WKD
    logger.info('Merging WKD data...')
    _append_to_combined(combined_dir, extract_wkd_data(wkd_dir), 'WKD')

    logger.info('Combined GTFS (ZTM + KM + WKD) written successfully.')
    return combined_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from core.gtfs_downloader import GTFSDownloader

    ztm_dir  = GTFSDownloader().get_latest_data_dir()
    combined = merge_km_into_ztm(ztm_dir)
    print(f'\nCombined GTFS ready at: {combined}')
