"""
Koleje Mazowieckie GTFS Merger
================================
Downloads polish_trains.zip from mkuran.pl, filters to KM routes only,
and merges into a combined directory alongside ZTM Warsaw GTFS data.

Pipeline position: run once after gtfs_downloader.py.
Pass the returned combined_dir to RouteLineBuilder instead of the raw ZTM dir.

Usage:
    from core.km_gtfs_merger import merge_km_into_ztm
    combined_dir = merge_km_into_ztm(ztm_dir, ANALYSIS_DATE)
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
from config import POLISH_TRAINS_URL, RAW_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KM_PREFIX = 'km_'


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_polish_trains(output_dir: Path = RAW_DIR) -> Path:
    """Download and extract polish_trains.zip. Returns the extracted directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_path = output_dir / f'polish_trains_{timestamp}.zip'
    extract_dir = output_dir / f'polish_trains_{timestamp}'

    logger.info(f'Downloading polish_trains from {POLISH_TRAINS_URL}')
    response = requests.get(POLISH_TRAINS_URL, stream=True, timeout=60)
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

    logger.info(f'Polish trains GTFS extracted to {extract_dir}')
    return extract_dir


def get_latest_polish_trains_dir(raw_dir: Path = RAW_DIR) -> Path:
    """Return the most recent polish_trains directory, downloading if none exists."""
    raw_dir = Path(raw_dir)
    dirs = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith('polish_trains_')],
        key=lambda x: x.name,
        reverse=True,
    )
    if not dirs:
        return download_polish_trains(raw_dir)
    logger.info(f'Using existing polish_trains data from {dirs[0]}')
    return dirs[0]


# ---------------------------------------------------------------------------
# KM extraction
# ---------------------------------------------------------------------------

def _find_km_agency_id(trains_dir: Path) -> str:
    """Return the agency_id for Koleje Mazowieckie from agency.txt."""
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


def _expand_calendar_txt(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Convert calendar.txt (day-of-week patterns) to calendar_dates.txt format."""
    day_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    rows = []
    for _, row in calendar_df.iterrows():
        start = datetime.strptime(str(int(row['start_date'])), '%Y%m%d')
        end = datetime.strptime(str(int(row['end_date'])), '%Y%m%d')
        current = start
        while current <= end:
            if int(row[day_cols[current.weekday()]]) == 1:
                rows.append({
                    'service_id': row['service_id'],
                    'date': int(current.strftime('%Y%m%d')),
                    'exception_type': 1,
                })
            current += timedelta(days=1)
    return pd.DataFrame(rows)


def extract_km_data(trains_dir: Path) -> dict:
    """
    Extract KM-only data from polish_trains directory and apply km_ prefix to all IDs.

    Returns dict with DataFrames: trips, shapes, stop_times, stops, calendar_dates.
    Route IDs are kept as-is (R1, R2, ...) since they don't collide with ZTM IDs.
    """
    trains_dir = Path(trains_dir)
    km_agency_id = _find_km_agency_id(trains_dir)

    # Routes → get KM route_ids
    routes = pd.read_csv(trains_dir / 'routes.txt', dtype=str)
    km_route_ids = set(routes.loc[routes['agency_id'] == km_agency_id, 'route_id'])
    logger.info(f'KM routes ({len(km_route_ids)}): {sorted(km_route_ids)}')

    # Trips
    trips = pd.read_csv(trains_dir / 'trips.txt', dtype=str)
    km_trips = trips[trips['route_id'].isin(km_route_ids)].copy()
    km_trip_ids = set(km_trips['trip_id'])
    km_shape_ids = set(km_trips['shape_id'].dropna())
    km_service_ids = set(km_trips['service_id'])
    logger.info(f'KM trips: {len(km_trips)}, shapes: {len(km_shape_ids)}, services: {len(km_service_ids)}')

    # Shapes
    shapes = pd.read_csv(trains_dir / 'shapes.txt', dtype=str)
    km_shapes = shapes[shapes['shape_id'].isin(km_shape_ids)].copy()

    # Stop times
    stop_times = pd.read_csv(trains_dir / 'stop_times.txt', dtype=str)
    km_stop_times = stop_times[stop_times['trip_id'].isin(km_trip_ids)].copy()
    km_stop_ids = set(km_stop_times['stop_id'])

    # Stops
    stops = pd.read_csv(trains_dir / 'stops.txt', dtype=str)
    km_stops = stops[stops['stop_id'].isin(km_stop_ids)].copy()

    # Calendar — handle both calendar.txt and calendar_dates.txt
    if (trains_dir / 'calendar_dates.txt').exists():
        cal_dates_raw = pd.read_csv(trains_dir / 'calendar_dates.txt', dtype=str)
        cal_dates_raw['date'] = cal_dates_raw['date'].astype(int)
        cal_dates_raw['exception_type'] = cal_dates_raw['exception_type'].astype(int)
        km_cal_dates = cal_dates_raw[cal_dates_raw['service_id'].isin(km_service_ids)].copy()
        logger.info(f'KM calendar entries from calendar_dates.txt: {len(km_cal_dates)}')
    elif (trains_dir / 'calendar.txt').exists():
        calendar = pd.read_csv(trains_dir / 'calendar.txt', dtype=str)
        km_calendar = calendar[calendar['service_id'].isin(km_service_ids)].copy()
        km_cal_dates = _expand_calendar_txt(km_calendar)
        logger.info(f'KM calendar entries expanded from calendar.txt: {len(km_cal_dates)}')
    else:
        raise FileNotFoundError('Neither calendar.txt nor calendar_dates.txt found in polish_trains')

    # Apply km_ prefix
    km_trips['trip_id'] = KM_PREFIX + km_trips['trip_id']
    km_trips['shape_id'] = KM_PREFIX + km_trips['shape_id']
    km_trips['service_id'] = KM_PREFIX + km_trips['service_id']

    km_shapes['shape_id'] = KM_PREFIX + km_shapes['shape_id']

    km_stop_times['trip_id'] = KM_PREFIX + km_stop_times['trip_id']
    km_stop_times['stop_id'] = KM_PREFIX + km_stop_times['stop_id']

    km_stops['stop_id'] = KM_PREFIX + km_stops['stop_id']

    km_cal_dates['service_id'] = KM_PREFIX + km_cal_dates['service_id'].astype(str)

    return {
        'trips': km_trips,
        'shapes': km_shapes,
        'stop_times': km_stop_times,
        'stops': km_stops,
        'calendar_dates': km_cal_dates,
    }


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_km_into_ztm(ztm_dir: Path, trains_dir: Path = None) -> Path:
    """
    Merge KM GTFS data into a new combined directory alongside ZTM data.

    The original ZTM directory is never modified.
    If the combined directory already exists, it is returned immediately.

    Args:
        ztm_dir:    Extracted ZTM GTFS directory (e.g. data/raw/warsaw_gtfs_20260422_200502)
        trains_dir: Optional path to already-downloaded polish_trains directory.
                    If None, the latest existing one is used (or a fresh download).

    Returns:
        Path to combined directory ready for use as GTFS data_dir.
    """
    ztm_dir = Path(ztm_dir)
    combined_dir = ztm_dir.parent / (ztm_dir.name + '_with_km')

    if combined_dir.exists():
        logger.info(f'Combined directory already exists, reusing: {combined_dir}')
        return combined_dir

    if trains_dir is None:
        trains_dir = get_latest_polish_trains_dir()

    logger.info(f'Creating combined GTFS at {combined_dir}')
    combined_dir.mkdir()

    # Copy all ZTM files verbatim
    for f in ztm_dir.glob('*.txt'):
        shutil.copy(f, combined_dir / f.name)

    km_data = extract_km_data(trains_dir)

    # Files to append KM rows into
    file_map = {
        'trips':          'trips.txt',
        'shapes':         'shapes.txt',
        'stop_times':     'stop_times.txt',
        'stops':          'stops.txt',
        'calendar_dates': 'calendar_dates.txt',
    }

    for key, filename in file_map.items():
        target = combined_dir / filename
        ztm_df = pd.read_csv(target, dtype=str)
        km_df = km_data[key].astype(str)

        # Only keep columns that exist in the ZTM file to avoid schema mismatches
        common_cols = [c for c in ztm_df.columns if c in km_df.columns]
        merged = pd.concat([ztm_df, km_df[common_cols]], ignore_index=True)
        merged.to_csv(target, index=False)
        logger.info(f'  {filename}: {len(ztm_df)} ZTM + {len(km_df)} KM = {len(merged)} rows')

    logger.info('Combined GTFS written successfully.')
    return combined_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from config import RAW_DIR
    from core.gtfs_downloader import GTFSDownloader

    ztm_dir = GTFSDownloader().get_latest_data_dir()
    combined = merge_km_into_ztm(ztm_dir)
    print(f'\nCombined GTFS ready at: {combined}')
