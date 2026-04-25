"""
Module for parsing and filtering GTFS schedule data.
Called internally by route_line_builder.py — do not run directly.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, time
from typing import List, Tuple, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PEAK_HOURS, VEHICLE_TYPE_RULES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GTFSParser:
    """Parse and filter GTFS schedule data"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.shapes = None
        self.trips = None
        self.stop_times = None
        self.calendar = None
        
    def load_data(self):
        """Load all required GTFS files"""
        logger.info("Loading GTFS files...")
        
        self.shapes = pd.read_csv(self.data_dir / "shapes.txt")
        self.trips = pd.read_csv(self.data_dir / "trips.txt")
        self.stop_times = pd.read_csv(
            self.data_dir / "stop_times.txt",
            dtype={'stop_id': str}
        )

        # Warsaw GTFS uses calendar_dates.txt instead of calendar.txt
        if (self.data_dir / "calendar_dates.txt").exists():
            self.calendar = pd.read_csv(self.data_dir / "calendar_dates.txt")
            logger.info("Using calendar_dates.txt")
        else:
            self.calendar = pd.read_csv(self.data_dir / "calendar.txt")
            logger.info("Using calendar.txt")

        logger.info(f"Loaded {len(self.shapes)} shape points")
        logger.info(f"Loaded {len(self.trips)} trips")
        logger.info(f"Loaded {len(self.stop_times)} stop times")

    @staticmethod
    def classify_vehicle(route_id: str) -> str:
        """
        Classify vehicle type based on route_id

        Args:
            route_id: Route identifier

        Returns:
            Vehicle type: 'Tram', 'Bus', or 'Train'
        """
        route_id = str(route_id).strip()

        for vehicle_type, rule in VEHICLE_TYPE_RULES.items():
            if rule(route_id):
                return vehicle_type.capitalize()

        return 'Bus'  # Default

    @staticmethod
    def parse_gtfs_time(time_str: str) -> Optional[time]:
        """
        Parse GTFS time string, handling times >= 24:00:00

        Args:
            time_str: Time string in format HH:MM:SS

        Returns:
            Python time object, or None if invalid
        """
        try:
            parts = time_str.split(':')
            hour = int(parts[0])

            # Filter out times beyond 24 hours (next day)
            if hour >= 24:
                return None

            return datetime.strptime(time_str, '%H:%M:%S').time()
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def is_in_time_ranges(t: time, time_ranges: List[Tuple[time, time]]) -> bool:
        """
        Check if time falls within any of the specified ranges

        Args:
            t: Time to check
            time_ranges: List of (start, end) time tuples

        Returns:
            True if time is in any range
        """
        for start, end in time_ranges:
            if start <= end:
                if start <= t <= end:
                    return True
            else:  # Wraps around midnight
                if start <= t or t <= end:
                    return True
        return False

    def filter_by_date(self, target_date: str) -> List[str]:
        """
        Find all service_ids active on target date.

        Args:
            target_date: Date in format YYYYMMDD

        Returns:
            List of service_id strings active on that date.
            Multiple IDs are normal when data from several feeds is merged
            (e.g. ZTM 'PcS' + KM 'km_weekday').
        """
        target_date_int = int(target_date)

        # calendar_dates.txt has columns: service_id, date, exception_type
        matching_services = self.calendar[
            (self.calendar['date'] == target_date_int) &
            (self.calendar['exception_type'] == 1)  # 1 = service added
        ]['service_id'].tolist()

        if not matching_services:
            # If no exact match, fall back to the most common service_id
            logger.warning(f"No exact service found for {target_date}")
            available_dates = sorted(self.calendar['date'].unique())
            logger.info(f"Date range: {min(available_dates)} to {max(available_dates)}")

            most_common_service = self.calendar[
                self.calendar['exception_type'] == 1
            ]['service_id'].mode()

            if len(most_common_service) > 0:
                fallback = most_common_service.iloc[0]
                logger.warning(f"Using most common service_id as fallback: {fallback}")
                matching_services = [fallback]
            else:
                raise ValueError(f"No service found for date {target_date} and no fallback available")
        else:
            logger.info(f"Found {len(matching_services)} service_id(s) for {target_date}: {matching_services}")

        return matching_services

    def filter_by_time(self, time_ranges: List[Tuple[time, time]] = PEAK_HOURS):
        """
        Filter stop_times by time ranges

        Args:
            time_ranges: List of (start, end) time tuples
        """
        logger.info("Parsing and filtering stop times...")

        # Parse times
        self.stop_times['arrival_time_parsed'] = self.stop_times['arrival_time'].apply(
            self.parse_gtfs_time
        )
        self.stop_times['departure_time_parsed'] = self.stop_times['departure_time'].apply(
            self.parse_gtfs_time
        )

        # Drop invalid times
        initial_count = len(self.stop_times)
        self.stop_times = self.stop_times.dropna(
            subset=['arrival_time_parsed', 'departure_time_parsed']
        )

        # Filter by time ranges
        mask = self.stop_times['arrival_time_parsed'].apply(
            lambda t: self.is_in_time_ranges(t, time_ranges)
        )
        self.stop_times = self.stop_times[mask]

        logger.info(
            f"Filtered stop times: {initial_count} -> {len(self.stop_times)} "
            f"({len(self.stop_times)/initial_count*100:.1f}%)"
        )

    def prepare_trips(self, service_ids: List[str]):
        """
        Prepare trips data with vehicle classification.

        Args:
            service_ids: List of service IDs active on the target date.
                         ZTM trips embed the service_id as a substring of their
                         trip service_id (e.g. 'PcS' inside '2026-04-22:PcS').
                         KM (and other merged feeds) use the service_id directly.
                         Both cases are handled by a substring search.
        """
        logger.info("Preparing trips data...")

        # A trip matches if any active service_id appears as a substring of its service_id.
        # This handles both ZTM format ('2026-04-22:PcS' contains 'PcS')
        # and merged-feed format ('km_weekday' contains 'km_weekday').
        mask = self.trips['service_id'].apply(
            lambda sid: any(s in str(sid) for s in service_ids)
        )
        self.trips = self.trips[mask]

        if len(self.trips) == 0:
            logger.warning(f"No trips found for service_ids {service_ids}")
            logger.info(f"Sample service_ids in trips: {self.trips['service_id'].head().tolist()}")
        else:
            logger.info(f"Filtered to {len(self.trips)} trips")

        # Add vehicle type
        self.trips['vehicle'] = self.trips['route_id'].apply(self.classify_vehicle)

        vehicle_counts = self.trips['vehicle'].value_counts()
        logger.info(f"Vehicle distribution:\n{vehicle_counts}")

    def aggregate_trip_counts(self) -> pd.DataFrame:
        """
        Aggregate trip counts per shape and merge with shapes

        Returns:
            DataFrame with shapes and trip counts
        """
        logger.info("Aggregating trip counts...")

        # Get valid trips (those with stops in time range)
        valid_trip_ids = self.stop_times['trip_id'].unique()
        valid_trips = self.trips[self.trips['trip_id'].isin(valid_trip_ids)]

        logger.info(f"Valid trips after time filtering: {len(valid_trips)}")

        # Count trips per shape_id and get vehicle type
        trip_counts = valid_trips.groupby('shape_id').agg({
            'trip_id': 'count',
            'vehicle': 'first'
        }).reset_index()
        trip_counts.columns = ['shape_id', 'trip_count', 'vehicle']

        # Merge with shapes
        shapes_with_counts = pd.merge(
            self.shapes,
            trip_counts,
            on='shape_id',
            how='left'
        )

        # Fill missing values
        shapes_with_counts['trip_count'] = shapes_with_counts['trip_count'].fillna(0)
        shapes_with_counts['vehicle'] = shapes_with_counts['vehicle'].fillna('Unknown')

        logger.info(
            f"Created dataset with {shapes_with_counts['shape_id'].nunique()} unique shapes "
            f"and {shapes_with_counts['trip_count'].sum():.0f} total trips"
        )

        return shapes_with_counts

    def expand_metro_frequencies(self, target_date: str = None):
        """
        Expand frequency-based metro trips (M1/M2) into synthetic individual departures.

        Warsaw metro is defined in frequencies.txt with headways rather than explicit
        trip times. This method generates one synthetic trip per departure so the rest
        of the pipeline treats metro identically to buses/trams.

        target_date: YYYYMMDD string used to look up the correct metro service_id
                     (e.g. 'PcM' for Mon-Thu) from calendar_dates.txt.
                     Without this, all day patterns would be expanded simultaneously.

        Adds synthetic rows to self.trips and self.stop_times.
        """
        freq_path = self.data_dir / "frequencies.txt"
        if not freq_path.exists():
            logger.info("No frequencies.txt — skipping metro expansion")
            return

        frequencies = pd.read_csv(freq_path)
        metro_freq = frequencies[
            frequencies['trip_id'].str.startswith(('M1:', 'M2:'), na=False)
        ]
        if metro_freq.empty:
            logger.info("No metro entries in frequencies.txt")
            return

        # Find the metro-specific service_id active on target_date (e.g. 'PcM').
        # calendar_dates.txt has both the regular service (e.g. '2026-04-27:PcS')
        # and a separate metro entry (e.g. 'PcM') for the same date.
        active_metro_service_ids = set()
        if target_date is not None:
            calendar = pd.read_csv(self.data_dir / "calendar_dates.txt")
            date_int = int(target_date)
            active_ids = calendar[
                (calendar['date'] == date_int) &
                (calendar['exception_type'] == 1)
            ]['service_id'].tolist()
            # Metro service_ids are short day-pattern codes like 'PcM', 'NdM', 'SbM'
            active_metro_service_ids = {s for s in active_ids if not s.startswith('20')}
            logger.info(f"Active metro service_ids for {target_date}: {active_metro_service_ids}")

        # Load template trips from trips.txt and filter to active day pattern
        all_trips = pd.read_csv(self.data_dir / "trips.txt")
        template_ids = metro_freq['trip_id'].unique()
        metro_templates = all_trips[all_trips['trip_id'].isin(template_ids)].copy()

        if active_metro_service_ids:
            metro_templates = metro_templates[
                metro_templates['service_id'].isin(active_metro_service_ids)
            ]
            # Filter frequencies to only those template trips
            metro_freq = metro_freq[metro_freq['trip_id'].isin(metro_templates['trip_id'])]

        metro_templates = metro_templates.copy()
        metro_templates['vehicle'] = 'Metro'

        if metro_templates.empty:
            logger.warning("Metro template trips not found in trips.txt")
            return

        logger.info(
            f"Expanding {len(metro_freq)} metro frequency entries "
            f"for routes: {metro_templates['route_id'].unique().tolist()}"
        )

        def _secs(t):
            h, m, s = t.split(':')
            return int(h) * 3600 + int(m) * 60 + int(s)

        def _fmt(secs):
            h, rem = divmod(int(secs), 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        synthetic_trips = []
        synthetic_stops = []

        for _, freq in metro_freq.iterrows():
            tmpl = metro_templates[metro_templates['trip_id'] == freq['trip_id']]
            if tmpl.empty:
                continue
            tmpl = tmpl.iloc[0]

            start  = _secs(freq['start_time'])
            end    = _secs(freq['end_time'])
            headway = int(freq['headway_secs'])

            t = start
            while t < end:
                syn_id  = f"{freq['trip_id']}__{t}"
                dep_str = _fmt(t)
                synthetic_trips.append({
                    'trip_id':    syn_id,
                    'route_id':   tmpl['route_id'],
                    'service_id': tmpl.get('service_id', ''),
                    'shape_id':   tmpl.get('shape_id', ''),
                    'vehicle':    'Metro',
                })
                synthetic_stops.append({
                    'trip_id':        syn_id,
                    'stop_id':        'metro_0',
                    'arrival_time':   dep_str,
                    'departure_time': dep_str,
                    'stop_sequence':  1,
                })
                t += headway

        if not synthetic_trips:
            logger.warning("No synthetic metro departures generated")
            return

        self.trips = pd.concat(
            [self.trips, pd.DataFrame(synthetic_trips)], ignore_index=True
        )
        self.stop_times = pd.concat(
            [self.stop_times, pd.DataFrame(synthetic_stops)], ignore_index=True
        )
        logger.info(f"Generated {len(synthetic_trips)} synthetic metro departures")

    def process(self, target_date: str) -> pd.DataFrame:
        """
        Complete processing pipeline

        Args:
            target_date: Date to analyze in format YYYYMMDD

        Returns:
            DataFrame with processed shapes and trip counts
        """
        self.load_data()
        service_ids = self.filter_by_date(target_date)
        self.prepare_trips(service_ids)
        self.expand_metro_frequencies(target_date)
        self.filter_by_time()

        return self.aggregate_trip_counts()


if __name__ == "__main__":
    from gtfs_downloader import GTFSDownloader
    from config import ANALYSIS_DATE

    # Get data
    downloader = GTFSDownloader()
    data_dir = downloader.get_latest_data_dir()

    # Process
    parser = GTFSParser(data_dir)
    result = parser.process(ANALYSIS_DATE)

    print(f"\nProcessed {len(result)} shape points")
    print(f"Unique shapes: {result['shape_id'].nunique()}")
    print(f"Total trips: {result['trip_count'].sum():.0f}")