"""
Exploration script for Warsaw ZTM live vehicle position API.

API: api.um.warszawa.pl
- Free, requires registration for an API key
- Register at: https://api.um.warszawa.pl (click "Zarejestruj się")
- Returns tram/bus GPS positions updated every ~15 seconds
- NO historic data — positions are only available in real time

Usage:
    Set API_KEY below, then run:
        python live/explore_api.py
"""

import requests
from datetime import datetime
from pprint import pprint

try:
    from live.secrets import API_KEY
except ImportError:
    from secrets import API_KEY

# Resource IDs
TRAMS_RESOURCE = "c7238cfe-8b1f-4c38-bb4a-de386db7e776"
BUSES_RESOURCE = "f2e5503e-927d-4ad3-9500-4ab9e55deb59"

BASE_URL = "https://api.um.warszawa.pl/api/action/busestrams_get/"


def fetch_positions(vehicle_type: int) -> list[dict]:
    """
    Fetch live vehicle positions.
    vehicle_type: 1 = buses, 2 = trams
    """
    params = {
        "resource_id": TRAMS_RESOURCE if vehicle_type == 2 else BUSES_RESOURCE,
        "apikey": API_KEY,
        "type": vehicle_type,
    }
    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "result" not in data:
        print("Unexpected response structure:")
        pprint(data)
        return []

    result = data["result"]
    if isinstance(result, str):
        print(f"API error message: {result}")
        return []

    return result


def main():
    print(f"Fetching live tram positions at {datetime.now().strftime('%H:%M:%S')}...\n")

    trams = fetch_positions(vehicle_type=2)

    if not trams:
        print("No data returned.")
        return

    print(f"Total trams visible right now: {len(trams)}")
    print()

    # Show field names from first record
    print("=== Available fields ===")
    first = trams[0]
    for key, value in first.items():
        print(f"  {key}: {value!r}")

    print()

    # Show a few sample records
    print("=== Sample records (first 5) ===")
    for t in trams[:5]:
        line    = t.get("Lines", "?")
        lat     = t.get("Lat", "?")
        lon     = t.get("Lon", "?")
        brigade = t.get("Brigade", "?")
        time    = t.get("Time", "?")
        vehicle = t.get("VehicleNumber", "?")
        print(f"  Line {line:>4}  brigade {brigade}  vehicle {vehicle}  "
              f"({lat}, {lon})  last seen: {time}")

    print()

    # Check data freshness
    timestamps = []
    for t in trams:
        try:
            ts = datetime.strptime(t["Time"], "%Y-%m-%d %H:%M:%S")
            timestamps.append(ts)
        except Exception:
            pass

    if timestamps:
        oldest = min(timestamps)
        newest = max(timestamps)
        now = datetime.now()
        print(f"=== Data freshness ===")
        print(f"  Newest position:  {newest}  ({(now - newest).seconds}s ago)")
        print(f"  Oldest position:  {oldest}  ({(now - oldest).seconds}s ago)")
        print()
        stale = [t for t in timestamps if (now - t).seconds > 120]
        print(f"  Vehicles with position older than 2 min: {len(stale)} / {len(timestamps)}")

    print()
    print("=== Notes on historic data ===")
    print("  This API does NOT store history. To get a full day:")
    print("  - You must poll every ~30s and save to a local file yourself.")
    print("  - One day of tram data ≈ 1-5 MB (plain text CSV).")
    print("  - Delay = compare recorded position timestamp vs GTFS schedule.")


if __name__ == "__main__":
    main()
