"""
Exploration script for Warsaw ZTM live vehicle position API.

API: dane.um.warszawa.pl
- Free, requires API key from https://dane.um.warszawa.pl
- Returns tram/bus GPS positions updated every ~10 seconds
- NO historic data — positions are only available in real time

Usage:
    python live/explore_api.py
"""

import requests
from datetime import datetime

try:
    from live.secrets import API_KEY
except ImportError:
    from secrets import API_KEY

URL = "https://dane.um.warszawa.pl/api/action/get_ztm_lokalizacja_pojazdow"


def fetch_positions(vehicle_type: int) -> list[dict]:
    """
    Fetch live vehicle positions.
    vehicle_type: 1 = buses, 2 = trams
    """
    resp = requests.post(
        URL,
        headers={"Authorization": API_KEY, "Content-Type": "application/json"},
        json={"type": vehicle_type},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "result" in data:
        result = data["result"]
        if isinstance(result, list):
            return result
        print(f"API error: {data}")
        return []

    print(f"Unexpected response: {str(data)[:200]}")
    return []


def main():
    now = datetime.now()
    print(f"Fetching live positions at {now.strftime('%H:%M:%S')}...\n")

    for label, vtype in [("Trams", 2), ("Buses", 1)]:
        vehicles = fetch_positions(vtype)
        print(f"=== {label} — {len(vehicles)} vehicles visible ===")

        if not vehicles:
            print("  (none)")
            continue

        # Show available fields from first record
        if label == "Trams":
            print("Fields:", list(vehicles[0].keys()))
            print()

        # Sample records
        for v in vehicles[:5]:
            print(f"  Line {v.get('Lines', '?'):>4}  brigade {v.get('Brigade', '?')}  "
                  f"vehicle {v.get('VehicleNumber', '?')}  "
                  f"({v.get('Lat', '?')}, {v.get('Lon', '?')})  "
                  f"last seen: {v.get('Time', '?')}")

        # Data freshness
        timestamps = []
        for v in vehicles:
            try:
                timestamps.append(datetime.strptime(v["Time"], "%Y-%m-%d %H:%M:%S"))
            except Exception:
                pass

        if timestamps:
            newest = max(timestamps)
            oldest = min(timestamps)
            stale = sum(1 for t in timestamps if (now - t).total_seconds() > 120)
            print(f"\n  Newest: {newest}  ({int((now - newest).total_seconds())}s ago)")
            print(f"  Oldest: {oldest}  ({int((now - oldest).total_seconds())}s ago)")
            print(f"  Stale (>2 min): {stale} / {len(timestamps)}")
        print()

    print("=== Notes on historic data ===")
    print("  This API has NO history endpoint. To get a full day:")
    print("  - Poll every ~30s and save to a local CSV.")
    print("  - One day of tram data ≈ 2-5 MB.")
    print("  - Delay = compare recorded position vs GTFS schedule.")


if __name__ == "__main__":
    main()
