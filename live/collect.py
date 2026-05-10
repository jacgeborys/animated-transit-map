"""
Collector for Warsaw ZTM live vehicle positions.

Polls the API every INTERVAL seconds, keeps only fresh positions,
and appends to a daily CSV in live/data/.

Usage:
    python live/collect.py              # runs until Ctrl+C
    python live/collect.py --minutes 5  # runs for 5 minutes
    python live/collect.py --type 2     # trams only (1=bus, 2=tram, default=both)
"""

import argparse
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

try:
    from live.api_keys import API_KEY
except ImportError:
    from api_keys import API_KEY

URL      = "https://dane.um.warszawa.pl/api/action/get_ztm_lokalizacja_pojazdow"
INTERVAL = 10   # seconds between polls
STALE    = 60   # seconds — discard positions older than this
DATA_DIR = Path(__file__).parent / "data"
FIELDNAMES = ["poll_time", "vehicle_type", "Lines", "Brigade", "VehicleNumber", "Lat", "Lon", "gps_time"]


def fetch(vehicle_type: int) -> list[dict]:
    resp = requests.post(
        URL,
        headers={"Authorization": API_KEY, "Content-Type": "application/json"},
        json={"type": vehicle_type},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else data.get("result", [])


def collect(vehicle_types: list[int], stop_at: datetime | None, interval: int):
    DATA_DIR.mkdir(exist_ok=True)
    date_str  = datetime.now().strftime("%Y%m%d")
    csv_path  = DATA_DIR / f"positions_{date_str}.csv"
    is_new    = not csv_path.exists()

    total_rows = 0
    poll_count = 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()

        print(f"Saving to: {csv_path}")
        print(f"Polling every {interval}s. Press Ctrl+C to stop.\n")

        try:
            while True:
                now = datetime.now()
                if stop_at and now >= stop_at:
                    print(f"\nTime limit reached.")
                    break

                poll_str   = now.strftime("%Y-%m-%d %H:%M:%S")
                fresh_rows = 0

                for vtype in vehicle_types:
                    label = "Tram" if vtype == 2 else "Bus"
                    try:
                        vehicles = fetch(vtype)
                    except Exception as e:
                        print(f"  [{poll_str}] {label} fetch error: {e}")
                        continue

                    for v in vehicles:
                        try:
                            gps_time = datetime.strptime(v["Time"], "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            continue
                        if (now - gps_time).total_seconds() > STALE:
                            continue  # skip stale/parked vehicles

                        writer.writerow({
                            "poll_time":    poll_str,
                            "vehicle_type": label,
                            "Lines":        v.get("Lines", ""),
                            "Brigade":      v.get("Brigade", ""),
                            "VehicleNumber":v.get("VehicleNumber", ""),
                            "Lat":          v.get("Lat", ""),
                            "Lon":          v.get("Lon", ""),
                            "gps_time":     v["Time"],
                        })
                        fresh_rows += 1

                f.flush()
                total_rows += fresh_rows
                poll_count += 1
                print(f"  [{poll_str}]  fresh vehicles: {fresh_rows:>4}  total rows so far: {total_rows}")

                next_poll = now + timedelta(seconds=interval)
                sleep_secs = (next_poll - datetime.now()).total_seconds()
                if sleep_secs > 0:
                    time.sleep(sleep_secs)

        except KeyboardInterrupt:
            print(f"\nStopped by user.")

    print(f"\nDone. {poll_count} polls, {total_rows} rows written to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=None,
                        help="Stop after this many minutes (default: run forever)")
    parser.add_argument("--type", type=int, choices=[1, 2], default=None,
                        help="1=buses only, 2=trams only (default: both)")
    args = parser.parse_args()

    stop_at = datetime.now() + timedelta(minutes=args.minutes) if args.minutes else None
    vtypes  = [args.type] if args.type else [1, 2]
    label   = {1: "buses", 2: "trams", None: "buses + trams"}[args.type]

    print(f"Collecting {label}" + (f" for {args.minutes} minutes" if args.minutes else " until Ctrl+C"))
    print(f"Freshness threshold: positions older than {STALE}s are skipped\n")

    collect(vtypes, stop_at, INTERVAL)


if __name__ == "__main__":
    main()
