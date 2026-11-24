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
    result = data if isinstance(data, list) else data.get("result")
    if not isinstance(result, list):
        raise RuntimeError(f"API returned no vehicle list: {str(data)[:150]}")
    return result


RETRY_WAITS = [2, 5]   # seconds to wait before 2nd and 3rd attempt
LOG_PATH = DATA_DIR / "collect_errors.log"


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print("  " + line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
    except OSError:
        pass


def fetch_with_retry(vehicle_type: int) -> list[dict]:
    """Try up to len(RETRY_WAITS)+1 times; raise the last error if all fail."""
    for attempt, wait in enumerate(RETRY_WAITS + [None], start=1):
        try:
            return fetch(vehicle_type)
        except Exception as e:
            if wait is None:
                raise
            print(f"    attempt {attempt} failed ({type(e).__name__}), retrying in {wait}s", flush=True)
            time.sleep(wait)


def collect(vehicle_types: list[int], stop_at: datetime | None, interval: int):
    DATA_DIR.mkdir(exist_ok=True)

    def open_day(day: str):
        path = DATA_DIR / f"positions_{day}.csv"
        is_new = not path.exists()
        fh = open(path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if is_new:
            w.writeheader()
        print(f"Saving to: {path}", flush=True)
        return fh, w, path

    total_rows = 0
    poll_count = 0
    last_fix: dict[tuple, str] = {}   # (type, vehicle) -> last gps_time written (dedupe repeats)
    down_since: dict[str, datetime] = {}   # label -> start of current outage

    date_str = datetime.now().strftime("%Y%m%d")
    f, writer, csv_path = open_day(date_str)
    print(f"Polling every {interval}s. Press Ctrl+C to stop.\n", flush=True)

    with f:
        try:
            while True:
                now = datetime.now()
                if stop_at and now >= stop_at:
                    print(f"\nTime limit reached.")
                    break

                if now.strftime("%Y%m%d") != date_str:   # midnight rollover
                    f.close()
                    date_str = now.strftime("%Y%m%d")
                    f, writer, csv_path = open_day(date_str)

                poll_str   = now.strftime("%Y-%m-%d %H:%M:%S")
                fresh_rows = 0

                for vtype in vehicle_types:
                    label = "Tram" if vtype == 2 else "Bus"
                    try:
                        vehicles = fetch_with_retry(vtype)
                    except Exception as e:
                        if label not in down_since:
                            down_since[label] = now
                            log(f"{label} OUTAGE started: {type(e).__name__}: {e}")
                        continue
                    if label in down_since:
                        mins = (now - down_since.pop(label)).total_seconds() / 60
                        log(f"{label} back online after {mins:.1f} min")

                    for v in vehicles:
                        try:
                            gps_time = datetime.strptime(v["Time"], "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            continue
                        if (now - gps_time).total_seconds() > STALE:
                            continue  # skip stale/parked vehicles
                        key = (label, v.get("VehicleNumber", ""))
                        if last_fix.get(key) == v["Time"]:
                            continue  # same GPS fix as last poll
                        last_fix[key] = v["Time"]

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
                print(f"  [{poll_str}]  fresh vehicles: {fresh_rows:>4}  total rows so far: {total_rows}", flush=True)

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

    while True:   # restart on any unexpected crash; returns normally on Ctrl+C / time limit
        try:
            collect(vtypes, stop_at, INTERVAL)
            break
        except Exception as e:
            log(f"Collector crashed ({type(e).__name__}: {e}), restarting in 10s")
            time.sleep(10)


if __name__ == "__main__":
    main()
