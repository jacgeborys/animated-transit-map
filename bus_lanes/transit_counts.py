"""
Step B — Count buses and trams per shape for one Wednesday.

For every shape used on the analysis date and every stop along it, counts how many
trips *pass that stop* in the AM peak, PM peak and over the whole day. Using the
time at each stop (not the trip start time) keeps peak counts correct for long lines.

Outputs (bus_lanes/_data/):
    shape_profiles.pkl   DataFrame: shape_id, mode, dist_km, am_n, pm_n, day_n
    analysis_date.txt    the chosen date (YYYYMMDD)
"""
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bl_config import (ANALYSIS_DATE, DATA_DIR, PEAK_END_H, PEAK_START_H, PM_END_H,
                       PM_START_H, RANDOM_SEED, RAW_DIR)

MODES = {3: "bus", 0: "tram"}


def latest_feed() -> Path:
    dirs = sorted(d for d in RAW_DIR.iterdir()
                  if d.is_dir() and d.name.startswith("warsaw_gtfs_") and "_with_" not in d.name)
    return dirs[-1]


def pick_wednesday(cal: pd.DataFrame) -> str:
    if ANALYSIS_DATE:
        return ANALYSIS_DATE
    dates = sorted({str(d) for d in cal.date.unique()})
    weds = [d for d in dates if datetime.strptime(d, "%Y%m%d").weekday() == 2]
    # only "regular weekday" service (ZTM code PcS = Mon–Thu school-term)
    weds = [d for d in weds
            if cal[(cal.date == int(d))].service_id.str.contains("PcS").any()]
    rng = random.Random(RANDOM_SEED)
    choice = rng.choice(weds)
    print(f"Wednesdays in feed: {', '.join(weds)} → picked {choice}")
    return choice


def to_secs(s: pd.Series) -> np.ndarray:
    parts = s.str.split(":", expand=True).astype(int)
    return (parts[0] * 3600 + parts[1] * 60 + parts[2]).to_numpy()


def main():
    feed = latest_feed()
    print(f"Feed: {feed.name}")
    cal = pd.read_csv(feed / "calendar_dates.txt")
    date = pick_wednesday(cal)
    services = set(cal[(cal.date == int(date)) & (cal.exception_type == 1)].service_id)

    routes = pd.read_csv(feed / "routes.txt", dtype={"route_id": str})
    routes = routes[routes.route_type.isin(MODES)]
    trips = pd.read_csv(feed / "trips.txt", dtype={"route_id": str, "shape_id": str},
                        usecols=["trip_id", "route_id", "service_id", "shape_id"])
    trips = trips[trips.service_id.isin(services) & trips.route_id.isin(routes.route_id)]
    trips = trips.merge(routes[["route_id", "route_type"]], on="route_id")
    trips["mode"] = trips.route_type.map(MODES)
    print(f"Trips on {date}: " + ", ".join(f"{m} {n:,}" for m, n in trips["mode"].value_counts().items()))

    wanted = set(trips.trip_id)
    chunks = []
    for ch in pd.read_csv(feed / "stop_times.txt", chunksize=2_000_000,
                          usecols=["trip_id", "stop_sequence", "departure_time", "shape_dist_traveled"]):
        chunks.append(ch[ch.trip_id.isin(wanted)])
    st = pd.concat(chunks, ignore_index=True)
    st["t"] = to_secs(st.departure_time) % 86400  # 24:xx night trips wrap to early morning
    st = st.merge(trips[["trip_id", "shape_id", "mode"]], on="trip_id")
    print(f"Stop times loaded: {len(st):,}")

    am = (st.t >= PEAK_START_H * 3600) & (st.t < PEAK_END_H * 3600)
    pm = (st.t >= PM_START_H * 3600) & (st.t < PM_END_H * 3600)
    st["am"] = am.astype(int)
    st["pm"] = pm.astype(int)

    # Per shape and stop position: number of trips passing (trips on one shape share stops)
    prof = (st.groupby(["shape_id", "mode", "shape_dist_traveled"])
              .agg(am_n=("am", "sum"), pm_n=("pm", "sum"), day_n=("trip_id", "count"))
              .reset_index()
              .rename(columns={"shape_dist_traveled": "dist_km"})
              .sort_values(["shape_id", "dist_km"]))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prof.to_pickle(DATA_DIR / "shape_profiles.pkl")
    (DATA_DIR / "analysis_date.txt").write_text(f"{date}\n{feed.name}\n", encoding="utf-8")
    print(f"Saved profiles for {prof.shape_id.nunique():,} shapes ({len(prof):,} rows)")


if __name__ == "__main__":
    main()
