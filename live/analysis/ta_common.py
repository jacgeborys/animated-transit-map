"""Shared loaders for the tram analysis (cached as parquet in _cache/)."""

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

import ta_config as C


def gtfs_dir():
    dirs = sorted(d for d in C.RAW_DIR.glob("warsaw_gtfs_*") if d.is_dir() and "with" not in d.name)
    return dirs[-1]


def hms(s: pd.Series) -> np.ndarray:
    p = s.str.split(":", expand=True).astype(int)
    return (p[0] * 3600 + p[1] * 60 + p[2]).to_numpy()


def fleet_lookup(numbers: pd.Series) -> pd.DataFrame:
    n = pd.to_numeric(numbers, errors="coerce")
    out = pd.DataFrame({"model": "unknown", "places": np.nan, "length_m": np.nan}, index=numbers.index)
    for lo, hi, typ, places, length, _ in C.FLEET:
        k = n.between(lo, hi)
        out.loc[k, ["model", "places", "length_m"]] = typ, places, length
    return out


def load_gtfs():
    """Tram trips of the day, their stop_times (sec since midnight, m along shape) and shapes (EPSG:2180)."""
    C.CACHE.mkdir(exist_ok=True)
    f_trips, f_st, f_shp = (C.CACHE / f"{n}_{C.DATE}.parquet" for n in ("trips", "stop_times", "shapes"))
    if f_trips.exists() and f_st.exists() and f_shp.exists():
        return pd.read_parquet(f_trips), pd.read_parquet(f_st), pd.read_parquet(f_shp)

    g = gtfs_dir()
    iso = f"{C.DATE[:4]}-{C.DATE[4:6]}-{C.DATE[6:]}"
    routes = pd.read_csv(g / "routes.txt", dtype=str)
    trams = set(routes.loc[routes.route_type == "0", "route_id"])
    trips = pd.read_csv(g / "trips.txt", dtype=str)
    trips = trips[trips.service_id.str.startswith(iso) & trips.route_id.isin(trams)].copy()
    if trips.empty:
        raise SystemExit(f"{g.name} has no tram trips on {iso}")
    keep = set(trips.trip_id)
    st = pd.concat(ch[ch.trip_id.isin(keep)] for ch in pd.read_csv(
        g / "stop_times.txt", dtype={"trip_id": str, "stop_id": str}, chunksize=2_000_000,
        usecols=["trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time", "shape_dist_traveled"]))
    st["arr_s"], st["dep_s"] = hms(st.arrival_time), hms(st.departure_time)
    st["m"] = st.shape_dist_traveled.astype(float) * 1000
    st = st.drop(columns=["arrival_time", "departure_time", "shape_dist_traveled"]).sort_values(["trip_id", "stop_sequence"])

    stops = pd.read_csv(g / "stops.txt", dtype={"stop_id": str})
    st = st.merge(stops[["stop_id", "stop_name", "street_name"]], on="stop_id", how="left")

    shp = pd.read_csv(g / "shapes.txt", dtype={"shape_id": str})
    shp = shp[shp.shape_id.isin(set(trips.shape_id))].sort_values(["shape_id", "shape_pt_sequence"])
    xy = gpd.points_from_xy(shp.shape_pt_lon, shp.shape_pt_lat, crs=4326).to_crs(C.CRS)
    shp = pd.DataFrame({"shape_id": shp.shape_id.to_numpy(), "x": xy.x, "y": xy.y})

    e = st.groupby("trip_id").agg(start_s=("dep_s", "first"), end_s=("arr_s", "last"),
                                  first_stop=("stop_name", "first"), last_stop=("stop_name", "last"), n_stops=("stop_id", "size"))
    trips = trips.join(e, on="trip_id")
    trips = trips[["trip_id", "route_id", "shape_id", "trip_headsign", "direction_id", "block_short_name",
                   "fleet_type", "start_s", "end_s", "first_stop", "last_stop", "n_stops"]]
    trips.to_parquet(f_trips); st.to_parquet(f_st); shp.to_parquet(f_shp)
    return trips, st, shp


def shape_lines(shp: pd.DataFrame) -> dict:
    return {sid: shapely.linestrings(g[["x", "y"]].to_numpy()) for sid, g in shp.groupby("shape_id")}


def load_fixes() -> pd.DataFrame:
    """Tram GPS fixes of the day: line, brigade, vehicle, t (sec since midnight), x, y."""
    f = C.CACHE / f"tram_fixes_{C.DATE}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    cols = ["vehicle_type", "Lines", "Brigade", "VehicleNumber", "Lat", "Lon", "gps_time"]
    df = pd.concat(c[c.vehicle_type == "Tram"] for c in pd.read_csv(
        C.LIVE_DATA / f"positions_{C.DATE}.csv", usecols=cols, dtype=str, chunksize=2_000_000, on_bad_lines="skip"))
    df["gps_time"] = pd.to_datetime(df.gps_time, errors="coerce")
    df = df.dropna(subset=["gps_time", "Lat", "Lon"])
    day0 = pd.Timestamp(f"{C.DATE[:4]}-{C.DATE[4:6]}-{C.DATE[6:]}")
    pts = gpd.points_from_xy(df.Lon.astype(float), df.Lat.astype(float), crs=4326).to_crs(C.CRS)
    out = pd.DataFrame({"line": df.Lines.to_numpy(), "brigade": df.Brigade.to_numpy(), "veh": df.VehicleNumber.to_numpy(),
                        "t": (df.gps_time - day0).dt.total_seconds().to_numpy(), "x": pts.x, "y": pts.y})
    out = out.drop_duplicates(["veh", "t"]).sort_values(["veh", "t"]).reset_index(drop=True)
    C.CACHE.mkdir(exist_ok=True)
    out.to_parquet(f)
    return out


def peak(t_s) -> np.ndarray:
    h = np.asarray(t_s) / 3600
    return np.any([(h >= a) & (h < b) for a, b in C.PEAK_HOURS], axis=0)


def clock(s) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}"
