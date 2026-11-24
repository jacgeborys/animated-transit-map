"""
Where do trams and buses get stuck?  (first pass, no schedule comparison)

Method
  1. Each GPS fix is matched to its OWN line's GTFS shapes (today's trips).
     Fixes > OFF_ROUTE_M from the line (depots, detours, GPS junk) are dropped.
  2. Consecutive fixes of one vehicle on one line form intervals (MIN_DT..MAX_DT s).
     Interval speed < STUCK_KMH  ->  "stuck" time.
  3. Stuck time that is normal operation is removed:
       - near a stop the line serves (STOP_M: passengers boarding),
       - near a terminus of the line (TERMINUS_M: layover loops),
       - stand-stills longer than LAYOVER_MIN in one go (layover / breakdown).
     What remains is queuing at lights, junctions, traffic jams.
  4. Stuck minutes are summed on a GRID_M grid, per vehicle type.

Output (live/data/):
  stuck_cells_<date>.gpkg  — grid cells: tram_min, bus_min, total_min, vehicles,
                             top_lines, peak_hour, near_stop (nearest stop name)
  stuck_<date>.png         — overview map

Usage:
  python live/stuck_points.py                 # latest positions_*.csv
  python live/stuck_points.py --date 20261124
Safe to run while collect.py is still writing (reads a snapshot).
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

ROOT     = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR  = ROOT / "data" / "raw"
OSM_ROADS = Path(r"D:\QGIS\mapy_warszawy_misc\data\osm\roads.shp")
CRS = 2180  # PL-1992, metres

STUCK_KMH   = 5.0
MIN_DT, MAX_DT = 5, 90          # s between fixes
OFF_ROUTE_M = 40
STOP_M      = {"Bus": 35, "Tram": 50}
TERMINUS_M  = 150
LAYOVER_MIN = 10
GRID_M      = 50


# ---------------------------------------------------------------- GTFS
def latest_gtfs() -> Path:
    dirs = sorted(d for d in RAW_DIR.glob("warsaw_gtfs_*") if d.is_dir() and "with" not in d.name)
    return dirs[-1]


def load_gtfs(gtfs: Path, day: str):
    """Per route: union of shapes, MultiPoint of served stops, MultiPoint of termini."""
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    trips = pd.read_csv(gtfs / "trips.txt", dtype=str, usecols=["trip_id", "route_id", "service_id", "shape_id"])
    trips = trips[trips.service_id.str.startswith(iso)]
    if trips.empty:
        raise SystemExit(f"GTFS {gtfs.name} has no trips for {iso} — download a fresh feed.")
    print(f"GTFS {gtfs.name}: {len(trips)} trips on {iso}")

    stops = pd.read_csv(gtfs / "stops.txt", dtype={"stop_id": str})
    stops_g = gpd.GeoDataFrame(stops[["stop_id", "stop_name", "location_type"]],
                               geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat, crs=4326)).to_crs(CRS)
    stop_geom = dict(zip(stops_g.stop_id, stops_g.geometry))

    # stop_times is big: stream it, keep today's trips only
    keep = set(trips.trip_id)
    parts = []
    for ch in pd.read_csv(gtfs / "stop_times.txt", dtype={"trip_id": str, "stop_id": str},
                          usecols=["trip_id", "stop_sequence", "stop_id"], chunksize=2_000_000):
        parts.append(ch[ch.trip_id.isin(keep)])
    st = pd.concat(parts).merge(trips[["trip_id", "route_id"]], on="trip_id")
    st = st.sort_values(["trip_id", "stop_sequence"])
    ends = pd.concat([st.groupby("trip_id").head(1), st.groupby("trip_id").tail(1)])

    shp = pd.read_csv(gtfs / "shapes.txt", dtype={"shape_id": str})
    shp = shp[shp.shape_id.isin(set(trips.shape_id))].sort_values(["shape_id", "shape_pt_sequence"])
    pts = gpd.points_from_xy(shp.shape_pt_lon, shp.shape_pt_lat, crs=4326).to_crs(CRS)
    shp["x"], shp["y"] = pts.x, pts.y
    shape_line = {sid: shapely.linestrings(g[["x", "y"]].to_numpy()) for sid, g in shp.groupby("shape_id")}

    routes = {}
    for rid, g in trips.groupby("route_id"):
        lines = [shape_line[s] for s in g.shape_id.unique() if s in shape_line]
        if not lines:
            continue
        routes[rid] = {
            "line": shapely.union_all(lines),
            "stops": shapely.multipoints([stop_geom[s] for s in st.loc[st.route_id == rid, "stop_id"].unique() if s in stop_geom]),
            "termini": shapely.multipoints([stop_geom[s] for s in ends.loc[ends.route_id == rid, "stop_id"].unique() if s in stop_geom]),
        }
    return routes, stops_g


# ---------------------------------------------------------------- positions
def load_positions(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype={"Lines": str, "Brigade": str, "VehicleNumber": str},
                     usecols=["vehicle_type", "Lines", "VehicleNumber", "Lat", "Lon", "gps_time"],
                     on_bad_lines="skip")  # last line may be half-written
    df["gps_time"] = pd.to_datetime(df.gps_time, errors="coerce")
    df = df.dropna(subset=["gps_time", "Lat", "Lon", "Lines"])
    pts = gpd.points_from_xy(df.Lon, df.Lat, crs=4326).to_crs(CRS)
    df["x"], df["y"] = pts.x, pts.y
    return df


def match_routes(df: pd.DataFrame, routes: dict) -> pd.DataFrame:
    df["d_route"] = np.nan
    df["d_stop"] = np.nan
    df["d_term"] = np.nan
    pts = shapely.points(df.x.to_numpy(), df.y.to_numpy())
    for rid, idx in df.groupby("Lines").indices.items():
        r = routes.get(rid)
        if r is None:
            continue
        p = pts[idx]
        df.iloc[idx, df.columns.get_loc("d_route")] = shapely.distance(p, r["line"])
        df.iloc[idx, df.columns.get_loc("d_stop")]  = shapely.distance(p, r["stops"])
        df.iloc[idx, df.columns.get_loc("d_term")]  = shapely.distance(p, r["termini"])
    return df


def intervals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["vehicle_type", "VehicleNumber", "gps_time"]).reset_index(drop=True)
    same = (df.VehicleNumber.eq(df.VehicleNumber.shift()) & df.vehicle_type.eq(df.vehicle_type.shift())
            & df.Lines.eq(df.Lines.shift()))
    dt = (df.gps_time - df.gps_time.shift()).dt.total_seconds()
    dist = np.hypot(df.x - df.x.shift(), df.y - df.y.shift())
    iv = pd.DataFrame({
        "vehicle_type": df.vehicle_type, "Lines": df.Lines, "VehicleNumber": df.VehicleNumber,
        "t": df.gps_time, "dt": dt, "speed_kmh": dist / dt * 3.6,
        "x": (df.x + df.x.shift()) / 2, "y": (df.y + df.y.shift()) / 2,
        # both ends must satisfy the filters -> use max distance to route, min to stop/terminus
        "d_route": np.maximum(df.d_route, df.d_route.shift()),
        "d_stop": np.minimum(df.d_stop, df.d_stop.shift()),
        "d_term": np.minimum(df.d_term, df.d_term.shift()),
    })[same & dt.between(MIN_DT, MAX_DT)]
    iv = iv[iv.speed_kmh < 120]  # GPS jumps

    iv["slow"] = iv.speed_kmh < STUCK_KMH
    # stand-still episodes: consecutive slow intervals of one vehicle
    new_ep = ~iv.slow | ~iv.slow.shift(fill_value=False) | iv.VehicleNumber.ne(iv.VehicleNumber.shift())
    iv["episode"] = new_ep.cumsum()
    ep_len = iv[iv.slow].groupby("episode").dt.sum()
    iv["ep_min"] = iv.episode.map(ep_len).fillna(0) / 60
    return iv


def classify(iv: pd.DataFrame) -> pd.DataFrame:
    stop_m = iv.vehicle_type.map(STOP_M)
    iv["why"] = "moving"
    s = iv.slow
    iv.loc[s, "why"] = "stuck"
    iv.loc[s & (iv.d_stop < stop_m), "why"] = "at_stop"
    iv.loc[s & (iv.d_term < TERMINUS_M), "why"] = "terminus"
    iv.loc[s & (iv.ep_min > LAYOVER_MIN), "why"] = "layover"
    iv.loc[iv.d_route.isna() | (iv.d_route > OFF_ROUTE_M), "why"] = "off_route"
    return iv


def grid(iv: pd.DataFrame, stops_g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    st = iv[iv.why == "stuck"].copy()
    st["gx"] = (st.x // GRID_M).astype(int)
    st["gy"] = (st.y // GRID_M).astype(int)
    st["hour"] = st.t.dt.hour
    g = st.groupby(["gx", "gy"])
    cells = pd.DataFrame({
        "total_min": g.dt.sum() / 60,
        "tram_min": st[st.vehicle_type == "Tram"].groupby(["gx", "gy"]).dt.sum() / 60,
        "bus_min": st[st.vehicle_type == "Bus"].groupby(["gx", "gy"]).dt.sum() / 60,
        "vehicles": g.VehicleNumber.nunique(),
        "lines_n": g.Lines.nunique(),
        "top_lines": g.apply(lambda d: ",".join(d.groupby("Lines").dt.sum().nlargest(5).index), include_groups=False),
        "peak_hour": g.apply(lambda d: int(d.groupby("hour").dt.sum().idxmax()), include_groups=False),
    }).fillna({"tram_min": 0, "bus_min": 0}).reset_index()
    cells["min_per_veh"] = cells.total_min / cells.vehicles
    geom = [box(x * GRID_M, y * GRID_M, (x + 1) * GRID_M, (y + 1) * GRID_M) for x, y in zip(cells.gx, cells.gy)]
    cells = gpd.GeoDataFrame(cells, geometry=geom, crs=CRS)
    boarding = stops_g[stops_g.location_type.fillna(0).astype(int).eq(0) & ~stops_g.stop_name.str.fullmatch(r"\d+")]
    near = gpd.sjoin_nearest(cells.set_geometry(cells.centroid), boarding[["stop_name", "geometry"]], how="left")
    cells["near_stop"] = near.groupby(level=0).stop_name.first()
    for c in ["total_min", "tram_min", "bus_min", "min_per_veh"]:
        cells[c] = cells[c].round(1)
    return cells.sort_values("total_min", ascending=False)


def plot(cells: gpd.GeoDataFrame, out: Path, title: str):
    top = cells[cells.total_min >= 5]
    fig, ax = plt.subplots(figsize=(14, 14), facecolor="#111")
    ax.set_facecolor("#111")
    xmin, ymin, xmax, ymax = cells.total_bounds
    cx, cy = cells.geometry.centroid.x.median(), cells.geometry.centroid.y.median()
    r = 12_000
    if OSM_ROADS.exists():
        roads = gpd.read_file(OSM_ROADS, bbox=tuple(gpd.GeoSeries([box(cx - r, cy - r, cx + r, cy + r)], crs=CRS).to_crs(4326).total_bounds))
        roads.to_crs(CRS).plot(ax=ax, color="#333", linewidth=0.3)
    pts = top.geometry.centroid
    sc = ax.scatter(pts.x, pts.y, s=np.clip(top.total_min, 0, 600) / 2, c=top.tram_min / top.total_min,
                    cmap="coolwarm", vmin=0, vmax=1, alpha=0.75, linewidths=0)
    for _, c in top.head(15).iterrows():
        p = c.geometry.centroid
        ax.annotate(f"{c.near_stop} ({c.total_min:.0f} min)", (p.x, p.y), color="white", fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r); ax.set_axis_off()
    cb = fig.colorbar(sc, ax=ax, shrink=0.4); cb.set_label("share of trams (blue = buses, red = trams)", color="w")
    cb.ax.tick_params(colors="w")
    ax.set_title(title, color="white", fontsize=14)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYYMMDD (default: latest positions_*.csv)")
    a = ap.parse_args()
    csv = DATA_DIR / f"positions_{a.date}.csv" if a.date else sorted(DATA_DIR.glob("positions_*.csv"))[-1]
    day = csv.stem.split("_")[1]

    routes, stops_g = load_gtfs(latest_gtfs(), day)
    df = load_positions(csv)
    print(f"{len(df):,} fixes, {df.gps_time.min():%H:%M}–{df.gps_time.max():%H:%M}")
    df = match_routes(df, routes)
    iv = classify(intervals(df))

    summ = iv.groupby(["vehicle_type", "why"]).dt.sum().div(3600).round(1).unstack(0)
    print("\nVehicle-hours by category:\n", summ.to_string())

    cells = grid(iv, stops_g)
    gpkg = DATA_DIR / f"stuck_cells_{day}.gpkg"
    cells.to_file(gpkg, driver="GPKG")
    png = DATA_DIR / f"stuck_{day}.png"
    span = f"{df.gps_time.min():%H:%M}–{df.gps_time.max():%H:%M}"
    plot(cells, png, f"Warsaw ZTM — where vehicles stand still (excl. stops, termini) · {day} {span}")
    print(f"\nTop 20 cells:\n{cells.drop(columns='geometry').head(20).to_string(index=False)}")
    print(f"\nSaved {gpkg}\nSaved {png}")


if __name__ == "__main__":
    main()
