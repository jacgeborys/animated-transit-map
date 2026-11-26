"""
Step 3 — signalised crossings: how often does a tram have to stop at each one, per line and direction?

  * OSM signal nodes (fetch_signals.py) within SIG_SNAP_M of a trip's shape become signal points along it;
    nodes closer than SIG_MERGE_M along the route are one junction.
  * For each observed run past a signal point: did it stand still (outside platform zones) in the approach
    window [first node - APPROACH_M, last node + 5 m]? How long?
  * A platform inside the window hides the red phase in the dwell, so those points are flagged
    `platform_at_signal` and their stop rate is a lower bound.

Reading the stop rate: a tram meets a fixed-time signal at random -> it stops roughly as often as the
red share for its movement (typically 40-70 %). Working active priority -> ~5-20 %.

Output (_output/):  signals_<date>.csv  (one row per line / direction / signal point)
                    signals_<date>.gpkg (same, points)
Usage: python live/analysis/signals.py
"""

import json

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

import ta_config as C
import ta_common as T
from places import road_names

SIG_SNAP_M  = 18
SIG_MERGE_M = 60
APPROACH_M  = 90


def load_signals():
    d = json.loads((C.CACHE / "signals.json").read_text(encoding="utf-8"))
    el = [e for e in d["elements"] if e["type"] == "node"]
    g = gpd.GeoDataFrame({"osm_id": [e["id"] for e in el]},
                         geometry=gpd.points_from_xy([e["lon"] for e in el], [e["lat"] for e in el], crs=4326)).to_crs(C.CRS)
    return shapely.points(g.geometry.x.to_numpy(), g.geometry.y.to_numpy())


def signal_points(L, sig, tree, stops_m):
    near = tree.query(L, predicate="dwithin", distance=SIG_SNAP_M)
    if not len(near):
        return []
    m = np.sort(shapely.line_locate_point(L, sig[near]))
    groups = np.split(m, np.flatnonzero(np.diff(m) > SIG_MERGE_M) + 1)
    out = []
    for g in groups:
        lo, hi = g.min(), g.max()
        plat = np.any((stops_m >= lo - APPROACH_M - C.STOP_AFTER_M) & (stops_m <= hi + C.STOP_BEFORE_M))
        out.append((lo, hi, plat))
    return out


def main():
    trips, st, shp = T.load_gtfs()
    geoms = T.shape_lines(shp)
    sig = load_signals()
    tree = shapely.STRtree(sig)
    obs = pd.read_parquet(C.CACHE / f"trips_obs_{C.DATE}.parquet")
    iv = pd.read_parquet(C.CACHE / f"intervals_{C.DATE}.parquet")
    iv = iv[iv.stand & ~iv.anomaly & (iv.zone != "terminus")]
    iv_by_trip = {k: g for k, g in iv.groupby("trip_id")}
    st_by_trip = {k: g for k, g in st.groupby("trip_id")}
    # stop m along the shape is in stop_times (scaled in match_trips; here unscaled — <0.1 % difference)

    rows = []
    for shape_id, ob in obs.groupby("shape_id"):
        L = geoms[shape_id]
        sm = st_by_trip[ob.trip_id.iloc[0]].m.to_numpy()
        pts = signal_points(L, sig, tree, sm)
        for k, (lo, hi, plat) in enumerate(pts):
            if lo - APPROACH_M < sm[0] + C.TERMINUS_M or hi > sm[-1] - C.TERMINUS_M:
                continue
            for r in ob.itertuples():
                if not (r.m_first < lo - APPROACH_M and r.m_last > hi):
                    continue
                s = iv_by_trip.get(r.trip_id)
                lost_line = lost_plat = 0.0
                if s is not None:
                    w = s[(s.m >= lo - APPROACH_M) & (s.m <= hi + 5)]
                    lost_line = w.loc[w.zone == "line", "dt"].sum()
                    lost_plat = w.loc[w.zone == "platform", "dt"].sum()
                rows.append((r.route_id, r.direction_id, r.trip_headsign, shape_id, k, lo, hi, plat, r.trip_id,
                             r.t_first, r.places, lost_line, lost_plat))
    d = pd.DataFrame(rows, columns=["line", "dir", "headsign", "shape_id", "k", "m_lo", "m_hi", "platform_at_signal",
                                    "trip_id", "t", "places", "lost_line_s", "lost_platform_s"])
    d["stopped"] = d.lost_line_s > 0
    d["peak"] = T.peak(d.t)
    p = shapely.line_interpolate_point(np.array([geoms[s] for s in d.shape_id]), d.m_lo.to_numpy())
    p2 = shapely.line_interpolate_point(np.array([geoms[s] for s in d.shape_id]), d.m_lo.to_numpy() + 10)
    d["x"], d["y"] = shapely.get_x(p).round(), shapely.get_y(p).round()
    hd = np.degrees(np.arctan2(shapely.get_y(p2) - d.y, shapely.get_x(p2) - d.x)) % 360
    d["hs"] = np.floor(((hd + 22.5) % 360) / 45).astype(int)          # 8 heading sectors, 0 = east
    # one physical approach can sit at slightly different m on variant shapes -> group by location + heading
    d["loc"] = (d.x // 50).astype(int).astype(str) + "_" + (d.y // 50).astype(int).astype(str) + "_h" + d.hs.astype(str)
    g = d.groupby(["line", "dir", "loc"])
    res = pd.DataFrame({
        "headsign": g.headsign.agg(lambda v: v.value_counts().index[0]),
        "m": g.m_lo.median().round(), "x": g.x.median(), "y": g.y.median(),
        "platform_at_signal": g.platform_at_signal.max(),
        "passes": g.size(), "stop_prob": g.stopped.mean().round(3),
        "stop_prob_peak": d[d.peak].groupby(["line", "dir", "loc"]).stopped.mean().round(3),
        "lost_per_pass_s": g.lost_line_s.mean().round(1),
        "lost_per_stop_s": (g.lost_line_s.sum() / g.stopped.sum().replace(0, np.nan)).round(1),
        "lost_h_day": (g.lost_line_s.sum() / 3600).round(2),
        "place_h_day": (d.lost_line_s * d.places.fillna(200)).groupby([d.line, d.dir, d["loc"]]).sum().div(3600).round(1),
    }).reset_index()
    res = res[res.passes >= 10]
    gdf = gpd.GeoDataFrame(res, geometry=gpd.points_from_xy(res.x, res.y), crs=C.CRS)
    gdf["name"] = road_names(gdf.geometry)
    C.OUT.mkdir(exist_ok=True)
    gdf.drop(columns="geometry").to_csv(C.OUT / f"signals_{C.DATE}.csv", index=False)
    gdf.to_file(C.OUT / f"signals_{C.DATE}.gpkg", driver="GPKG")
    d.drop(columns=["shape_id"]).to_parquet(C.CACHE / f"signal_passes_{C.DATE}.parquet")

    clean = gdf[~gdf.platform_at_signal]
    w = clean.groupby(["line", "dir"]).apply(lambda q: pd.Series({
        "headsign": q.headsign.iloc[0], "signals": len(q),
        "stop_rate": np.average(q.stop_prob, weights=q.passes),
        "stops_per_run": q.stop_prob.sum(), "lost_min_per_run": q.lost_per_pass_s.sum() / 60}), include_groups=False)
    print(f"{len(gdf)} line/direction signal approaches ({(~gdf.platform_at_signal).sum()} without a platform in the approach)")
    print("\nPer line & direction (signals without platform):\n" + w.sort_values("stop_rate").round(2).to_string())
    print(f"\nSaved {C.OUT / f'signals_{C.DATE}.csv'}")


if __name__ == "__main__":
    main()
