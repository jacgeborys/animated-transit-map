"""
Step 2 — where do trams stand still away from platforms, per direction of travel?

Stand-stills in the "line" zone (not at a platform, not at a terminus, not an > 8 min incident) are pooled
across lines on a 40 m grid, separately for 8 travel-heading sectors, then neighbouring cells of one
heading are merged into PLACES (junction approaches). For each place:

  passes      trams that drove through (any line)          — from the matched traces
  stop_prob   share of passing trams that had to stop      — ~0.05-0.15 = priority works, 0.4+ = none / bad
  lost_s      seconds standing, total / per pass / per stopping tram
  place_h     standing time x nominal places of the vehicle (capacity-hours, measured fleet)

Output (_output/):  places_<date>.gpkg  (layer places + layer cells),  places_<date>.csv
Usage: python live/analysis/places.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from sklearn.cluster import DBSCAN

import ta_config as C
import ta_common as T

COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]   # heading 0° = east (math angle)


def key(x, y, h):
    return (np.floor(x / C.CELL_M).astype(np.int64), np.floor(y / C.CELL_M).astype(np.int64),
            (np.floor(((h + 22.5) % 360) / (360 / C.HEADINGS))).astype(np.int64))


def pass_counts(obs, geoms):
    """Number of distinct trips through each (cell, heading) key."""
    gx, gy, hs, tid = [], [], [], []
    for r in obs.itertuples():
        L = geoms[r.shape_id]
        m = np.arange(r.m_first, r.m_last, C.SAMPLE_M)
        if len(m) < 2:
            continue
        a = shapely.line_interpolate_point(L, m)
        b = shapely.line_interpolate_point(L, m + 8)
        x, y = shapely.get_x(a), shapely.get_y(a)
        h = np.degrees(np.arctan2(shapely.get_y(b) - y, shapely.get_x(b) - x)) % 360
        k = key(x, y, h)
        gx.append(k[0]); gy.append(k[1]); hs.append(k[2]); tid.append(np.full(len(m), r.Index))
    p = pd.DataFrame({"gx": np.concatenate(gx), "gy": np.concatenate(gy), "hs": np.concatenate(hs),
                      "trip": np.concatenate(tid)}).drop_duplicates()
    return p


def road_names(pts):
    if not C.OSM_ROADS.exists():
        return [""] * len(pts)
    bb = tuple(gpd.GeoSeries([shapely.box(*pts.total_bounds).buffer(200)], crs=C.CRS).to_crs(4326).total_bounds)
    roads = gpd.read_file(C.OSM_ROADS, bbox=bb).to_crs(C.CRS)
    roads = roads[roads.name.notna() & roads.hierarchy.astype(float).le(6)][["name", "geometry"]]
    sidx = roads.sindex
    out = []
    for p in pts:
        cand = roads.iloc[sidx.query(p.buffer(45))]
        cand = cand.assign(d=cand.distance(p)).sort_values("d").drop_duplicates("name")
        out.append(" × ".join(cand.name.head(2)))
    return out


def main():
    trips, st, shp = T.load_gtfs()
    geoms = T.shape_lines(shp)
    obs = pd.read_parquet(C.CACHE / f"trips_obs_{C.DATE}.parquet").set_index("trip_id")
    iv = pd.read_parquet(C.CACHE / f"intervals_{C.DATE}.parquet")
    iv = iv.join(obs[["route_id", "trip_headsign", "places", "model"]], on="trip_id")

    s = iv[iv.stand & (iv.zone == "line") & ~iv.anomaly].copy()
    s["gx"], s["gy"], s["hs"] = key(s.x.to_numpy(), s.y.to_numpy(), s.heading.to_numpy())
    s["ep"] = s.trip_id + "#" + s.episode.astype(str)
    s["peak"] = T.peak(s.t0)
    s["place_s"] = s.dt * s.places.fillna(200)
    lf = np.where(s.peak, C.LOAD_FACTOR["peak"], C.LOAD_FACTOR["offpeak"])
    s["pax_s"] = s.place_s * lf                                   # ASSUMPTION: load factor

    print("counting passes ...")
    p = pass_counts(obs, geoms)
    passes = p.groupby(["gx", "gy", "hs"]).size().rename("passes")

    cells = s.groupby(["gx", "gy", "hs"]).agg(lost_s=("dt", "sum"), trips_stopping=("trip_id", "nunique")).join(passes).reset_index()
    cells["x"] = (cells.gx + 0.5) * C.CELL_M; cells["y"] = (cells.gy + 0.5) * C.CELL_M

    # merge neighbouring cells of one heading into places
    cells["place"] = -1
    nxt = 0
    for h, g in cells.groupby("hs"):
        lab = DBSCAN(eps=C.CELL_M * 1.5, min_samples=1).fit_predict(g[["x", "y"]].to_numpy())
        cells.loc[g.index, "place"] = lab + nxt
        nxt += lab.max() + 1
    s = s.merge(cells[["gx", "gy", "hs", "place"]], on=["gx", "gy", "hs"])

    # passes of a place = distinct trips through any of its cells
    pp = p.merge(cells[["gx", "gy", "hs", "place"]], on=["gx", "gy", "hs"]).groupby("place").trip.nunique()
    g = s.groupby("place")
    pl = pd.DataFrame({
        "lost_s": g.dt.sum(), "stops": g.trip_id.nunique(), "stop_events": g.ep.nunique(), "passes": pp,
        "n_cells": g.apply(lambda d: d[["gx", "gy"]].drop_duplicates().shape[0], include_groups=False),
        "lost_peak_s": s[s.peak].groupby("place").dt.sum(),
        "place_h": g.place_s.sum() / 3600, "pax_h_est": g.pax_s.sum() / 3600,
        "lines": g.apply(lambda d: ",".join(d.groupby("route_id").dt.sum().sort_values(ascending=False).index), include_groups=False),
        "towards": g.trip_headsign.agg(lambda v: v.value_counts().index[0]),
        "x": g.apply(lambda d: np.average(d.x, weights=d.dt), include_groups=False),
        "y": g.apply(lambda d: np.average(d.y, weights=d.dt), include_groups=False),
        "hs": g.hs.first(),
    }).fillna({"lost_peak_s": 0})
    pl = pl[pl.passes >= C.MIN_PASSES]
    pl["stop_prob"] = (pl.stops / pl.passes).clip(upper=1)
    pl["lost_per_pass_s"] = pl.lost_s / pl.passes
    pl["lost_per_stop_s"] = pl.lost_s / pl.stops
    pl["heading"] = pl.hs.map(dict(enumerate(COMPASS)))
    # saving if priority removed PRIORITY_EFFECT of the standing + the stop-and-go penalty   (ASSUMPTION)
    pl["save_h_day"] = C.PRIORITY_EFFECT * (pl.lost_s + pl.stop_events * C.ACCEL_PENALTY_S) / 3600

    gdf = gpd.GeoDataFrame(pl, geometry=gpd.points_from_xy(pl.x, pl.y), crs=C.CRS)
    st_g = st.drop_duplicates("stop_id")
    stops = pd.read_csv(T.gtfs_dir() / "stops.txt", dtype={"stop_id": str}).set_index("stop_id")
    sg = gpd.GeoDataFrame(st_g[["stop_name"]].reset_index(drop=True),
                          geometry=gpd.points_from_xy(stops.loc[st_g.stop_id, "stop_lon"], stops.loc[st_g.stop_id, "stop_lat"], crs=4326)).to_crs(C.CRS)
    near = gpd.sjoin_nearest(gdf, sg, how="left", distance_col="stop_dist")
    gdf["near_stop"] = near.groupby(level=0).stop_name.first()
    gdf["streets"] = road_names(gdf.geometry)
    gdf["name"] = np.where(gdf.streets != "", gdf.streets, gdf.near_stop)
    gdf = gdf.sort_values("lost_s", ascending=False).reset_index()
    for c in ["lost_s", "lost_peak_s", "x", "y"]:
        gdf[c] = gdf[c].round(0)
    for c in ["place_h", "pax_h_est", "stop_prob", "lost_per_pass_s", "lost_per_stop_s", "save_h_day"]:
        gdf[c] = gdf[c].round(3 if c == "stop_prob" else 1)

    C.OUT.mkdir(exist_ok=True)
    gpkg = C.OUT / f"places_{C.DATE}.gpkg"
    gdf.to_file(gpkg, layer="places", driver="GPKG")
    cg = cells[cells.passes.fillna(0) >= C.MIN_PASSES / 2].copy()
    cg["stop_prob"] = (cg.trips_stopping / cg.passes).clip(upper=1).round(3)
    gpd.GeoDataFrame(cg, geometry=[shapely.box(a * C.CELL_M, b * C.CELL_M, (a + 1) * C.CELL_M, (b + 1) * C.CELL_M)
                                   for a, b in zip(cg.gx, cg.gy)], crs=C.CRS).to_file(gpkg, layer="cells", driver="GPKG")
    gdf.drop(columns="geometry").to_csv(C.OUT / f"places_{C.DATE}.csv", index=False)

    cols = ["name", "near_stop", "heading", "towards", "lines", "n_cells", "passes", "stop_prob", "lost_per_pass_s", "lost_s", "save_h_day"]
    print(f"\n{len(gdf)} places, {gdf.lost_s.sum()/3600:.0f} tram-hours standing away from platforms")
    print("\nTop 25 by total time lost:\n" + gdf[cols].head(25).to_string(index=False))
    busy = gdf[gdf.passes >= 150].sort_values("stop_prob", ascending=False)
    print("\nWorst stop probability (>= 150 trams/day):\n" + busy[cols].head(20).to_string(index=False))
    print(f"\nSaved {gpkg}")


if __name__ == "__main__":
    main()
