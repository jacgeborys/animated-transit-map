"""
Step 7a — the tram network as 10 m samples along every observed shape, each labelled with the street it runs on.

  * nearest named OSM road (hierarchy <= 6) within 35 m of the sample, then a rolling mode over ±120 m so that
    a cross street at a junction does not split a corridor
  * heading of travel, distance from the city centre (PKiN) — used to orient corridors (0 = centre end)

Output: _cache/net_samples_<date>.parquet  (shape_id, m, x, y, heading, street)
Usage:  py -3.11 live/analysis/network.py
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

import ta_config as C
import ta_common as T

STEP_M = 10
NAME_SNAP_M = 35
SMOOTH_N = 25           # samples (±120 m)


def rolling_mode(a, n):
    out = a.copy()
    h = n // 2
    codes, uniq = pd.factorize(pd.Series(a))
    for i in range(len(a)):
        w = codes[max(0, i - h): i + h + 1]
        w = w[w >= 0]
        if len(w):
            out[i] = uniq[np.bincount(w).argmax()]
    return out


def main():
    trips, st, shp = T.load_gtfs()
    geoms = T.shape_lines(shp)
    obs = pd.read_parquet(C.CACHE / f"trips_obs_{C.DATE}.parquet")
    rows = []
    for sid in obs.shape_id.unique():
        L = geoms[sid]
        m = np.arange(0, L.length, STEP_M)
        p = shapely.line_interpolate_point(L, m)
        q = shapely.line_interpolate_point(L, np.minimum(m + 5, L.length))
        r = shapely.line_interpolate_point(L, np.maximum(m - 5, 0))
        dx, dy = shapely.get_x(q) - shapely.get_x(r), shapely.get_y(q) - shapely.get_y(r)
        rows.append(pd.DataFrame({"shape_id": sid, "m": m, "x": shapely.get_x(p), "y": shapely.get_y(p),
                                  "heading": np.degrees(np.arctan2(dx, dy)) % 360}))
    ns = pd.concat(rows, ignore_index=True)

    pts = shapely.points(ns.x, ns.y)
    bb = tuple(gpd.GeoSeries([shapely.box(ns.x.min(), ns.y.min(), ns.x.max(), ns.y.max()).buffer(200)], crs=C.CRS)
               .to_crs(4326).total_bounds)
    roads = gpd.read_file(C.OSM_ROADS, bbox=bb).to_crs(C.CRS)
    roads = roads[roads.name.notna() & roads.hierarchy.astype(float).le(6)].reset_index(drop=True)
    tree = shapely.STRtree(roads.geometry.values)
    i_pt, i_rd = tree.query_nearest(pts, max_distance=NAME_SNAP_M, all_matches=False)
    name = np.full(len(ns), None, dtype=object)
    name[i_pt] = roads.name.to_numpy()[i_rd]
    ns["street_raw"] = name
    ns["street"] = None
    for sid, g in ns.groupby("shape_id"):
        ns.loc[g.index, "street"] = rolling_mode(g.street_raw.to_numpy(dtype=object), SMOOTH_N)
    ns = ns.drop(columns="street_raw")
    ns.to_parquet(C.CACHE / f"net_samples_{C.DATE}.parquet")

    # tram-km per street (trips x length) to see which corridors matter
    nt = obs.groupby("shape_id").size()
    ns["runs"] = ns.shape_id.map(nt)
    s = (ns.groupby("street").runs.sum() * STEP_M / 1000).sort_values(ascending=False)
    print("tram-km/day by street:\n", s.head(45).round(0).to_string())


if __name__ == "__main__":
    main()
