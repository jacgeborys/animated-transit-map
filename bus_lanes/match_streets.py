"""
Step C — Match GTFS shapes to OSM streets (per direction) and classify bus lane need.

Each shape is sampled every SAMPLE_STEP_M metres. Every sample is matched to the nearest
OSM way within MATCH_MAX_DIST_M whose bearing is parallel (±MATCH_MAX_ANGLE_DEG) and whose
oneway rule allows that travel direction — this separates dual carriageways correctly.
Counts are then summed per (OSM way, direction), so a street with a bus lane only
in one direction is evaluated per direction.

Outputs:
    bus_lanes/_output/bus_lane_need.gpkg
        layer 'bus_dir'   one row per (way, direction); geometry drawn in travel direction
        layer 'tram_dir'  same for tram tracks
        layer 'boundary'  Warsaw boundary
    bus_lanes/_output/bus_lane_gaps_ranking.csv   streets without bus lanes, ranked

Bus lanes = OSM tags OR city layer OR manual list (bus_lanes/_data/extra_lanes.gpkg, built by
official_lanes.py). Columns bl_osm / bl_city / bl_manual record which source says so.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from bl_config import (DATA_DIR, MATCH_MAX_ANGLE_DEG, MATCH_MAX_DIST_M, NEED_HIGH, NEED_MED,
                       OUTPUT_DIR, POLAND_CRS, RAW_DIR, SAMPLE_STEP_M)

# small distance penalty (m) so a parallel parking aisle doesn't steal a bus from the main road
HIGHWAY_PENALTY = {"service": 6, "living_street": 6, "residential": 2, "unclassified": 1}


def load_shapes(feed: Path, shape_ids: set) -> gpd.GeoDataFrame:
    sh = pd.read_csv(feed / "shapes.txt", dtype={"shape_id": str})
    sh = sh[sh.shape_id.isin(shape_ids)].sort_values(["shape_id", "shape_pt_sequence"])
    geoms, ids, dmax = [], [], []
    for sid, g in sh.groupby("shape_id"):
        if len(g) < 2:
            continue
        geoms.append(LineString(zip(g.shape_pt_lon, g.shape_pt_lat)))
        ids.append(sid)
        dmax.append(g.shape_dist_traveled.max())
    return gpd.GeoDataFrame({"shape_id": ids, "dist_max": dmax}, geometry=geoms,
                            crs="EPSG:4326").to_crs(POLAND_CRS)


def sample_shapes(shapes: gpd.GeoDataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """Midpoints every SAMPLE_STEP_M along each shape, with bearing and passing counts."""
    rows = []
    prof_by_shape = {k: g for k, g in prof.groupby("shape_id")}
    for sid, geom, dmax in tqdm(zip(shapes.shape_id, shapes.geometry, shapes.dist_max),
                                total=len(shapes), desc="Sampling shapes"):
        p = prof_by_shape.get(sid)
        if p is None:
            continue
        L = geom.length
        n = max(int(L // SAMPLE_STEP_M), 1)
        d = (np.arange(n) + 0.5) * (L / n)
        a = shapely.get_coordinates(geom.interpolate(np.clip(d - 3, 0, L)))
        b = shapely.get_coordinates(geom.interpolate(np.clip(d + 3, 0, L)))
        mid = shapely.get_coordinates(geom.interpolate(d))
        bearing = np.degrees(np.arctan2(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])) % 360
        # counts valid from the last stop at or before this point
        dist_km = d / L * dmax
        idx = np.clip(np.searchsorted(p.dist_km.to_numpy(), dist_km, side="right") - 1, 0, len(p) - 1)
        rows.append(pd.DataFrame({
            "shape_id": sid, "mode": p["mode"].iat[0], "x": mid[:, 0], "y": mid[:, 1],
            "bearing": bearing,
            "am_n": p.am_n.to_numpy()[idx], "pm_n": p.pm_n.to_numpy()[idx],
            "day_n": p.day_n.to_numpy()[idx],
        }))
    return pd.concat(rows, ignore_index=True)


def way_bearing_at(ways_geom: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pos = shapely.line_locate_point(ways_geom, pts)
    L = shapely.length(ways_geom)
    a = shapely.get_coordinates(shapely.line_interpolate_point(ways_geom, np.clip(pos - 3, 0, L)))
    b = shapely.get_coordinates(shapely.line_interpolate_point(ways_geom, np.clip(pos + 3, 0, L)))
    return np.degrees(np.arctan2(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])) % 360


def match(samples: pd.DataFrame, ways: gpd.GeoDataFrame, chunk=300_000) -> pd.DataFrame:
    """Returns samples with way index (row in `ways`) and dir (+1 forward / -1 backward)."""
    tree = shapely.STRtree(ways.geometry.values)
    wgeom = ways.geometry.values
    oneway = ways["oneway"].to_numpy() if "oneway" in ways else np.zeros(len(ways), int)
    penalty = ways["highway"].map(HIGHWAY_PENALTY).fillna(0).to_numpy() if "highway" in ways \
        else np.zeros(len(ways))
    out_way = np.full(len(samples), -1)
    out_dir = np.zeros(len(samples), int)
    for s in tqdm(range(0, len(samples), chunk), desc="Matching"):
        sub = samples.iloc[s:s + chunk]
        pts = shapely.points(sub.x.to_numpy(), sub.y.to_numpy())
        pi, wi = tree.query(pts, predicate="dwithin", distance=MATCH_MAX_DIST_M)
        if len(pi) == 0:
            continue
        dist = shapely.distance(pts[pi], wgeom[wi])
        wb = way_bearing_at(wgeom[wi], pts[pi])
        diff = (sub.bearing.to_numpy()[pi] - wb) % 360
        fwd = (diff < MATCH_MAX_ANGLE_DEG) | (diff > 360 - MATCH_MAX_ANGLE_DEG)
        bwd = np.abs(diff - 180) < MATCH_MAX_ANGLE_DEG
        ow = oneway[wi]
        ok = (fwd & (ow != -1)) | (bwd & (ow != 1))
        cand = pd.DataFrame({"p": pi[ok], "w": wi[ok], "dir": np.where(fwd[ok], 1, -1),
                             "score": dist[ok] + penalty[wi[ok]]})
        best = cand.sort_values("score").drop_duplicates("p")
        out_way[s + best.p.to_numpy()] = best.w.to_numpy()
        out_dir[s + best.p.to_numpy()] = best.dir.to_numpy()
    samples = samples.assign(way=out_way, dir=out_dir)
    print(f"  matched {100 * (out_way >= 0).mean():.1f}% of samples")
    return samples[samples.way >= 0]


def aggregate(samples: pd.DataFrame, ways: gpd.GeoDataFrame) -> pd.DataFrame:
    """Sum trips per (way, dir). Each shape counts once per way-direction it really runs along."""
    g = (samples.groupby(["way", "dir", "shape_id"])
                .agg(n=("x", "size"), am_n=("am_n", "median"), pm_n=("pm_n", "median"),
                     day_n=("day_n", "median"))
                .reset_index())
    wlen = ways.geometry.length.to_numpy()[g.way]
    covered = g.n * SAMPLE_STEP_M
    g = g[covered >= np.minimum(0.4 * wlen, 40)]  # drop grazing matches at intersections
    agg = (g.groupby(["way", "dir"])
             .agg(am_n=("am_n", "sum"), pm_n=("pm_n", "sum"), day_n=("day_n", "sum"),
                  n_shapes=("shape_id", "nunique"))
             .reset_index())
    agg["per_h"] = np.maximum(agg.am_n, agg.pm_n) / 2.0
    return agg


def directional_geoms(agg: pd.DataFrame, ways: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    w = ways.iloc[agg.way.to_numpy()].reset_index(drop=True)
    geom = w.geometry.values
    back = agg.dir.to_numpy() == -1
    geom[back] = shapely.reverse(geom[back])
    out = pd.concat([agg.reset_index(drop=True), w.drop(columns="geometry")], axis=1)
    return gpd.GeoDataFrame(out, geometry=geom, crs=POLAND_CRS)


def extra_lane_cover(gdf: gpd.GeoDataFrame, lanes: gpd.GeoDataFrame, step=10, dist=25,
                     max_angle=35, min_share=0.5) -> np.ndarray:
    """Per directional way: True if ≥ min_share of it runs within `dist` m of a lane line
    pointing the same way (lane lines are directional: vertex order = travel direction)."""
    if lanes.empty:
        return np.zeros(len(gdf), bool)
    lg = lanes.geometry.values
    tree = shapely.STRtree(lg)
    hit = np.zeros(len(gdf), bool)
    for i, geom in enumerate(gdf.geometry.values):
        L = geom.length
        n = max(int(L // step), 1)
        d = (np.arange(n) + 0.5) * (L / n)
        pts = shapely.line_interpolate_point(geom, d)
        a = shapely.get_coordinates(shapely.line_interpolate_point(geom, np.clip(d - 3, 0, L)))
        b = shapely.get_coordinates(shapely.line_interpolate_point(geom, np.clip(d + 3, 0, L)))
        hb = np.degrees(np.arctan2(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])) % 360
        pi, li = tree.query(pts, predicate="dwithin", distance=dist)
        if len(pi) == 0:
            continue
        lb = way_bearing_at(lg[li], pts[pi])
        ok = np.abs((hb[pi] - lb + 180) % 360 - 180) < max_angle
        hit[i] = len(np.unique(pi[ok])) >= min_share * n
    return hit


def compass(bearing: float) -> str:
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    pl = {"N": "północ", "NE": "płn.-wsch.", "E": "wschód", "SE": "płd.-wsch.",
          "S": "południe", "SW": "płd.-zach.", "W": "zachód", "NW": "płn.-zach."}
    return pl[names[int(((bearing + 22.5) % 360) // 45)]]


def ranking(bus: gpd.GeoDataFrame) -> pd.DataFrame:
    gaps = bus[bus.status.isin(["high", "medium"]) & bus.in_city & bus.name.notna()].copy()
    c = shapely.get_coordinates(shapely.get_point(gaps.geometry.values, 0))
    e = shapely.get_coordinates(shapely.get_point(gaps.geometry.values, -1))
    gaps["heading"] = np.degrees(np.arctan2(e[:, 0] - c[:, 0], e[:, 1] - c[:, 1])) % 360
    gaps["len_m"] = gaps.geometry.length
    # One main axis per street (length-weighted axial mean), then each piece is labelled by
    # which way along that axis it runs → exactly two directions per street name.
    ang2 = np.radians(2 * gaps.heading)
    axis = (gaps.assign(sx=np.sin(ang2) * gaps.len_m, cx=np.cos(ang2) * gaps.len_m)
                .groupby("name")[["sx", "cx"]].sum())
    axis = (np.degrees(np.arctan2(axis.sx, axis.cx)) / 2) % 180
    ax = gaps.name.map(axis)
    along = np.cos(np.radians(gaps.heading - ax)) >= 0
    gaps["dir_label"] = [compass(a if f else a + 180) for a, f in zip(ax, along)]
    gaps["w"] = gaps.per_h * gaps.len_m
    r = (gaps.groupby(["name", "dir_label"])
             .agg(len_m=("len_m", "sum"), w=("w", "sum"), max_per_h=("per_h", "max"),
                  day_max=("day_n", "max"))
             .reset_index())
    r = r[r.len_m >= 100]
    r["avg_per_h"] = (r.w / r.len_m).round(1)
    r["len_km"] = (r.len_m / 1000).round(2)
    r["score"] = (r.avg_per_h * r.len_km).round(1)  # bus-km per hour stuck in general traffic
    return (r.drop(columns=["w", "len_m"])
             .sort_values("score", ascending=False)
             .rename(columns={"name": "ulica", "dir_label": "kierunek"})
             [["ulica", "kierunek", "len_km", "avg_per_h", "max_per_h", "day_max", "score"]])


def extra_lane_flags(gdf: gpd.GeoDataFrame, lanes: gpd.GeoDataFrame, step=10, dist=25,
                     max_angle=35, min_cover=0.5) -> dict:
    """For each directional way geometry: does a same-direction lane line from `lanes` run
    along at least `min_cover` of it? Returns {source: bool array} per lane source."""
    geoms = gdf.geometry.values
    L = shapely.length(geoms)
    n = np.maximum((L // step).astype(int), 1)
    owner = np.repeat(np.arange(len(geoms)), n)
    frac = np.concatenate([(np.arange(k) + 0.5) / k for k in n])
    d = frac * L[owner]
    pts = shapely.line_interpolate_point(geoms[owner], d)
    a = shapely.line_interpolate_point(geoms[owner], np.clip(d - 3, 0, L[owner]))
    b = shapely.line_interpolate_point(geoms[owner], np.clip(d + 3, 0, L[owner]))
    ca, cb = shapely.get_coordinates(a), shapely.get_coordinates(b)
    brg = np.degrees(np.arctan2(cb[:, 0] - ca[:, 0], cb[:, 1] - ca[:, 1])) % 360
    out = {}
    for src, lg in lanes.groupby("zrodlo"):
        lgeom = lg.geometry.values
        pi, li = shapely.STRtree(lgeom).query(pts, predicate="dwithin", distance=dist)
        lb = way_bearing_at(lgeom[li], pts[pi])
        ok = np.abs((brg[pi] - lb + 180) % 360 - 180) < max_angle
        hit = np.zeros(len(pts), bool)
        hit[pi[ok]] = True
        cover = np.bincount(owner, weights=hit, minlength=len(geoms)) / n
        out[src] = cover >= min_cover
    return out


def main():
    date, feed_name = (DATA_DIR / "analysis_date.txt").read_text(encoding="utf-8").split()
    feed = RAW_DIR / feed_name
    prof = pd.read_pickle(DATA_DIR / "shape_profiles.pkl")
    osm = DATA_DIR / "osm_streets.gpkg"
    streets = gpd.read_file(osm, layer="streets")
    trams = gpd.read_file(osm, layer="trams")
    boundary = gpd.read_file(osm, layer="boundary")

    shapes = load_shapes(feed, set(prof.shape_id))
    print(f"Shapes: {len(shapes):,}")
    samples = sample_shapes(shapes, prof)
    print(f"Samples: {len(samples):,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "bus_lane_need.gpkg"
    city = boundary.geometry.iloc[0]

    for mode, ways in (("bus", streets), ("tram", trams)):
        print(f"\n[{mode}]")
        s = match(samples[samples["mode"] == mode].reset_index(drop=True), ways)
        agg = aggregate(s, ways)
        gdf = directional_geoms(agg, ways)
        gdf["in_city"] = gdf.geometry.intersects(city)
        if mode == "bus":
            gdf["bl_osm"] = (np.where(gdf.dir == 1, gdf.bl_fwd, gdf.bl_bwd) == 1).astype(int)
            has_lane = gdf.bl_osm.to_numpy() == 1
            extra = DATA_DIR / "extra_lanes.gpkg"   # city layer + manual list (official_lanes.py)
            if extra.exists():
                flags = extra_lane_flags(gdf, gpd.read_file(extra, layer="lanes"))
                for src, col in (("miasto", "bl_city"), ("reczne", "bl_manual")):
                    gdf[col] = flags.get(src, np.zeros(len(gdf), bool)).astype(int)
                    has_lane = has_lane | (gdf[col].to_numpy() == 1)
                    km = gdf.geometry.length[gdf[col] == 1].sum() / 1000
                    new = gdf.geometry.length[(gdf[col] == 1) & (gdf.bl_osm == 0)].sum() / 1000
                    print(f"  {col}: {km:.1f} km of bus street-direction, {new:.1f} km not in OSM")
            gdf["bus_lane"] = has_lane.astype(int)
            gdf["status"] = np.select(
                [has_lane, gdf.per_h >= NEED_HIGH, gdf.per_h >= NEED_MED],
                ["lane", "high", "medium"], default="low")
        gdf["date"] = date
        gdf.to_file(out, layer=f"{mode}_dir", driver="GPKG")
        print(f"  {len(gdf):,} way-directions, max {gdf.per_h.max():.0f}/h")

        if mode == "bus":
            c = gdf[gdf.in_city]
            c = c[~((c.bus_only == 1) & (c.highway == "service"))]  # skip loops/depots
            km = c.geometry.length / 1000
            print("  Warsaw km by status:", {k: round(v, 1) for k, v in km.groupby(c.status).sum().items()})
            hi = c.per_h >= NEED_HIGH
            print(f"  ≥{NEED_HIGH}/h: {km[hi].sum():.1f} km, of which with bus lane "
                  f"{km[hi & (c.bus_lane == 1)].sum():.1f} km")
            for col in ("bl_osm", "bl_city", "bl_manual"):
                print(f"    lane km per source ({col}): all {km[c[col] == 1].sum():.1f}, "
                      f"on ≥{NEED_HIGH}/h {km[hi & (c[col] == 1)].sum():.1f}")
            rk = ranking(gdf)
            rk.to_csv(OUTPUT_DIR / "bus_lane_gaps_ranking.csv", index=False, encoding="utf-8-sig")
            print(rk.head(25).to_string(index=False))

    boundary.to_file(out, layer="boundary", driver="GPKG")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
