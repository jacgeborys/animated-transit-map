"""
Speed along the route, per line and direction, from all live runs of the day.

City-wide version of the single-ride speed strip in D:/QGIS/gpx_tracker (tram audit, 23 Nov 2026):
  top    — average speed in 100 m bins along the main GTFS shape (total distance / total time of all runs)
  bottom — seconds per run spent at platforms (green) and stuck away from them (orange) in each bin
Same interval classification as stuck_points.py (at_stop / stuck / moving; termini, layovers, off-route excluded).

Direction: fixes are projected onto both directions' main shapes; the direction is the one along which the
vehicle progresses over ~±5 intervals (both directions share the street, so distance alone can't decide).

Output: live/data/route_profiles_<date>/<mode>_<line>_dir<0|1>.png  +  route_profiles_<mode>_<date>.csv (all bins)

Usage:
  python live/route_profiles.py                    # all tram lines
  python live/route_profiles.py --mode Bus         # all bus lines
  python live/route_profiles.py --lines 1 14 31    # selected lines
  python live/route_profiles.py --hours 7 10       # only runs between 07:00 and 10:00
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

import stuck_points as sp

BIN_M       = 100
MATCH_M     = 30      # interval endpoints must be this close to the direction's main shape
DIR_WINDOW  = 11      # intervals (centered) used to decide direction of travel
MIN_RUNS    = 5       # bins with fewer runs are not drawn
REF_KMH     = 40

# colours from the gpx_tracker tram report (light theme)
SURFACE, INK, INK2, MUTED, GRID = "#ffffff", "#121820", "#4b5562", "#7b8592", "#eef0f3"
RUN, WASH, STUCK, DWELL = "#2a78d6", (42/255, 120/255, 214/255, 0.10), "#eb6834", "#1baf7a"
SERVICE = ["moving", "at_stop", "stuck"]


def main_shapes(gtfs, day, lines):
    """(route, direction) -> shape geometry, headsign, stops [(name, m)] for the most used shape of the day."""
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    trips = pd.read_csv(gtfs / "trips.txt", dtype=str,
                        usecols=["trip_id", "route_id", "service_id", "shape_id", "trip_headsign", "direction_id"])
    trips = trips[trips.service_id.str.startswith(iso) & trips.route_id.isin(lines)]
    top = (trips.groupby(["route_id", "direction_id", "shape_id"]).size().rename("n").reset_index()
           .sort_values("n", ascending=False).drop_duplicates(["route_id", "direction_id"]))
    sample = trips[trips.shape_id.isin(top.shape_id)].drop_duplicates("shape_id")

    shp = pd.read_csv(gtfs / "shapes.txt", dtype={"shape_id": str})
    shp = shp[shp.shape_id.isin(set(top.shape_id))].sort_values(["shape_id", "shape_pt_sequence"])
    xy = sp.gpd.points_from_xy(shp.shape_pt_lon, shp.shape_pt_lat, crs=4326).to_crs(sp.CRS)
    shp["x"], shp["y"] = xy.x, xy.y
    geom = {sid: shapely.linestrings(g[["x", "y"]].to_numpy()) for sid, g in shp.groupby("shape_id")}

    keep = set(sample.trip_id)
    st = pd.concat(ch[ch.trip_id.isin(keep)] for ch in pd.read_csv(
        gtfs / "stop_times.txt", dtype={"trip_id": str, "stop_id": str},
        usecols=["trip_id", "stop_sequence", "stop_id"], chunksize=2_000_000)).sort_values(["trip_id", "stop_sequence"])
    stops = pd.read_csv(gtfs / "stops.txt", dtype={"stop_id": str})
    sxy = sp.gpd.points_from_xy(stops.stop_lon, stops.stop_lat, crs=4326).to_crs(sp.CRS)
    spt = dict(zip(stops.stop_id, sxy)); sname = dict(zip(stops.stop_id, stops.stop_name))

    out = {}
    for r in top.itertuples():
        line = geom[r.shape_id]
        trip = sample.loc[sample.shape_id == r.shape_id].iloc[0]
        ids = st.loc[st.trip_id == trip.trip_id, "stop_id"]
        stop_m = [(sname[s], float(shapely.line_locate_point(line, spt[s]))) for s in ids if s in spt]
        out[(r.route_id, r.direction_id)] = {"line": line, "headsign": trip.trip_headsign, "stops": stop_m,
                                             "trips": int(r.n)}
    return out


def assign_direction(iv: pd.DataFrame, shapes: dict, route: str):
    """Return iv rows of one route with columns dir, m0, m1 (distance along that direction's shape)."""
    dirs = [d for (r, d) in shapes if r == route]
    if not dirs:
        return None
    p0 = shapely.points(iv.x0.to_numpy(), iv.y0.to_numpy())
    p1 = shapely.points(iv.x1.to_numpy(), iv.y1.to_numpy())
    res = {}
    for d in dirs:
        L = shapes[(route, d)]["line"]
        res[d] = (shapely.line_locate_point(L, p0), shapely.line_locate_point(L, p1),
                  np.maximum(shapely.distance(p0, L), shapely.distance(p1, L)))
    # progress along direction-0 shape (or the only one), smoothed per vehicle
    ref = dirs[0]
    prog = pd.Series(res[ref][1] - res[ref][0], index=iv.index)
    veh = iv.VehicleNumber
    smooth = prog.groupby(veh).transform(lambda s: s.rolling(DIR_WINDOW, center=True, min_periods=1).sum())
    if len(dirs) == 2:
        other = dirs[1]
        dsel = np.where(smooth.to_numpy() >= 0, ref, other)
    else:
        dsel = np.where(smooth.to_numpy() >= 0, ref, None)
    m0 = np.full(len(iv), np.nan); m1 = m0.copy(); dist = m0.copy()
    for d in dirs:
        k = dsel == d
        m0[k], m1[k], dist[k] = res[d][0][k], res[d][1][k], res[d][2][k]
    out = iv.assign(dir=dsel, m0=m0, m1=m1, dshape=dist)
    out = out[out.dir.notna() & (out.dshape <= MATCH_M)]
    # moving backwards along the chosen direction by > 30 m = mis-assigned / short-turn: drop
    return out[(out.m1 - out.m0) > -30]


def spread(iv: pd.DataFrame) -> pd.DataFrame:
    """Spread each interval's time over the BIN_M bins it covers (uniform speed within the interval)."""
    lo = np.minimum(iv.m0.to_numpy(), iv.m1.to_numpy()); hi = np.maximum(iv.m0.to_numpy(), iv.m1.to_numpy())
    b0 = (lo // BIN_M).astype(int); b1 = (hi // BIN_M).astype(int)
    n = b1 - b0 + 1
    rep = np.repeat(np.arange(len(iv)), n)
    b = b0[rep] + (np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n))
    seg = np.maximum(hi - lo, 1e-9)
    ov = np.clip(np.minimum(hi[rep], (b + 1) * BIN_M) - np.maximum(lo[rep], b * BIN_M), 0, None)
    frac = np.where((hi - lo)[rep] < 1, 1.0, ov / seg[rep])
    return pd.DataFrame({"bin": b, "t_s": iv.dt.to_numpy()[rep] * frac, "d_m": (hi - lo)[rep] * frac,
                         "why": iv.why.to_numpy()[rep], "run": iv.run.to_numpy()[rep]})


def profile(pieces: pd.DataFrame) -> pd.DataFrame:
    g = pieces.groupby("bin")
    prof = pd.DataFrame({"runs": g.run.nunique(), "t_s": g.t_s.sum(), "d_m": g.d_m.sum()})
    for w in ["at_stop", "stuck"]:
        prof[f"{w}_s"] = pieces[pieces.why == w].groupby("bin").t_s.sum()
    prof = prof.fillna(0)
    prof["kmh"] = prof.d_m / prof.t_s * 3.6
    prof["at_stop_per_run"] = prof.at_stop_s / prof.runs
    prof["stuck_per_run"] = prof.stuck_s / prof.runs
    prof["sec_per_run"] = prof.t_s / prof.runs
    return prof.reindex(range(int(prof.index.max()) + 1))


def chart(prof, meta, mode, route, d, span, out):
    p = prof[prof.runs >= MIN_RUNS]
    if len(p) < 5:
        return False
    km = (p.index + 0.5) * BIN_M / 1000
    total_km = meta["line"].length / 1000
    full = prof.reindex(range(int(np.ceil(meta["line"].length / BIN_M))))
    avg = full.d_m.sum() / full.t_s.sum() * 3.6
    run_min = full.sec_per_run.sum() / 60
    stuck_min = full.stuck_per_run.sum() / 60
    runs = int(p.runs.median())

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 6.6), height_ratios=[1.5, 1], sharex=True, facecolor=SURFACE)
    for a in (a1, a2):
        a.set_facecolor(SURFACE)
        for s in ["top", "right", "left"]:
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9, length=0)
        a.grid(axis="y", color=GRID, linewidth=1); a.set_axisbelow(True)

    # top: speed strip (line + wash, like the gpx report)
    y = p.kmh.clip(upper=60).to_numpy()
    a1.fill_between(km, 0, y, color=WASH, step="mid", linewidth=0)
    a1.step(km, y, where="mid", color=RUN, linewidth=2)
    a1.axhline(REF_KMH, color=INK2, alpha=0.5, linewidth=1)
    a1.text(total_km, REF_KMH + 1.2, f"{REF_KMH} km/h reference", color=INK2, fontsize=9, ha="right")
    for name, m in meta["stops"]:
        a1.axvline(m / 1000, color=DWELL, alpha=0.28, linewidth=3, zorder=0)
    a1.set_ylim(0, 60); a1.set_yticks([0, 20, 40, 60])
    a1.set_ylabel("km/h", color=MUTED, fontsize=9)
    a1.set_title(f"{mode} {route} → {meta['headsign']}  ·  avg {avg:.1f} km/h  ·  {run_min:.0f} min per run, "
                 f"{stuck_min:.1f} min of it stuck  ·  {runs} runs ({span})", loc="left", color=INK, fontsize=12)

    # bottom: seconds per run spent at platforms / stuck, per bin
    w = BIN_M / 1000 * 0.8
    a2.bar(km, p.at_stop_per_run, width=w, color=DWELL, edgecolor=SURFACE, linewidth=0.8, label="At platform")
    a2.bar(km, p.stuck_per_run, width=w, bottom=p.at_stop_per_run, color=STUCK, edgecolor=SURFACE, linewidth=0.8,
           label="Stuck away from platform")
    a2.set_ylabel("s per run", color=MUTED, fontsize=9)
    a2.legend(frameon=False, fontsize=9, loc="upper left", ncols=2, labelcolor=INK)
    # label the 3 worst stuck bins
    for b, r in p.nlargest(3, "stuck_per_run").iterrows():
        if r.stuck_per_run < 5:
            continue
        near = min(meta["stops"], key=lambda s: abs(s[1] - (b + 0.5) * BIN_M))[0]
        a2.annotate(f"{r.stuck_per_run:.0f} s · {near}", ((b + 0.5) * BIN_M / 1000, r.at_stop_per_run + r.stuck_per_run),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    a2.set_ylim(0, max(10, (p.at_stop_per_run + p.stuck_per_run).max() * 1.25))

    # stop names along the x axis
    a2.set_xticks([m / 1000 for _, m in meta["stops"]], [n for n, _ in meta["stops"]],
                  rotation=90, fontsize=7.5, color=INK2)
    a2.set_xlim(0, total_km)
    sec = a2.secondary_xaxis("top"); sec.set_xticks(range(int(total_km) + 1)); sec.tick_params(colors=MUTED, labelsize=8, length=0)
    sec.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--mode", default="Tram", choices=["Tram", "Bus"])
    ap.add_argument("--lines", nargs="*")
    ap.add_argument("--hours", nargs=2, type=int, metavar=("FROM", "TO"))
    a = ap.parse_args()
    csv = sp.DATA_DIR / f"positions_{a.date}.csv" if a.date else sorted(sp.DATA_DIR.glob("positions_*.csv"))[-1]
    day = csv.stem.split("_")[1]
    cache = sp.CACHE_DIR / f"intervals_{day}.pkl"
    iv = pd.read_pickle(cache) if cache.exists() else sp.build(day)[0]
    iv = iv[(iv.vehicle_type == a.mode) & iv.why.isin(SERVICE)]
    if a.hours:
        iv = iv[(iv.t.dt.hour >= a.hours[0]) & (iv.t.dt.hour < a.hours[1])]
    lines = a.lines or sorted(iv.Lines.unique())
    span = f"{iv.t.min():%H:%M}–{iv.t.max():%H:%M}"
    print(f"{a.mode}: {len(lines)} lines, {len(iv):,} intervals ({span})")

    shapes = main_shapes(sp.latest_gtfs(), day, set(lines))
    out_dir = sp.DATA_DIR / f"route_profiles_{day}"; out_dir.mkdir(exist_ok=True)
    tag = f"_h{a.hours[0]:02d}-{a.hours[1]:02d}" if a.hours else ""
    rows, n = [], 0
    for route in lines:
        r = iv[iv.Lines == route].sort_values(["VehicleNumber", "t"])
        r = assign_direction(r, shapes, route)
        if r is None or r.empty:
            continue
        # a run = one vehicle travelling one direction without a long gap
        brk = (r.VehicleNumber.ne(r.VehicleNumber.shift()) | r.dir.ne(r.dir.shift())
               | (r.t - r.t.shift()).dt.total_seconds().gt(300))
        r = r.assign(run=brk.cumsum())
        for d, rd in r.groupby("dir"):
            prof = profile(spread(rd))
            meta = shapes[(route, d)]
            n += chart(prof, meta, a.mode, route, d, span, out_dir / f"{a.mode.lower()}_{route}_dir{d}{tag}.png")
            rows.append(prof.dropna(subset=["runs"]).reset_index().assign(mode=a.mode, line=route, dir=d,
                                                                           headsign=meta["headsign"]))
    if rows:
        pd.concat(rows).round(1).to_csv(sp.DATA_DIR / f"route_profiles_{a.mode.lower()}_{day}{tag}.csv", index=False)
    print(f"Saved {n} charts to {out_dir}")


if __name__ == "__main__":
    main()
