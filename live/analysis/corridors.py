"""
Step 7b — junction MOVEMENTS, street CORRIDORS, slow stretches, capacity and a time-sliced speed grid.

Junction movements
  * OSM signal nodes within 18 m of a tram shape, clustered (DBSCAN 45 m) -> one junction.
  * Per shape through a junction: entry arm = street + compass of the track 100 m before the first signal node,
    exit arm = 100 m after the last one. Movement = (junction, entry arm, exit arm), pooled over all lines.
  * A stand-still away from a platform belongs to the first junction downstream within 120 m (each counted once).
  * Per movement: trams through, stop rate, standing s per tram, tram-h, passenger-h (load-factor assumption).

Corridors
  * Streets carrying > CORR_MIN_TKM tram-km/day, split into spatially separate parts, >= 1.2 km long.
  * Reference line = the longest observed shape run along the part, 0 = end nearer the city centre (PKiN).
  * Every interval of every line on that street is placed on the reference (distance s, outbound / inbound),
    giving speed profiles in 50 m bins, an hour x distance speed grid and a time split per km.

Slow stretches: 40 m directional cells where the 85th-percentile MOVING speed is < 25 km/h (lost time vs 35 km/h), away from stops and
termini (speed restrictions, sharp curves, switches, crawling queues).

Output: _output/corridors_<date>.json (for report_corridors.py), _output/movements_<date>.csv,
        _output/corridors_<date>.csv, _output/slow_<date>.csv
Usage:  py -3.11 live/analysis/corridors.py      (needs match_trips.py, network.py, fetch_signals.py)
"""

import json

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from sklearn.cluster import DBSCAN

import ta_config as C
import ta_common as T
from signals import load_signals
from places import road_names

STEP = 10                      # network.py sample step
J_SNAP, J_EPS = 18, 45
ARM_M = 100
UPSTREAM_M, DOWN_M = 120, 20
CORR_MIN_TKM, CORR_MIN_LEN = 600, 1200
BIN_M = 50
ANIM_CELL, ANIM_SLOT = 80, 1800
SLOW_KMH, SLOW_REF_KMH = 25, 35
CENTRE = (637600, 487350)      # PKiN in EPSG:2180
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def compass(dx, dy):
    return COMPASS[int(((np.degrees(np.arctan2(dx, dy)) % 360) + 22.5) // 45) % 8]


GIVEN = set("""Jana Stefana Tytusa Juliusza Władysława Zygmunta Marcina Adama Tadeusza Krystyny Jerzego Romana Artura
Józefa Marii Ludwika Mieczysława Wincentego Gabriela Kazimierza Bolesława Henryka Ignacego Aleksandra Jacka Franciszka
Wojciecha Andrzeja Karola Antoniego Piotra Michała Stanisława Juliana Fryderyka Emilii Haliny Leopolda Ryszarda Cypriana
Tomasza Feliksa Hugona Witolda Jarosława Lucjana Edwarda Eugeniusza Floriana Wacława Władysławy Gustawa""".split())
PREFIX = {"Aleja": "Al.", "Aleje": "Al.", "Plac": "Pl.", "Rondo": "Rondo", "Most": "Most"}
TITLES = {"Generała", "Generał", "Świętego", "Świętych", "Księdza", "Księcia", "Marszałka", "Prymasa", "Kardynała"}


def short(name):
    """Aleja Jana III Sobieskiego -> Sobieskiego, Aleja Jana Pawła II stays, Rondo Romana Dmowskiego -> Rondo Dmowskiego."""
    if not name:
        return "—"
    w = name.replace("„", "").replace("”", "").split()
    pre = []
    if w and w[0] in PREFIX:
        pre = [PREFIX[w[0]]] if w[0] not in ("Aleja", "Aleje") or len(w) == 2 or w[-1] in ("II",) else ["Al."]
        w = w[1:]
    w = [x for x in w if x not in TITLES]
    if w and w[-1] in ("II", "III"):
        pass
    else:
        while len(w) > 1 and w[0] in GIVEN:
            w = w[1:]
        w = [x for x in w if x not in ("II", "III")] or w
        if w and w[0] == "Pawła" and len(w) > 1:
            w = w[1:]
    if pre and pre[0] == "Al." and w and w[-1] == "Sobieskiego":
        pre = []
    return " ".join(pre + w)


def lf(t):
    return np.where(T.peak(t), C.LOAD_FACTOR["peak"], C.LOAD_FACTOR["offpeak"])


def main():
    trips, st, shp = T.load_gtfs()
    geoms = T.shape_lines(shp)
    obs = pd.read_parquet(C.CACHE / f"trips_obs_{C.DATE}.parquet")
    obs["places"] = obs.places.fillna(200)
    ns = pd.read_parquet(C.CACHE / f"net_samples_{C.DATE}.parquet")
    ns["idx"] = np.rint(ns.m / STEP).astype(int)
    iv = pd.read_parquet(C.CACHE / f"intervals_{C.DATE}.parquet")
    iv = iv[~iv.anomaly & (iv.dm < 250)].join(obs.set_index("trip_id")[["shape_id", "route_id", "places"]], on="trip_id")
    iv["idx"] = np.rint((iv.m + iv.dm / 2) / STEP).astype(int)
    iv = iv.merge(ns[["shape_id", "idx", "street"]], on=["shape_id", "idx"], how="left")
    print(f"{len(iv):,} intervals")

    rep_trip = obs.groupby("shape_id").trip_id.first()
    stop_m = {k: g.m.to_numpy() for k, g in st[st.trip_id.isin(rep_trip.values)].groupby("trip_id")}
    shape_stops = {sid: stop_m[t] for sid, t in rep_trip.items()}
    st_sh = st[st.trip_id.isin(rep_trip.values)].merge(rep_trip.rename("trip_id").reset_index(), on="trip_id")

    # ===================================================================== junctions
    sig = load_signals()
    allpts = shapely.points(ns.x, ns.y)
    near = shapely.STRtree(allpts).query(sig, predicate="dwithin", distance=J_SNAP)
    sig = sig[np.unique(near[0])]
    sx, sy = shapely.get_x(sig), shapely.get_y(sig)
    lab = DBSCAN(eps=J_EPS, min_samples=1).fit_predict(np.column_stack([sx, sy]))
    J = pd.DataFrame({"j": lab, "x": sx, "y": sy}).groupby("j").agg(x=("x", "mean"), y=("y", "mean"), nodes=("x", "size"))
    J["name"] = [" × ".join(short(p) for p in n.split(" × ")) for n in road_names(gpd.GeoSeries(shapely.points(J.x, J.y), crs=C.CRS))]
    print(f"{len(J)} signalised junctions on the tram network")

    by_shape = {k: g for k, g in ns.groupby("shape_id")}
    jrows = []          # (shape, junction, lo, hi, entry, exit)
    for sid, g in by_shape.items():
        L = geoms[sid]
        d = shapely.distance(L, sig)
        k = np.flatnonzero(d <= J_SNAP)
        if not len(k):
            continue
        mm = shapely.line_locate_point(L, sig[k])
        sm = shape_stops[sid]
        for j, grp in pd.DataFrame({"j": lab[k], "m": mm}).groupby("j"):
            lo, hi = grp.m.min(), grp.m.max()
            if lo - UPSTREAM_M < sm[0] + C.TERMINUS_M or hi > sm[-1] - C.TERMINUS_M or hi - lo > 250:
                continue
            cx, cy = J.at[j, "x"], J.at[j, "y"]
            a = g.iloc[min(len(g) - 1, max(0, int((lo - ARM_M) / STEP)))]
            b = g.iloc[min(len(g) - 1, int((hi + ARM_M) / STEP))]
            plat = bool(np.any((sm >= lo - UPSTREAM_M - C.STOP_AFTER_M) & (sm <= hi + C.STOP_BEFORE_M)))
            jrows.append((sid, j, lo, hi, short(a.street), compass(a.x - cx, a.y - cy),
                          short(b.street), compass(b.x - cx, b.y - cy), plat))
    JS = pd.DataFrame(jrows, columns=["shape_id", "j", "lo", "hi", "in_st", "in_c", "out_st", "out_c", "plat"])

    # assign line stand-stills to the first junction downstream (within UPSTREAM_M)
    stand = iv[iv.stand & (iv.zone == "line")]
    stand = stand.assign(pax_s=stand.dt * stand.places * lf(stand.t0))
    parts = []
    for sid, js in JS.groupby("shape_id"):
        s = stand[stand.shape_id == sid]
        if s.empty:
            continue
        js = js.sort_values("lo")
        lo, hi = js.lo.to_numpy(), js.hi.to_numpy()
        k = np.searchsorted(hi + DOWN_M, s.m.to_numpy())            # first junction whose end is ahead
        ok = k < len(js)
        k2 = np.where(ok, k, 0)
        ok &= s.m.to_numpy() >= lo[k2] - UPSTREAM_M
        parts.append(s[ok].assign(jrow=js.index.to_numpy()[k2[ok]]))
    sa = pd.concat(parts)
    per_run = sa.groupby(["jrow", "trip_id"]).agg(stand_s=("dt", "sum"), pax_s=("pax_s", "sum"))

    # passes per (shape, junction) = observed trips covering the window
    pas = []
    for r in JS.itertuples():
        o = obs[(obs.shape_id == r.shape_id) & (obs.m_first < r.lo - UPSTREAM_M) & (obs.m_last > r.hi + DOWN_M)]
        pas.append(pd.DataFrame({"jrow": r.Index, "trip_id": o.trip_id.to_numpy(), "route_id": o.route_id.to_numpy(),
                                 "places": o.places.to_numpy(), "t": (o.start_s.to_numpy() + o.end_s.to_numpy()) / 2,
                                 "dest": (o.route_id + " → " + o.trip_headsign).to_numpy()}))
    P = pd.concat(pas).join(per_run, on=["jrow", "trip_id"]).fillna({"stand_s": 0, "pax_s": 0})
    P = P.join(JS[["j", "in_st", "in_c", "out_st", "out_c", "plat"]], on="jrow")
    P["peak"] = T.peak(P.t)
    P["stopped"] = P.stand_s > 0

    mk = ["j", "in_st", "in_c", "out_st", "out_c"]
    M = P.groupby(mk).agg(trams=("trip_id", "size"), stop_rate=("stopped", "mean"), stand_s=("stand_s", "mean"),
                          tram_h=("stand_s", lambda v: v.sum() / 3600), pax_h=("pax_s", lambda v: v.sum() / 3600),
                          plat=("plat", "max"), lines=("route_id", lambda v: sorted(set(v), key=lambda x: (len(x), x))),
                          dest=("dest", lambda v: list(v.value_counts().index[:8])))
    pk = P[P.peak].groupby(mk).agg(peak_stop_rate=("stopped", "mean"), peak_stand_s=("stand_s", "mean"))
    M = M.join(pk).reset_index()
    M = M[M.trams >= 30]
    M["wait_if_stopped_s"] = M.stand_s / M.stop_rate.replace(0, np.nan)
    M = M.join(J[["name", "x", "y"]], on="j")
    M["movement"] = np.where(M.in_st == M.out_st, M.in_st + " " + M.in_c + "→" + M.out_c,
                             M.in_st + " (" + M.in_c + ") → " + M.out_st + " (" + M.out_c + ")")
    M = M.sort_values("tram_h", ascending=False)
    M.assign(lines=M.lines.str.join(" "), dest=M.dest.str.join("; ")).round(3).to_csv(C.OUT / f"movements_{C.DATE}.csv", index=False)
    JT = M.groupby("j").agg(tram_h=("tram_h", "sum"), pax_h=("pax_h", "sum"), trams=("trams", "sum"))
    JT = JT.join(J).sort_values("tram_h", ascending=False)
    tot_line = stand.dt.sum() / 3600
    print(f"line stand-stills {tot_line:.0f} h, at signalised junctions {M.tram_h.sum():.0f} h")
    print(M.head(25)[["name", "movement", "lines", "trams", "stop_rate", "stand_s", "tram_h"]].to_string())

    # junction diagrams for the top junctions
    diagrams = []
    for j, jr in JT.head(16).iterrows():
        cx, cy = jr.x, jr.y
        R = 230
        sel = ns[(ns.x - cx) ** 2 + (ns.y - cy) ** 2 < R * R]
        tracks, seen = [], set()
        for sid, g in sel.groupby("shape_id"):
            for run in np.split(g, np.flatnonzero(np.diff(g.idx.to_numpy()) > 1) + 1):
                if len(run) <= 2:
                    continue
                cells = set(zip(np.floor(run.x / 4).astype(int), np.floor(run.y / 4).astype(int)))
                if len(cells - seen) < 0.25 * len(cells):        # same rails already drawn
                    continue
                seen |= {(a + i, b + k) for a, b in cells for i in (-1, 0, 1) for k in (-1, 0, 1)}
                tracks.append([[round(x - cx), round(cy - y)] for x, y in zip(run.x[::2], run.y[::2])])
        mv = []
        for r in M[M.j == j].itertuples():
            cand = JS[(JS.j == j) & (JS.in_st == r.in_st) & (JS.in_c == r.in_c) & (JS.out_st == r.out_st) & (JS.out_c == r.out_c)]
            sid = obs[obs.shape_id.isin(cand.shape_id)].shape_id.value_counts().index[0]
            c = cand[cand.shape_id == sid].iloc[0]
            g = by_shape[sid]
            g = g[(g.m >= c.lo - 160) & (g.m <= c.hi + 140)]
            g = g[(g.x - cx) ** 2 + (g.y - cy) ** 2 < R * R]
            mv.append({"path": [[round(x - cx), round(cy - y)] for x, y in zip(g.x, g.y)],
                       "label": r.movement, "lines": r.lines, "trams": int(r.trams), "stop_rate": round(r.stop_rate, 3),
                       "stand_s": round(r.stand_s, 1), "tram_h": round(r.tram_h, 2), "pax_h": round(r.pax_h, 1),
                       "peak_stop_rate": None if pd.isna(r.peak_stop_rate) else round(r.peak_stop_rate, 3),
                       "wait_s": None if pd.isna(r.wait_if_stopped_s) else round(r.wait_if_stopped_s, 0), "plat": bool(r.plat),
                       "dest": r.dest})
        names = sorted(set(sel.street.dropna().map(short)))
        diagrams.append({"j": int(j), "name": jr["name"], "x": cx, "y": cy, "tram_h": round(jr.tram_h, 1),
                         "pax_h": round(jr.pax_h, 0), "trams": int(jr.trams), "tracks": tracks, "moves": mv, "streets": names})

    # ===================================================================== corridors
    nt = obs.groupby("shape_id").size()
    ns["runs"] = ns.shape_id.map(nt).fillna(0)
    tkm = ns.groupby("street").runs.sum() * STEP / 1000
    corridors = []
    ivc_parts = []
    for street in tkm[tkm > CORR_MIN_TKM].index:
        g = ns[ns.street == street]
        cl = DBSCAN(eps=60, min_samples=3).fit_predict(g[["x", "y"]].to_numpy())
        for c in set(cl) - {-1}:
            gc = g[cl == c]
            # longest contiguous run of one shape = reference
            best = None
            for sid, h in gc.groupby("shape_id"):
                for run in np.split(h, np.flatnonzero(np.diff(h.idx.to_numpy()) > 3) + 1):
                    if best is None or len(run) > len(best):
                        best = run
            if best is None or len(best) * STEP < CORR_MIN_LEN:
                continue
            ref = shapely.linestrings(best[["x", "y"]].to_numpy())
            d0 = np.hypot(best.x.iloc[0] - CENTRE[0], best.y.iloc[0] - CENTRE[1])
            d1 = np.hypot(best.x.iloc[-1] - CENTRE[0], best.y.iloc[-1] - CENTRE[1])
            if d1 < d0:
                ref = shapely.reverse(ref)
            # place every sample of this street part on the reference
            p = shapely.points(gc.x, gc.y)
            dist = shapely.distance(ref, p)
            s = shapely.line_locate_point(ref, p)
            t1 = shapely.line_interpolate_point(ref, np.minimum(s + 5, ref.length))
            t0 = shapely.line_interpolate_point(ref, np.maximum(s - 5, 0))
            tang = np.degrees(np.arctan2(shapely.get_x(t1) - shapely.get_x(t0), shapely.get_y(t1) - shapely.get_y(t0))) % 360
            dh = (gc.heading.to_numpy() - tang + 180) % 360 - 180
            ok = (dist < 25) & (np.abs(np.abs(dh) - 90) > 45)
            cid = len(corridors)
            samp = pd.DataFrame({"shape_id": gc.shape_id.to_numpy()[ok], "idx": gc.idx.to_numpy()[ok],
                                 "s": s[ok], "out": np.abs(dh[ok]) < 90, "cid": cid})
            end_pt = shapely.get_point(ref, -1)
            st_pt = shapely.get_point(ref, 0)
            mid = shapely.line_interpolate_point(ref, ref.length / 2)
            comp = compass(shapely.get_x(end_pt) - shapely.get_x(st_pt), shapely.get_y(end_pt) - shapely.get_y(st_pt))
            corridors.append({"cid": cid, "street": short(street), "street_full": street, "len_m": ref.length,
                              "ref": ref, "out_compass": comp, "mid": (shapely.get_x(mid), shapely.get_y(mid))})
            ivc_parts.append(samp)
    CS = pd.concat(ivc_parts).drop_duplicates(["shape_id", "idx"])
    # name parts of the same street by their position
    cdf = pd.DataFrame(corridors)
    for street, g in cdf.groupby("street"):
        if len(g) > 1:
            for i in g.index:
                mx, my = cdf.at[i, "mid"]
                cdf.at[i, "street"] = f"{street} ({compass(mx - CENTRE[0], my - CENTRE[1])})"
    print(f"{len(cdf)} corridors")

    ivc = iv[iv.zone != "terminus"].merge(CS, on=["shape_id", "idx"])
    ivc["kind"] = np.where(~ivc.stand, "moving", np.where(ivc.zone == "platform", "platform", "line"))
    ivc["hour"] = (ivc.t0 // 3600).astype(int)
    ivc["bin"] = (ivc.s // BIN_M).astype(int)
    ivc["pax_s"] = ivc.dt * ivc.places * lf(ivc.t0)

    stops_s = st_sh.copy()
    stops_s["idx"] = np.rint(stops_s.m / STEP).astype(int)
    stops_s = stops_s.merge(CS, on=["shape_id", "idx"])

    out_c = []
    for c in cdf.itertuples():
        q = ivc[ivc.cid == c.cid]
        if q.empty:
            continue
        nb = int(np.ceil(c.len_m / BIN_M))
        dirs = {}
        for o, qd in q.groupby("out"):
            runs = qd.trip_id.nunique()
            if runs < 50:
                continue
            day = qd[(qd.hour >= 6) & (qd.hour < 21)]
            gb = day.groupby("bin").agg(dm=("dm", "sum"), dt=("dt", "sum"))
            gp = day[T.peak(day.t0)].groupby("bin").agg(dm=("dm", "sum"), dt=("dt", "sum"))
            gm = day[day.kind == "moving"].groupby("bin").kmh.quantile(.85)
            sp = lambda gg: [None if (b not in gg.index or gg.dt[b] < 120) else round(gg.dm[b] / gg.dt[b] * 3.6, 1) for b in range(nb)]
            hb = (qd.s // 100).astype(int)
            hm = qd.assign(hb=hb).groupby(["hour", "hb"]).agg(dm=("dm", "sum"), dt=("dt", "sum"))
            hours = list(range(5, 24))
            grid = [[None if ((h, b) not in hm.index or hm.dt[(h, b)] < 60) else round(hm.dm[(h, b)] / hm.dt[(h, b)] * 3.6, 1)
                     for b in range(int(np.ceil(c.len_m / 100)))] for h in hours]
            pk = qd[(qd.t0 >= 7 * 3600) & (qd.t0 < 9 * 3600)].drop_duplicates(["trip_id", "bin"])
            pk = pk.groupby("bin").agg(n=("trip_id", "size"), places=("places", "sum")).sort_values("n").iloc[-1]   # busiest section
            split = qd.groupby("kind").dt.sum()
            km = qd.dm.sum() / 1000
            stp = stops_s[(stops_s.cid == c.cid) & (stops_s.out == o)].groupby("stop_name").s.median().sort_values()
            # covered length: bins actually observed
            dirs["out" if o else "in"] = {
                "runs": int(runs), "lines": sorted(qd.route_id.unique().tolist(), key=lambda x: (len(x), x)),
                "speed": sp(gb), "speed_peak": sp(gp),
                "cruise": [None if b not in gm.index else round(gm[b], 1) for b in range(nb)],
                "grid": grid, "hours": hours,
                "min_per_km": {k: round(split.get(k, 0) / 60 / km, 3) for k in ("moving", "platform", "line")},
                "avg_kmh": round(km / (split.sum() / 3600), 1),
                "peak_tph": round(pk.n / 2, 1), "peak_places_h": int(pk.places / 2),
                "pax_h_line": round(qd[qd.kind == "line"].pax_s.sum() / 3600, 0),
                "tram_h_line": round(split.get("line", 0) / 3600, 1),
                "stops": [[n, round(v, 0)] for n, v in stp.items()],
            }
        if not dirs:
            continue
        ref = c.ref
        simp = shapely.simplify(ref, 15)
        jj = J[shapely.distance(ref, shapely.points(J.x, J.y)) < 25]
        out_c.append({"cid": c.cid, "street": c.street, "len_m": round(c.len_m), "out_compass": c.out_compass,
                      "geom": [[round(x), round(y)] for x, y in shapely.get_coordinates(simp)],
                      "signals": sorted(round(v) for v in shapely.line_locate_point(ref, shapely.points(jj.x, jj.y))),
                      "dirs": dirs})
    # corridor table
    crow = []
    for c in out_c:
        for dname, d in c["dirs"].items():
            crow.append({"corridor": c["street"], "dir": dname, "len_km": c["len_m"] / 1000, "runs": d["runs"],
                         "lines": " ".join(d["lines"]), "avg_kmh": d["avg_kmh"], **{f"min_km_{k}": v for k, v in d["min_per_km"].items()},
                         "tram_h_line": d["tram_h_line"], "pax_h_line": d["pax_h_line"],
                         "peak_tph": d["peak_tph"], "peak_places_h": d["peak_places_h"]})
    CT = pd.DataFrame(crow).sort_values("avg_kmh")
    CT.to_csv(C.OUT / f"corridors_{C.DATE}.csv", index=False)
    print(CT.to_string())

    # ===================================================================== directional cells: capacity, slow, animation
    ivn = iv[iv.zone != "terminus"].copy()
    ivn["hs"] = (np.floor(((ivn.heading + 22.5) % 360) / 45)).astype(int)
    xm = ivn.x + np.sin(np.radians(ivn.heading)) * ivn.dm / 2
    ym = ivn.y + np.cos(np.radians(ivn.heading)) * ivn.dm / 2
    # slow cruise cells (40 m)
    ivn["cx"], ivn["cy"] = np.floor(xm / 40).astype(int), np.floor(ym / 40).astype(int)
    mv = ivn[~ivn.stand]
    cr = mv.groupby(["cx", "cy", "hs"]).agg(p85=("kmh", lambda v: v.quantile(.85)), n=("kmh", "size"),
                                            trips=("trip_id", "nunique"), street=("street", lambda v: v.mode().iat[0] if v.notna().any() else ""),
                                            lines=("route_id", lambda v: " ".join(sorted(set(v), key=lambda x: (len(x), x)))))
    cr = cr[cr.trips >= 60]
    # distance of each cell to the nearest stop / terminus
    sp_pts = []
    for sid in obs.shape_id.unique():
        L = geoms[sid]
        sp_pts.append(shapely.line_interpolate_point(L, shape_stops[sid]))
    sp_pts = np.concatenate(sp_pts)
    cpts = shapely.points((cr.index.get_level_values(0) + .5) * 40, (cr.index.get_level_values(1) + .5) * 40)
    dstop = shapely.STRtree(sp_pts).query_nearest(cpts, return_distance=True, all_matches=False)[1]
    cr["d_stop"] = dstop
    slow = cr[(cr.p85 < SLOW_KMH) & (cr.d_stop > 70)].reset_index()
    if len(slow):
        lab2 = DBSCAN(eps=60, min_samples=1).fit_predict(np.column_stack([slow.cx * 40, slow.cy * 40, slow.hs * 1000]))
        slow["grp"] = lab2
        SL = slow.groupby("grp").agg(cells=("p85", "size"), p85=("p85", "median"), trips=("trips", "max"),
                                      street=("street", lambda v: v.mode().iat[0]), lines=("lines", "first"),
                                      x=("cx", lambda v: (v.mean() + .5) * 40), y=("cy", lambda v: (v.mean() + .5) * 40),
                                      hs=("hs", "first"))
        SL["len_m"] = SL.cells * 40
        SL["lost_s_per_tram"] = SL.len_m / (SL.p85 / 3.6) - SL.len_m / (SLOW_REF_KMH / 3.6)
        SL["tram_h"] = SL.lost_s_per_tram * SL.trips / 3600
        SL["dir"] = SL.hs.map(lambda h: COMPASS[h])
        SL["street"] = SL.street.map(short)
        SL = SL[SL.len_m >= 80].sort_values("tram_h", ascending=False)
        SL.round(2).to_csv(C.OUT / f"slow_{C.DATE}.csv", index=False)
        print(SL.head(20).to_string())
    else:
        SL = pd.DataFrame()

    # animation + capacity on 80 m directional cells
    ivn["ax"], ivn["ay"] = np.floor(xm / ANIM_CELL).astype(int), np.floor(ym / ANIM_CELL).astype(int)
    ivn["slot"] = (ivn.t0 // ANIM_SLOT).astype(int)
    a = ivn[~(ivn.stand & (ivn.zone == "platform"))]
    A = a.groupby(["ax", "ay", "hs", "slot"]).agg(dm=("dm", "sum"), dt=("dt", "sum"))
    A = A[A.dt >= 30]
    A["kmh"] = A.dm / A.dt * 3.6
    slots = list(range(5 * 2, 23 * 2))
    cells = A.reset_index().groupby(["ax", "ay", "hs"])
    # capacity: places per hour 07-09 passing each cell (distinct trips)
    pk = ivn[(ivn.t0 >= 7 * 3600) & (ivn.t0 < 9 * 3600)].drop_duplicates(["ax", "ay", "hs", "trip_id"])
    cap = pk.groupby(["ax", "ay", "hs"]).agg(places=("places", "sum"), trams=("trip_id", "size"))
    lost = ivn[ivn.stand & (ivn.zone == "line")].assign(pax_s=lambda d: d.dt * d.places * lf(d.t0)).groupby(["ax", "ay", "hs"]).pax_s.sum() / 3600
    alls = ivn.groupby(["ax", "ay", "hs"]).agg(dm=("dm", "sum"), dt=("dt", "sum"), trips=("trip_id", "nunique"))
    alls = alls[alls.trips >= 40]
    code = "0123456789"
    bounds = [6, 9, 12, 15, 18, 21, 25, 30, 36]     # km/h -> bucket 0..9
    anim = []
    for key, r in alls.iterrows():
        try:
            g = cells.get_group(key).set_index("slot").kmh
        except KeyError:
            g = pd.Series(dtype=float)
        sstr = "".join("-" if s not in g.index else code[int(np.searchsorted(bounds, g[s]))] for s in slots)
        cp = cap.loc[key] if key in cap.index else None
        anim.append([int(key[0]), int(key[1]), int(key[2]), sstr,
                     0 if cp is None else int(cp.places / 2), 0 if cp is None else int(cp.trams / 2),
                     round(float(lost.get(key, 0)), 1), round(r["dm"] / r["dt"] * 3.6, 1)])
    print(f"{len(anim)} animation cells")

    # overview arrows: top movements
    arrows = []
    for r in M.head(60).itertuples():
        cand = JS[(JS.j == r.j) & (JS.in_st == r.in_st) & (JS.in_c == r.in_c) & (JS.out_st == r.out_st) & (JS.out_c == r.out_c)].iloc[0]
        g = by_shape[cand.shape_id]
        a0 = g.iloc[max(0, int((cand.lo - 260) / STEP))]
        a1 = g.iloc[max(0, int((cand.lo - 10) / STEP))]
        arrows.append({"j": int(r.j), "x0": round(a0.x), "y0": round(a0.y), "x1": round(a1.x), "y1": round(a1.y),
                       "name": r.name, "label": r.movement, "lines": r.lines, "trams": int(r.trams),
                       "stop_rate": round(r.stop_rate, 3), "stand_s": round(r.stand_s, 1), "tram_h": round(r.tram_h, 2),
                       "pax_h": round(r.pax_h, 1)})

    net = []
    for sid in obs.shape_id.value_counts().index:
        net.append([[round(x), round(y)] for x, y in shapely.get_coordinates(shapely.simplify(geoms[sid], 20))])

    data = {"date": C.DATE, "cell": ANIM_CELL, "slot_s": ANIM_SLOT, "slots": slots, "bounds": bounds,
            "anim": anim, "arrows": arrows, "diagrams": diagrams, "corridors": out_c, "net": net,
            "slow": SL.head(25).round(2).to_dict("records") if len(SL) else [],
            "movements": M.head(40).assign(lines=M.lines).drop(columns=["x", "y"]).round(3).to_dict("records"),
            "totals": {"line_h": round(tot_line, 0), "junction_h": round(M.tram_h.sum(), 0),
                       "junction_pax_h": round(M.pax_h.sum(), 0), "n_junctions": int(len(J)),
                       "n_movements": int(len(M))},
            "load_factor": C.LOAD_FACTOR}
    (C.OUT / f"corridors_{C.DATE}.json").write_text(json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o),
                                                              ensure_ascii=False), encoding="utf-8")
    print("wrote", C.OUT / f"corridors_{C.DATE}.json")


if __name__ == "__main__":
    main()
