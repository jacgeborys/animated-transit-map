"""
Step 1 — tie every tram GPS fix to its scheduled GTFS trip and measure it against the timetable.

How
  * Live data carries line + brigade; GTFS block_short_name is the brigade. For every brigade, each of its
    scheduled trips is a candidate for the fixes inside [start - 5 min, end + 25 min].
  * A fix is projected onto each candidate trip's OWN shape -> distance along (m), off-shape distance and
    lateness = gps time - scheduled time at that point. Pass 1 picks the smallest |lateness|; pass 2 re-picks
    by closeness to the trip's median lateness (separates the two directions on a shared street).
  * Per trip, fixes moving backwards are dropped; consecutive fixes form intervals that are classified
    moving / standing at a platform / standing elsewhere ("line" = junction, signal, queue) / terminus.

Output (_cache/):
  fixes_<date>.parquet       matched fixes (trip_id, veh, t, m, lat_s)
  intervals_<date>.parquet   classified intervals with position, heading, nearest stop
  passages_<date>.parquet    per trip & stop: scheduled vs actual departure, standing time at the platform
  trips_obs_<date>.parquet   per observed trip: vehicle, model, places, coverage, delays

Usage: python live/analysis/match_trips.py           (~5-10 min first time; GTFS + fixes are cached)
"""

import time

import numpy as np
import pandas as pd
import shapely

import ta_config as C
import ta_common as T


def sched_curve(st_trip: pd.DataFrame):
    """Piecewise-linear scheduled time along the shape (arrival then departure at each stop)."""
    m = st_trip.m.to_numpy()
    xp = np.column_stack([m, m + 0.01]).ravel()
    fp = np.column_stack([st_trip.arr_s.to_numpy(), st_trip.dep_s.to_numpy()]).ravel()
    return xp, fp


def candidates(fx, trips_b, st_by_trip, geoms):
    """All (fix, trip) pairs that are plausible for one brigade."""
    out = []
    t = fx.t.to_numpy()
    pts = shapely.points(fx.x.to_numpy(), fx.y.to_numpy())
    for tr in trips_b.itertuples():
        k = np.flatnonzero((t >= tr.start_s - C.EARLY_S) & (t <= tr.end_s + C.LATE_S))
        if not len(k):
            continue
        L = geoms[tr.shape_id]
        d = shapely.distance(pts[k], L)
        ok = d <= C.OFF_SHAPE_M
        k, d = k[ok], d[ok]
        if not len(k):
            continue
        m = shapely.line_locate_point(L, pts[k])
        xp, fp = sched_curve(st_by_trip[tr.trip_id])
        lat = t[k] - np.interp(m, xp, fp)
        ok = (lat >= -C.EARLY_S) & (lat <= C.LATE_S)
        out.append(pd.DataFrame({"fix": fx.index.to_numpy()[k[ok]], "trip_id": tr.trip_id,
                                 "m": m[ok], "d_off": d[ok], "lat_s": lat[ok]}))
    return pd.concat(out) if out else None


def assign(cand: pd.DataFrame) -> pd.DataFrame:
    c = cand.assign(score=cand.lat_s.abs())
    first = c.sort_values("score").drop_duplicates("fix")
    med = first.groupby("trip_id").lat_s.median()
    c["score"] = (c.lat_s - c.trip_id.map(med)).abs() + c.d_off   # metres ~ seconds, both small when right
    c = c[c.trip_id.isin(med.index)]
    return c.sort_values("score").drop_duplicates("fix")


def clean_trip(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("t")
    if g.veh.nunique() > 1:                          # two vehicles on one brigade at once: keep the main one
        g = g[g.veh == g.veh.value_counts().idxmax()]
    m = g.m.to_numpy()
    keep = m >= np.maximum.accumulate(m) - C.BACKTRACK_M
    g = g[keep]
    return g.assign(mc=np.maximum.accumulate(g.m.to_numpy()))


def crossing(mc, t, x):
    """Interpolated time at which the (monotone) trace first reaches x; NaN outside or across a gap."""
    i = np.searchsorted(mc, x, side="left")
    ok = (i > 0) & (i < len(mc))
    i = np.clip(i, 1, len(mc) - 1)
    m0, m1, t0, t1 = mc[i - 1], mc[i], t[i - 1], t[i]
    res = t0 + (x - m0) / np.maximum(m1 - m0, 1e-6) * (t1 - t0)
    return np.where(ok & (t1 - t0 <= C.MAX_DT), res, np.nan)


def heading(L, m):
    a = shapely.line_interpolate_point(L, np.maximum(m - 8, 0))
    b = shapely.line_interpolate_point(L, m + 8)
    return np.degrees(np.arctan2(shapely.get_y(b) - shapely.get_y(a), shapely.get_x(b) - shapely.get_x(a))) % 360


def analyse_trip(g, tr, st, L):
    """Intervals + stop passages for one observed trip."""
    t, mc = g.t.to_numpy(), g.mc.to_numpy()
    sm = st.m.to_numpy()
    # --- intervals
    dt = np.diff(t); dm = np.diff(mc)
    ok = (dt > 0) & (dt <= C.MAX_DT)
    m_mid = (mc[1:] + mc[:-1]) / 2
    kmh = dm / np.maximum(dt, 1e-6) * 3.6
    stand = kmh < C.STAND_KMH
    # nearest stop "ahead or at" (platform zone = [stop - 50, stop + 25])
    j = np.clip(np.searchsorted(sm, m_mid - C.STOP_AFTER_M), 0, len(sm) - 1)
    at_platform = (m_mid >= sm[j] - C.STOP_BEFORE_M) & (m_mid <= sm[j] + C.STOP_AFTER_M)
    zone = np.where(at_platform, "platform", "line")
    zone = np.where((m_mid <= sm[0] + C.TERMINUS_M) | (m_mid >= sm[-1] - C.TERMINUS_M), "terminus", zone)
    p = shapely.line_interpolate_point(L, m_mid)
    iv = pd.DataFrame({"trip_id": tr.trip_id, "t0": t[:-1], "dt": dt, "m": m_mid, "dm": dm, "kmh": kmh,
                       "stand": stand, "zone": zone, "stop_seq": st.stop_sequence.to_numpy()[j],
                       "x": shapely.get_x(p), "y": shapely.get_y(p), "heading": heading(L, m_mid)})[ok]
    # stand-still episodes (consecutive standing intervals) -> flag very long ones as anomalies
    ep = (~iv.stand | ~iv.stand.shift(fill_value=False)).cumsum()
    ep_len = iv.dt.groupby(ep).transform("sum").where(iv.stand, 0)
    iv["episode"] = ep.to_numpy()
    iv["anomaly"] = iv.stand & (ep_len > C.ANOMALY_MIN * 60)
    # --- stop passages: departure = leaving the platform zone
    dep_act = crossing(mc, t, sm + C.STOP_AFTER_M)
    arr_act = crossing(mc, t, sm - C.STOP_BEFORE_M)
    plat = iv[iv.stand & (iv.zone != "terminus") & at_platform[ok]].groupby("stop_seq").dt.sum()
    ps = pd.DataFrame({"trip_id": tr.trip_id, "stop_sequence": st.stop_sequence.to_numpy(), "stop_id": st.stop_id.to_numpy(),
                       "stop_name": st.stop_name.to_numpy(), "m": sm, "arr_sched": st.arr_s.to_numpy(),
                       "dep_sched": st.dep_s.to_numpy(), "arr_act": arr_act, "dep_act": dep_act})
    ps["platform_stand_s"] = ps.stop_sequence.map(plat).fillna(0)
    ps.loc[ps.arr_act.isna() | ps.dep_act.isna(), "platform_stand_s"] = np.nan
    return iv, ps


def main():
    t0 = time.time()
    trips, st, shp = T.load_gtfs()
    geoms = T.shape_lines(shp)
    # rescale GTFS shape_dist to the projected shape length (projection scale ~0.1 %)
    last = st.groupby("trip_id").m.max()
    scale = (trips.set_index("trip_id").shape_id.map(lambda s: geoms[s].length) / last).clip(0.97, 1.03)
    st["m"] = st.m * st.trip_id.map(scale)
    st_by_trip = {k: g for k, g in st.groupby("trip_id")}
    fx = T.load_fixes()
    print(f"GTFS: {len(trips):,} tram trips · live: {len(fx):,} fixes, {fx.veh.nunique()} trams  [{time.time()-t0:.0f}s]")

    trips_by_block = {k: g.sort_values("start_s") for k, g in trips.groupby(["route_id", "block_short_name"])}
    matched = []
    for (line, brig), f in fx.groupby(["line", "brigade"]):
        tb = trips_by_block.get((line, brig))
        if tb is None:
            continue
        c = candidates(f, tb, st_by_trip, geoms)
        if c is None:
            continue
        a = assign(c)
        matched.append(a.join(f[["veh", "t"]], on="fix"))
    m = pd.concat(matched, ignore_index=True)
    print(f"matched {len(m):,} of {len(fx):,} fixes ({len(m)/len(fx):.0%})  [{time.time()-t0:.0f}s]")

    tri = trips.set_index("trip_id")
    fixes, ivs, pss, obs = [], [], [], []
    for tid, g in m.groupby("trip_id"):
        g = clean_trip(g)
        if len(g) < 5:
            continue
        tr = tri.loc[tid].copy(); tr["trip_id"] = tid
        s = st_by_trip[tid]
        iv, ps = analyse_trip(g, tr, s, geoms[tr.shape_id])
        fixes.append(g[["trip_id", "veh", "t", "m", "lat_s"]]); ivs.append(iv); pss.append(ps)
        L = geoms[tr.shape_id].length
        obs.append({"trip_id": tid, "veh": g.veh.iloc[0], "n_fix": len(g), "t_first": g.t.iloc[0], "t_last": g.t.iloc[-1],
                    "m_first": g.mc.iloc[0], "m_last": g.mc.iloc[-1], "shape_len": L,
                    "coverage": (g.mc.iloc[-1] - g.mc.iloc[0]) / L, "lat_median_s": g.lat_s.median()})
    obs = pd.DataFrame(obs).merge(trips, on="trip_id")
    obs = obs.join(T.fleet_lookup(obs.veh))
    iv = pd.concat(ivs, ignore_index=True); ps = pd.concat(pss, ignore_index=True)

    pd.concat(fixes).to_parquet(C.CACHE / f"fixes_{C.DATE}.parquet")
    iv.to_parquet(C.CACHE / f"intervals_{C.DATE}.parquet")
    ps.to_parquet(C.CACHE / f"passages_{C.DATE}.parquet")
    obs.to_parquet(C.CACHE / f"trips_obs_{C.DATE}.parquet")

    good = obs.coverage >= 0.8
    print(f"\n{len(obs):,} of {len(trips):,} scheduled tram trips observed, {good.sum():,} with >= 80 % of the route")
    print(f"models seen: {obs.drop_duplicates('veh').model.value_counts().to_dict()}")
    print(f"planned fleet_type vs actual model (trips):\n{pd.crosstab(obs.fleet_type.fillna('-'), obs.model).to_string()}")
    z = iv[~iv.anomaly].groupby(["zone", "stand"]).dt.sum().div(3600).round(1)
    print(f"\nvehicle-hours by zone / standing:\n{z.to_string()}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
