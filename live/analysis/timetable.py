"""
Step 5 — live vs timetable: where delay builds up, how regular the headways are, dwell by tram model.

  segments   per line / direction / stop pair: scheduled vs median actual time (departure -> departure).
             actual >> scheduled = timetable too tight here (delay is born); actual << scheduled = padding.
  headways   per line / direction at the middle stop: excess waiting time (EWT) = actual average wait
             E[h²]/2E[h] minus scheduled average wait — what irregularity (bunching) costs a passenger.
  models     platform standing time per stop visit relative to the median at that stop & hour -> dwell
             ratio per tram model (all models serve the same stops, so this compares like with like).

Output (_output/): tt_segments_<date>.csv, tt_headways_<date>.csv, tt_models_<date>.csv, tt_delay_profile_<date>.csv
Usage: python live/analysis/timetable.py
"""

import numpy as np
import pandas as pd

import ta_config as C
import ta_common as T


def awt(h):
    h = np.asarray(h, float)
    return (h ** 2).sum() / (2 * h.sum()) if len(h) and h.sum() > 0 else np.nan


def main():
    obs = pd.read_parquet(C.CACHE / f"trips_obs_{C.DATE}.parquet")
    ps = pd.read_parquet(C.CACHE / f"passages_{C.DATE}.parquet")
    meta = obs.set_index("trip_id")[["route_id", "direction_id", "trip_headsign", "shape_id", "model", "veh", "n_stops"]]
    ps = ps.join(meta, on="trip_id").sort_values(["trip_id", "stop_sequence"])
    ps["delay_s"] = ps.dep_act - ps.dep_sched
    ps["peak"] = T.peak(ps.dep_sched)
    ps["hour"] = (ps.dep_sched // 3600).astype(int)

    # ---------------- delay profile along each line / direction (most common shape)
    main_shape = obs.groupby(["route_id", "direction_id"]).shape_id.agg(lambda s: s.value_counts().index[0])
    mp = ps[ps.shape_id.isin(main_shape.values)]
    prof = (mp.groupby(["route_id", "direction_id", "stop_sequence", "stop_name"])
              .agg(delay_med_s=("delay_s", "median"), delay_p85_s=("delay_s", lambda s: s.quantile(.85)), n=("delay_s", "count"))
              .reset_index())
    prof.round(0).to_csv(C.OUT / f"tt_delay_profile_{C.DATE}.csv", index=False)

    # ---------------- segments (consecutive stops of one trip)
    nx = ps.groupby("trip_id").shift(-1)
    seg = pd.DataFrame({"line": ps.route_id, "dir": ps.direction_id, "headsign": ps.trip_headsign,
                        "from": ps.stop_name, "to": nx.stop_name, "from_id": ps.stop_id, "to_id": nx.stop_id,
                        "sched_s": nx.dep_sched - ps.dep_sched, "act_s": nx.dep_act - ps.dep_act,
                        "last": nx.stop_sequence == ps.groupby("trip_id").stop_sequence.transform("max"),
                        "peak": ps.peak}).dropna(subset=["act_s", "sched_s"])
    first = ps.stop_sequence == ps.groupby("trip_id").stop_sequence.transform("min")
    seg = seg[~seg["last"] & ~first.reindex(seg.index) & (seg.act_s > 0) & (seg["from"] != seg["to"])]
    g = seg.groupby(["line", "dir", "from_id", "to_id"])
    sg = pd.DataFrame({"headsign": g.headsign.agg(lambda v: v.value_counts().index[0]), "from": g["from"].first(), "to": g["to"].first(),
                       "runs": g.size(), "sched_s": g.sched_s.median(), "act_med_s": g.act_s.median(),
                       "act_p85_s": g.act_s.quantile(.85),
                       "act_peak_med_s": seg[seg.peak].groupby(["line", "dir", "from_id", "to_id"]).act_s.median(),
                       "sched_peak_s": seg[seg.peak].groupby(["line", "dir", "from_id", "to_id"]).sched_s.median()})
    sg = sg[sg.runs >= 15].reset_index()
    sg["excess_s"] = sg.act_med_s - sg.sched_s
    sg["excess_peak_s"] = sg.act_peak_med_s - sg.sched_peak_s
    sg.round(0).sort_values("excess_peak_s", ascending=False).to_csv(C.OUT / f"tt_segments_{C.DATE}.csv", index=False)

    # ---------------- headway regularity at the middle stop of the main shape, 06-20
    rows = []
    for (line, d), sh in main_shape.items():
        q = ps[(ps.shape_id == sh)]
        if q.empty:
            continue
        mid = int(q.stop_sequence.median())
        q = ps[(ps.route_id == line) & (ps.direction_id == d) & (ps.stop_sequence == mid) &
               (ps.stop_id == q[q.stop_sequence == mid].stop_id.mode()[0])]
        q = q[(q.dep_sched >= 6 * 3600) & (q.dep_sched < 20 * 3600)]
        act = np.sort(q.dep_act.dropna().to_numpy()); sch = np.sort(q.dep_sched.to_numpy())
        if len(act) < 20:
            continue
        ha, hs = np.diff(act), np.diff(sch)
        hs_med = np.median(hs)
        rows.append({"line": line, "dir": d, "headsign": q.trip_headsign.mode()[0], "stop": q.stop_name.iloc[0],
                     "deps_observed": len(act), "deps_sched": len(sch), "sched_headway_min": hs_med / 60,
                     "awt_sched_min": awt(hs) / 60, "awt_act_min": awt(ha) / 60,
                     "bunched_pct": (ha < 0.5 * hs_med).mean() * 100, "gaps_pct": (ha > 1.5 * hs_med).mean() * 100,
                     "cv": ha.std() / ha.mean()})
    hw = pd.DataFrame(rows)
    hw["ewt_min"] = hw.awt_act_min - hw.awt_sched_min
    hw = hw.sort_values("ewt_min", ascending=False).round(2)
    hw.to_csv(C.OUT / f"tt_headways_{C.DATE}.csv", index=False)

    # ---------------- dwell by model (only intermediate stops, real stand-still)
    dw = ps[(ps.stop_sequence > ps.groupby("trip_id").stop_sequence.transform("min")) &
            (ps.stop_sequence < ps.groupby("trip_id").stop_sequence.transform("max")) & (ps.platform_stand_s > 0)].copy()
    dw = dw[dw.platform_stand_s < 180]
    dw["ref"] = dw.groupby(["stop_id", "hour"]).platform_stand_s.transform("median")
    dw = dw[dw.groupby(["stop_id", "hour"]).platform_stand_s.transform("size") >= 8]
    dw["ratio"] = dw.platform_stand_s / dw.ref
    dw["excess"] = dw.platform_stand_s - dw.ref
    mod = dw.groupby("model").agg(stop_visits=("ratio", "size"), dwell_mean_s=("platform_stand_s", "mean"),
                                  dwell_ratio=("ratio", "mean"), dwell_excess_s=("excess", "mean"))
    se = dw.groupby("model").excess.std() / np.sqrt(mod.stop_visits)
    mod["excess_ci95_s"] = 1.96 * se
    mod = mod.join(T.fleet_lookup(pd.Series(mod.index.map({t: str(lo) for lo, hi, t, *_ in C.FLEET}), index=mod.index))[["places", "length_m"]])
    mod = mod.sort_values("dwell_excess_s").round(2)
    mod.to_csv(C.OUT / f"tt_models_{C.DATE}.csv")

    print("Segments where the timetable is most optimistic in the peak (median actual - scheduled):")
    print(sg.sort_values("excess_peak_s", ascending=False)[["line", "headsign", "from", "to", "runs", "sched_peak_s", "act_peak_med_s", "excess_peak_s"]].head(20).to_string(index=False))
    print("\nMost padded segments (actual faster than scheduled):")
    print(sg.sort_values("excess_s")[["line", "headsign", "from", "to", "runs", "sched_s", "act_med_s", "excess_s"]].head(10).to_string(index=False))
    print("\nHeadway regularity (06-20, middle stop) — worst excess waiting time:")
    print(hw.head(15).to_string(index=False))
    print(f"\nmean EWT {hw.ewt_min.mean():.2f} min")
    print("\nDwell by model (ratio to stop median):\n" + mod.to_string())


if __name__ == "__main__":
    main()
