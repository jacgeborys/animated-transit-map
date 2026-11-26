"""
Step 6 — build the report page from the step 1-5 outputs.

Writes _output/tram_report_<date>.html (standalone: one file, no external data; fonts from Google Fonts).
Usage: python live/analysis/report.py
"""

import json

import numpy as np
import pandas as pd
import shapely

import ta_config as C
import ta_common as T

TEMPLATE = C.HERE / "report_template.html"
SCALE = 10          # SVG units = metres / SCALE


def r(v, n=1):
    return None if pd.isna(v) else round(float(v), n)


def main():
    trips, st, shp = T.load_gtfs()
    geoms = T.shape_lines(shp)
    obs = pd.read_parquet(C.CACHE / f"trips_obs_{C.DATE}.parquet")
    iv = pd.read_parquet(C.CACHE / f"intervals_{C.DATE}.parquet")
    places = pd.read_csv(C.OUT / f"places_{C.DATE}.csv")
    sig = pd.read_csv(C.OUT / f"signals_{C.DATE}.csv", dtype={"line": str, "dir": str})
    pl = pd.read_csv(C.OUT / f"priority_lines_{C.DATE}.csv", dtype={"line": str})
    pa = pd.read_csv(C.OUT / f"priority_approaches_{C.DATE}.csv")
    seg = pd.read_csv(C.OUT / f"tt_segments_{C.DATE}.csv", dtype={"line": str})
    hw = pd.read_csv(C.OUT / f"tt_headways_{C.DATE}.csv", dtype={"line": str})
    mod = pd.read_csv(C.OUT / f"tt_models_{C.DATE}.csv")

    # ---------------- map: main shape per line & direction, simplified
    main = obs.groupby(["route_id", "direction_id"]).shape_id.agg(lambda s: s.value_counts().index[0])
    xs, ys = [], []
    lines_svg = []
    for (route, d), sid in main.items():
        if d != "0":
            continue
        g = shapely.simplify(geoms[sid], 12)
        c = shapely.get_coordinates(g)
        xs += [c[:, 0].min(), c[:, 0].max()]; ys += [c[:, 1].min(), c[:, 1].max()]
        lines_svg.append({"line": route, "pts": c})
    x0, x1, y0, y1 = min(xs) - 400, max(xs) + 400, min(ys) - 400, max(ys) + 400
    to_svg = lambda x, y: (round((x - x0) / SCALE), round((y1 - y) / SCALE))
    net = [{"line": l["line"], "d": "M" + "L".join(f"{a},{b}" for a, b in (to_svg(x, y) for x, y in l["pts"]))} for l in lines_svg]

    top = places.sort_values("lost_s", ascending=False).head(160).copy()
    top["sx"], top["sy"] = zip(*[to_svg(x, y) for x, y in zip(top.x, top.y)])
    pmap = [{"name": p["name"], "stop": p["near_stop"], "hd": p["heading"], "to": p["towards"], "lines": p["lines"],
             "passes": int(p["passes"]), "sp": r(p["stop_prob"], 3), "spp": r(p["lost_per_pass_s"]),
             "h": r(p["lost_s"] / 3600, 2), "pax": r(p["pax_h_est"], 0), "x": int(p["sx"]), "y": int(p["sy"])}
            for _, p in top.iterrows()]

    # ---------------- time split (tram-hours in service)
    s = iv[~iv.anomaly]
    split = {
        "moving": s[~s.stand & (s.zone != "terminus")].dt.sum() / 3600,
        "platform": s[s.stand & (s.zone == "platform")].dt.sum() / 3600,
        "line": s[s.stand & (s.zone == "line")].dt.sum() / 3600,
    }
    term_h = s[s.zone == "terminus"].dt.sum() / 3600
    hour = (s.t0 // 3600).astype(int)
    hourly = pd.DataFrame({
        "service": s[s.zone != "terminus"].groupby(hour).dt.sum() / 3600,
        "line": s[s.stand & (s.zone == "line")].groupby(hour).dt.sum() / 3600}).fillna(0)
    hourly = hourly.loc[5:22]
    hourly_out = [{"h": int(h), "service": r(v.service), "line": r(v.line), "share": r(v.line / v.service * 100)}
                  for h, v in hourly.iterrows()]

    # ---------------- direction asymmetry: same junction, opposite headings
    p = places[places.passes >= 150].copy()
    pairs = []
    used = set()
    for i, a in p.sort_values("lost_s", ascending=False).iterrows():
        if i in used:
            continue
        opp = p[((p.hs - a.hs) % 8 == 4) & (np.hypot(p.x - a.x, p.y - a.y) < 120)]
        if opp.empty:
            continue
        b = opp.iloc[0]
        if b.name in used:
            continue
        used |= {i, b.name}
        pairs.append((a, b))
    asym = []
    for a, b in pairs:
        hi, lo = (a, b) if a.stop_prob >= b.stop_prob else (b, a)
        asym.append({"name": hi["name"], "bad_hd": hi.heading, "bad_to": hi.towards, "bad_sp": r(hi.stop_prob, 3),
                     "bad_spp": r(hi.lost_per_pass_s), "good_hd": lo.heading, "good_to": lo.towards,
                     "good_sp": r(lo.stop_prob, 3), "good_spp": r(lo.lost_per_pass_s), "gap": r(hi.stop_prob - lo.stop_prob, 3),
                     "lines": hi.lines})
    asym = sorted(asym, key=lambda d: -d["gap"])[:10]

    # ---------------- Wilanów section (line 16, both directions), m from Goworka
    wil = {}
    st16 = st.merge(obs[["trip_id", "shape_id"]], on="trip_id")
    for d in ["0", "1"]:
        sid = main[("16", d)]
        q = st16[st16.shape_id == sid].drop_duplicates("stop_sequence").sort_values("stop_sequence")
        g = q.set_index("stop_name").m
        a, b = g["Goworka"], g["Miasteczko Wilanów"]
        lo, hi = min(a, b), max(a, b)
        sg = sig[(sig.line == "16") & (sig.dir == d) & sig.m.between(lo - 60, hi + 160)].sort_values("m")
        stops_ = q[q.m.between(lo - 5, hi + 5)]
        flip = a > b      # direction 1 runs Wilanów -> Goworka: measure distance from Wilanów
        base = lo
        wil[d] = {
            "headsign": sg.headsign.mode()[0] if len(sg) else "",
            "len": r((hi - lo) / 1000, 2),
            "signals": [{"km": r((m - base) / 1000, 3), "name": n, "sp": r(spv, 3), "spp": r(lp), "plat": bool(pf), "n": int(ps)}
                        for m, n, spv, lp, pf, ps in zip(sg.m, sg.name.fillna(""), sg.stop_prob, sg.lost_per_pass_s, sg.platform_at_signal, sg.passes)],
            "stops": [{"km": r((m - base) / 1000, 3), "name": n} for m, n in zip(stops_.m, stops_.stop_name)],
            "from": "Miasteczko Wilanów" if flip else "Goworka", "to": "Goworka" if flip else "Miasteczko Wilanów",
        }
    city = sig[~sig.platform_at_signal]
    city_q = {k: r(v, 3) for k, v in city.stop_prob.quantile([.25, .5, .75, .9]).items()}

    # ---------------- priority per line
    pl_out = [{"line": x.line, "trams": int(x.peak_trams) if not pd.isna(x.peak_trams) else None, "hw": r(x.peak_headway_min),
               "rt_s": r(x.rt_sched_min), "rt_a": r(x.rt_act_min), "late": r(x.late_vs_timetable_min),
               "save": r(x.rt_save_peak_min, 2), "freed": r(x.trams_freed_peak, 2), "pax": r(x.pax_h_saved_day, 0),
               "sig_h": r(x.tram_h_at_signals_day)} for x in pl.itertuples()]
    pa["hd"] = pa["loc"].str.extract(r"_h(\d)$")[0].astype(int).map(dict(enumerate(["E", "NE", "N", "NW", "W", "SW", "S", "SE"])))
    pa_out = [{"name": x.name, "hd": x.hd, "lines": x.lines, "to": x.towards, "passes": int(x.passes), "sp": r(x.stop_prob, 3),
               "save_h": r(x.save_h_day, 2), "pax": r(x.pax_h_day, 0)} for x in pa.head(20).itertuples()]

    seg_out = [{"line": x.line, "to_hs": x.headsign, "a": x._6, "b": x.to, "sched": r(x.sched_peak_s, 0), "act": r(x.act_peak_med_s, 0),
                "ex": r(x.excess_peak_s, 0), "runs": int(x.runs)}
               for x in seg.dropna(subset=["excess_peak_s"]).sort_values("excess_peak_s", ascending=False).head(14).itertuples()]
    pad_out = [{"line": x.line, "to_hs": x.headsign, "a": x._6, "b": x.to, "sched": r(x.sched_s, 0), "act": r(x.act_med_s, 0),
                "ex": r(x.excess_s, 0)} for x in seg.sort_values("excess_s").head(8).itertuples()]
    hw_out = [{"line": x.line, "to": x.headsign, "stop": x.stop, "hw": r(x.sched_headway_min), "ewt": r(x.ewt_min, 2),
               "bunch": r(x.bunched_pct, 0), "gaps": r(x.gaps_pct, 0), "awt_s": r(x.awt_sched_min, 2), "awt_a": r(x.awt_act_min, 2)}
              for x in hw.itertuples()]
    fleet = obs.drop_duplicates("veh").model.value_counts()
    mod_out = [{"model": x.model, "veh": int(fleet.get(x.model, 0)), "visits": int(x.stop_visits), "excess": r(x.dwell_excess_s, 2),
                "ci": r(x.excess_ci95_s, 2), "places": int(x.places), "len": r(x.length_m)} for x in mod.itertuples()]
    planned = pd.crosstab(obs.fleet_type.fillna("–"), obs.model)
    mism = int(((obs.fleet_type == "2 wagony") & ~obs.model.str.startswith(("105", "123"))).sum())

    tot = pl.sum(numeric_only=True)
    data = {
        "date": f"{C.DATE[6:]}.{C.DATE[4:6]}.{C.DATE[:4]}", "weekday": "Tuesday",
        "span": f"{T.clock(obs.t_first.min())}–{T.clock(obs.t_last.max())}",
        "n_trams": int(obs.veh.nunique()), "n_trips_obs": int(len(obs)), "n_trips_sched": int(len(trips)),
        "n_fixes": int(len(pd.read_parquet(C.CACHE / f"fixes_{C.DATE}.parquet", columns=["t"]))),
        "split": {k: r(v) for k, v in split.items()}, "terminus_h": r(term_h),
        "hourly": hourly_out, "net": net, "vb": [0, 0, round((x1 - x0) / SCALE), round((y1 - y0) / SCALE)],
        "places": pmap, "asym": asym, "wil": wil, "city_q": city_q,
        "prio": pl_out, "prio_app": pa_out,
        "prio_tot": {"sig_h": r(tot.tram_h_at_signals_day, 0), "save_h": r(tot.tram_h_saved_day, 0), "pax": r(tot.pax_h_saved_day, 0),
                     "pln": r(tot.pln_day, 0), "freed": r(tot.trams_freed_peak, 1),
                     "whole": int(np.floor(pl.trams_freed_peak).sum())},
        "seg": seg_out, "pad": pad_out, "hw": hw_out, "ewt_mean": r(hw.ewt_min.mean(), 2),
        "models": mod_out, "mismatch_2w": mism, "n_2w": int((obs.fleet_type == "2 wagony").sum()),
        "assume": {"lf_peak": C.LOAD_FACTOR["peak"], "lf_off": C.LOAD_FACTOR["offpeak"], "target": C.TARGET_STOP_RATE,
                   "accel": C.ACCEL_PENALTY_S, "vot": C.VALUE_OF_TIME_PLN_H},
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/{}", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out = C.OUT / f"tram_report_{C.DATE}.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved {out} ({out.stat().st_size/1e3:.0f} kB)")
    print(json.dumps({k: data[k] for k in ["split", "terminus_h", "city_q", "prio_tot", "ewt_mean", "mismatch_2w", "n_2w"]}, ensure_ascii=False))
    print(json.dumps(asym[:6], ensure_ascii=False, indent=0)[:2500])


if __name__ == "__main__":
    main()
