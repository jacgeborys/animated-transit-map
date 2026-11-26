"""
Step 4 — what would working tram priority buy?  (scenario on top of measured data)

Measured (signals.py): for every run past every signalised approach, seconds standing + whether it stopped.
Scenario: each approach is brought to TARGET_STOP_RATE (the level of the good signals on the Wilanów line).
  saving at an approach = (lost standing + ACCEL_PENALTY_S x stops) x max(0, 1 - TARGET / stop_rate)
  Approaches with a platform inside (red hidden in dwell) are left out -> conservative.

Per line:
  * minutes saved per run and per round trip, in the peak
  * trams freed in the peak  = round-trip saving / peak headway  (a brigade can be cut once the cycle
    shrinks by one headway; fractional = how far along the line is to that)
  * passenger-hours saved per day = saving x nominal places x LOAD_FACTOR (assumption, ta_config)
  * actual vs scheduled peak run time: if trams already run longer than the timetable, the first minutes
    saved only restore punctuality.

Output (_output/): priority_lines_<date>.csv, priority_approaches_<date>.csv
Usage: python live/analysis/priority_scenarios.py
"""

import numpy as np
import pandas as pd

import ta_config as C
import ta_common as T


def peak_service(trips: pd.DataFrame) -> pd.DataFrame:
    """Peak (07-09) trams in service and headway per line from the timetable."""
    t = trips[~trips.trip_headsign.str.contains("zajezdni", case=False, na=False)]
    a, b = 7 * 3600, 9 * 3600
    act = t[(t.start_s < 8 * 3600 + 600) & (t.end_s > 8 * 3600 - 600)]
    veh = act.groupby("route_id").block_short_name.nunique().rename("peak_trams")
    dep = t[(t.start_s >= a) & (t.start_s < b)].groupby(["route_id", "direction_id"]).size()
    tph = (dep.groupby("route_id").mean() / 2).rename("peak_tph")
    return pd.concat([veh, tph], axis=1).assign(peak_headway_min=lambda d: 60 / d.peak_tph)


def run_times(obs: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    """Peak median actual vs scheduled run time, 2nd stop -> 2nd-last stop, full runs only."""
    p = ps.dropna(subset=["dep_act"]).sort_values(["trip_id", "stop_sequence"])
    p = p[p.groupby("trip_id").stop_sequence.rank(method="first") > 1]           # skip 1st stop (layover)
    f = p.groupby("trip_id").agg(s0=("dep_sched", "first"), s1=("dep_sched", "last"), a0=("dep_act", "first"),
                                 a1=("dep_act", "last"), n=("stop_sequence", "size"))
    f = f.join(obs.set_index("trip_id")[["route_id", "direction_id", "n_stops"]])
    f = f[(f.n >= f.n_stops - 3) & T.peak(f.s0)]
    f["sched_min"], f["act_min"] = (f.s1 - f.s0) / 60, (f.a1 - f.a0) / 60
    g = f.groupby(["route_id", "direction_id"])[["sched_min", "act_min"]].median()
    return g.groupby("route_id").sum().rename(columns={"sched_min": "rt_sched_min", "act_min": "rt_act_min"})


def main():
    trips, st, shp = T.load_gtfs()
    obs = pd.read_parquet(C.CACHE / f"trips_obs_{C.DATE}.parquet")
    ps = pd.read_parquet(C.CACHE / f"passages_{C.DATE}.parquet")
    d = pd.read_parquet(C.CACHE / f"signal_passes_{C.DATE}.parquet")
    d = d[~d.platform_at_signal]

    a = d.groupby(["line", "dir", "loc"]).agg(stop_prob=("stopped", "mean"), passes=("stopped", "size"))
    a["keep"] = np.clip(1 - C.TARGET_STOP_RATE / a.stop_prob.replace(0, np.nan), 0, 1).fillna(0)
    d = d.join(a.keep, on=["line", "dir", "loc"])
    d["save_s"] = (d.lost_line_s + C.ACCEL_PENALTY_S * d.stopped) * d.keep
    lf = np.where(d.peak, C.LOAD_FACTOR["peak"], C.LOAD_FACTOR["offpeak"])
    d["pax_h"] = d.save_s * d.places.fillna(200) * lf / 3600

    per_run = d.groupby(["line", "dir", "trip_id", "peak"]).save_s.sum().reset_index()
    pk = per_run[per_run.peak].groupby(["line", "dir"]).save_s.mean().groupby("line").sum().rename("rt_save_peak_s")
    allday = per_run.groupby(["line", "dir"]).save_s.mean().groupby("line").sum().rename("rt_save_allday_s")
    lines = pd.concat([pk, allday,
                       d.groupby("line").save_s.sum().div(3600).rename("tram_h_saved_day"),
                       d.groupby("line").pax_h.sum().rename("pax_h_saved_day"),
                       d.groupby("line").lost_line_s.sum().div(3600).rename("tram_h_at_signals_day")], axis=1)
    lines = lines.join(peak_service(trips)).join(run_times(obs, ps))
    lines["rt_save_peak_min"] = lines.rt_save_peak_s / 60
    lines["trams_freed_peak"] = lines.rt_save_peak_min / lines.peak_headway_min
    lines["late_vs_timetable_min"] = lines.rt_act_min - lines.rt_sched_min
    lines["pln_day"] = lines.pax_h_saved_day * C.VALUE_OF_TIME_PLN_H
    lines = lines.sort_values("trams_freed_peak", ascending=False).round(2)
    lines.index.name = "line"

    worst = (d.groupby(["line", "dir", "loc"]).agg(save_h_day=("save_s", lambda s: s.sum() / 3600), pax_h_day=("pax_h", "sum"))
             .join(a[["stop_prob", "passes"]]).reset_index())
    sig = pd.read_csv(C.OUT / f"signals_{C.DATE}.csv", dtype={"line": str, "dir": str})
    worst["dir"] = worst.dir.astype(str)
    worst = worst.merge(sig[["line", "dir", "loc", "name", "headsign", "lost_per_pass_s", "x", "y"]], on=["line", "dir", "loc"])
    # one physical approach is used by many lines -> pool by location and heading-ish (loc + headsign group)
    phys = (worst.groupby(["loc", "name"], dropna=False).agg(save_h_day=("save_h_day", "sum"), pax_h_day=("pax_h_day", "sum"),
                                                passes=("passes", "sum"),
                                                stop_prob=("stop_prob", lambda s: np.average(s, weights=worst.loc[s.index, "passes"])),
                                                lines=("line", lambda s: ",".join(sorted(set(s), key=lambda v: (len(v), v)))),
                                                towards=("headsign", lambda s: " / ".join(sorted(set(s))[:3])),
                                                x=("x", "median"), y=("y", "median"))
            .reset_index().sort_values("pax_h_day", ascending=False).round(3))
    lines.to_csv(C.OUT / f"priority_lines_{C.DATE}.csv")
    phys.to_csv(C.OUT / f"priority_approaches_{C.DATE}.csv", index=False)

    tot = lines.sum(numeric_only=True)
    print(f"Scenario: every signal approach (no platform) brought to {C.TARGET_STOP_RATE:.0%} stop rate, "
          f"+{C.ACCEL_PENALTY_S}s per avoided stop, load factor {C.LOAD_FACTOR}")
    print(f"  tram-hours standing at those signals: {tot.tram_h_at_signals_day:.0f} h/day")
    print(f"  tram-hours saved: {tot.tram_h_saved_day:.0f} h/day · passenger-hours saved: {tot.pax_h_saved_day:,.0f}/day "
          f"(~{tot.pln_day:,.0f} PLN/day at {C.VALUE_OF_TIME_PLN_H} PLN/h)")
    print(f"  trams freed in the peak (sum of fractions): {tot.trams_freed_peak:.1f}  ·  "
          f"whole brigades (floor per line): {np.floor(lines.trams_freed_peak).sum():.0f}")
    cols = ["peak_trams", "peak_headway_min", "rt_sched_min", "rt_act_min", "late_vs_timetable_min",
            "rt_save_peak_min", "trams_freed_peak", "pax_h_saved_day"]
    print("\n" + lines[cols].to_string())
    print("\nTop 20 signal approaches by passenger-hours to gain:\n" +
          phys[["name", "lines", "towards", "passes", "stop_prob", "save_h_day", "pax_h_day"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
