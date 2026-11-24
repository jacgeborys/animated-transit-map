"""
Overlay route profiles of several lines (default: fastest and slowest tram from line_speeds_<date>.csv).

  row 1 — average speed along the route (300 m rolling mean of the 100 m bins)
  row 2 — cumulative minutes along the route (time–distance curve of an average run)
  columns — GTFS direction 0 / 1

Output: live/data/compare_<mode>_<lines>_<date>.png

Usage:
  python live/compare_lines.py                  # fastest vs slowest tram
  python live/compare_lines.py --lines 2 31 17  # any lines (max 4)
  python live/compare_lines.py --mode Bus --lines 733 106
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import route_profiles as rp
import stuck_points as sp

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]   # categorical slots 1, 2, 3, 7 — fixed order
SMOOTH_BINS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--mode", default="Tram", choices=["Tram", "Bus"])
    ap.add_argument("--lines", nargs="*")
    args = ap.parse_args()
    csv = sp.DATA_DIR / f"positions_{args.date}.csv" if args.date else sorted(sp.DATA_DIR.glob("positions_*.csv"))[-1]
    day = csv.stem.split("_")[1]

    if args.lines:
        lines = args.lines[:4]
    else:
        ls = pd.read_csv(sp.DATA_DIR / f"line_speeds_{day}.csv", dtype={"Lines": str})
        ls = ls[(ls.vehicle_type == args.mode) & (ls.veh_hours >= 5)].sort_values("avg_kmh", ascending=False)
        lines = [ls.Lines.iloc[0], ls.Lines.iloc[-1]]

    iv = pd.read_pickle(sp.CACHE_DIR / f"intervals_{day}.pkl")
    iv = iv[(iv.vehicle_type == args.mode) & iv.why.isin(rp.SERVICE) & iv.Lines.isin(lines)]
    span = f"{iv.t.min():%H:%M}–{iv.t.max():%H:%M}"
    shapes = rp.main_shapes(sp.latest_gtfs(), day, set(lines))

    profs = {}
    for route in lines:
        r = rp.assign_direction(iv[iv.Lines == route].sort_values(["VehicleNumber", "t"]), shapes, route)
        brk = (r.VehicleNumber.ne(r.VehicleNumber.shift()) | r.dir.ne(r.dir.shift())
               | (r.t - r.t.shift()).dt.total_seconds().gt(300))
        r = r.assign(run=brk.cumsum())
        for d, rd in r.groupby("dir"):
            meta = shapes[(route, d)]
            n_bins = int(np.ceil(meta["line"].length / rp.BIN_M))
            p = rp.profile(rp.spread(rd)).reindex(range(n_bins))
            p.loc[p.runs < rp.MIN_RUNS, ["kmh", "sec_per_run"]] = np.nan
            p["kmh_s"] = p.kmh.rolling(SMOOTH_BINS, center=True, min_periods=1).mean()
            sec = p.sec_per_run.fillna(p.sec_per_run.median())
            p["cum_min"] = sec.cumsum() / 60
            avg = p.d_m.sum() / p.t_s.sum() * 3.6
            profs[(route, d)] = (p, meta, avg)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), height_ratios=[1, 1.1], facecolor=rp.SURFACE)
    for col, d in enumerate(["0", "1"]):
        a1, a2 = axes[0, col], axes[1, col]
        for a in (a1, a2):
            a.set_facecolor(rp.SURFACE)
            for s in ["top", "right", "left"]:
                a.spines[s].set_visible(False)
            a.spines["bottom"].set_color(rp.GRID)
            a.tick_params(colors=rp.MUTED, labelsize=9, length=0)
            a.grid(color=rp.GRID, linewidth=1); a.set_axisbelow(True)
        heads = []
        for i, route in enumerate(lines):
            if (route, d) not in profs:
                continue
            p, meta, avg = profs[(route, d)]
            km = (p.index + 0.5) * rp.BIN_M / 1000
            c = COLORS[i]
            a1.plot(km, p.kmh_s, color=c, linewidth=2, label=f"{route} → {meta['headsign']} ({avg:.1f} km/h)")
            a2.plot(km, p.cum_min, color=c, linewidth=2)
            end = p.cum_min.dropna()
            a2.annotate(f"{route}: {end.iloc[-1]:.0f} min / {meta['line'].length/1000:.1f} km",
                        (km[end.index[-1]], end.iloc[-1]), xytext=(-6, 6), textcoords="offset points",
                        ha="right", fontsize=9.5, color=rp.INK)
            heads.append(f"{route} → {meta['headsign']}")
        a1.axhline(rp.REF_KMH, color=rp.INK2, alpha=0.5, linewidth=1)
        a1.set_ylim(0, 60); a1.set_yticks([0, 20, 40, 60])
        a1.set_ylabel("km/h (300 m average)", color=rp.MUTED, fontsize=9)
        a1.legend(frameon=False, fontsize=9.5, loc="lower center", ncols=1, labelcolor=rp.INK)
        a1.set_title(f"Direction {int(d) + 1}:  " + "  vs  ".join(heads), loc="left", color=rp.INK, fontsize=11.5)
        a2.set_ylabel("minutes since first stop", color=rp.MUTED, fontsize=9)
        a2.set_xlabel("km along the route", color=rp.MUTED, fontsize=9)
        a2.set_ylim(bottom=0); a2.set_xlim(left=0); a1.set_xlim(a2.get_xlim())
    fig.suptitle(f"Warsaw {args.mode.lower()}s — fastest vs slowest line, average of all runs · {day} {span}",
                 x=0.01, ha="left", color=rp.INK, fontsize=14)
    fig.text(0.01, 0.005, "Average speed incl. stop dwell and stuck time; terminus loops (first/last 150 m stand-still) "
             "excluded. Main GTFS shape per direction.", color=rp.MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = sp.DATA_DIR / f"compare_{args.mode.lower()}_{'_'.join(lines)}_{day}.png"
    fig.savefig(out, dpi=120, facecolor=rp.SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
