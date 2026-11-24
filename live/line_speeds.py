"""
Average speed of every ZTM line from live GPS, plus fastest / slowest ranking.

Speed = distance / time in service  (commercial speed: includes dwell at stops and
time stuck in traffic; excludes terminus areas, layovers > 10 min and off-route fixes —
same classification as stuck_points.py).

Output (live/data/):
  line_speeds_<date>.csv            — one row per line: avg_kmh, moving_kmh, shares, veh_hours, per-hour speeds
  line_speed_ranking_<date>.png     — all tram lines + 15 fastest / 15 slowest bus lines
  line_charts_<date>/<line>.png     — per line: speed by hour vs. mode average, time split per hour

Usage:
  python live/line_speeds.py                  # uses cache from stuck_points.py if present
  python live/line_speeds.py --fresh          # rebuild intervals from positions CSV (~3 min)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stuck_points as sp

MIN_VEH_HOURS = 5      # lines with less in-service observation are left out of the ranking
MIN_HOUR_VEH_H = 0.3   # per-line chart: skip hours with less data than this
N_BUS = 15

# chart styling (light surface; categorical slots from the dataviz reference palette)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1a1a19", "#6b6a63", "#e6e5e0"
MODE_COLOR = {"Bus": "#2a78d6", "Tram": "#eb6834"}
SPLIT = [("moving", "Moving", "#d3d1c7"), ("at_stop", "At stop", "#1baf7a"), ("stuck", "Stuck", "#eb6834")]
SERVICE = ["moving", "at_stop", "stuck"]
PLURAL = {"Bus": "buses", "Tram": "trams"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x" if ax.name == "rectilinear" else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def line_table(iv: pd.DataFrame) -> pd.DataFrame:
    s = iv[iv.why.isin(SERVICE)].copy()
    s["hour"] = s.t.dt.hour
    g = s.groupby(["vehicle_type", "Lines"])
    tab = pd.DataFrame({
        "avg_kmh": g.dist.sum() / g.dt.sum() * 3.6,
        "moving_kmh": s[s.why == "moving"].groupby(["vehicle_type", "Lines"]).apply(
            lambda d: d.dist.sum() / d.dt.sum() * 3.6, include_groups=False),
        "veh_hours": g.dt.sum() / 3600,
        "vehicles": g.VehicleNumber.nunique(),
    })
    share = s.pivot_table(index=["vehicle_type", "Lines"], columns="why", values="dt", aggfunc="sum").fillna(0)
    for w in SERVICE:
        tab[f"{w}_pct"] = (share.get(w, 0) / share.sum(axis=1) * 100)
    hourly = s.groupby(["vehicle_type", "Lines", "hour"]).agg(dist=("dist", "sum"), dt=("dt", "sum"))
    hk = (hourly.dist / hourly.dt * 3.6).where(hourly.dt >= MIN_HOUR_VEH_H * 3600).unstack("hour")
    hk.columns = [f"kmh_{h:02d}" for h in hk.columns]
    return tab.join(hk).round(1).reset_index().sort_values(["vehicle_type", "avg_kmh"], ascending=[True, False])


def ranking_chart(tab: pd.DataFrame, out, title: str):
    ok = tab[tab.veh_hours >= MIN_VEH_HOURS]
    trams = ok[ok.vehicle_type == "Tram"].sort_values("avg_kmh")
    buses = ok[ok.vehicle_type == "Bus"].sort_values("avg_kmh")
    b_slow, b_fast = buses.head(N_BUS), buses.tail(N_BUS)

    panels = [(f"Trams — all {len(trams)} lines", trams, "Tram"),
              (f"Buses — {N_BUS} fastest", b_fast, "Bus"),
              (f"Buses — {N_BUS} slowest", b_slow, "Bus")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), facecolor=SURFACE)
    xmax = ok.avg_kmh.max() * 1.15
    for ax, (name, d, mode) in zip(axes, panels):
        style(ax)
        y = np.arange(len(d))
        ax.barh(y, d.avg_kmh, height=0.72, color=MODE_COLOR[mode], edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(y, d.Lines, color=INK, fontsize=10)
        for yi, v in zip(y, d.avg_kmh):
            ax.text(v + xmax * 0.01, yi, f"{v:.1f}", va="center", fontsize=8.5, color=INK2)
        avg = (ok[ok.vehicle_type == mode].dist_sum.sum() / ok[ok.vehicle_type == mode].dt_sum.sum() * 3.6)
        ax.axvline(avg, color=INK2, linestyle=(0, (3, 3)), linewidth=1)
        ax.text(avg, len(d) - 0.2, f" all {PLURAL[mode]} {avg:.1f}", color=INK2, fontsize=8.5, va="bottom")
        ax.set_xlim(0, xmax)
        ax.set_title(name, loc="left", color=INK, fontsize=12, pad=12)
        ax.set_xlabel("average speed incl. stops, km/h", color=INK2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=14)
    fig.text(0.01, 0.005, f"Lines with ≥ {MIN_VEH_HOURS} vehicle-hours observed. Excludes terminus areas (150 m), "
             "layovers > 10 min and off-route positions.", color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def line_chart(iv_line: pd.DataFrame, mode_hourly: pd.Series, row, out):
    s = iv_line.copy()
    s["hour"] = s.t.dt.hour
    h = s.groupby("hour").agg(dist=("dist", "sum"), dt=("dt", "sum"))
    h = h[h.dt >= MIN_HOUR_VEH_H * 3600]
    if h.empty:
        return False
    kmh = h.dist / h.dt * 3.6
    split = s.pivot_table(index="hour", columns="why", values="dt", aggfunc="sum").reindex(h.index).fillna(0)
    split = split.div(split.sum(axis=1), axis=0) * 100

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), height_ratios=[1.3, 1], sharex=True, facecolor=SURFACE)
    for a in (a1, a2):
        style(a); a.grid(axis="y", color=GRID, linewidth=0.8); a.grid(axis="x", visible=False)
    mh = mode_hourly.reindex(h.index)
    a1.plot(mh.index, mh.values, color=INK2, linestyle=(0, (3, 3)), linewidth=1.5, label=f"All {PLURAL[row.vehicle_type]}")
    a1.plot(kmh.index, kmh.values, color=INK, linewidth=2, marker="o", markersize=5,
            label=f"Line {row.Lines}")
    a1.set_ylim(0, max(kmh.max(), mh.max()) * 1.2)
    a1.set_ylabel("km/h", color=INK2, fontsize=9)
    a1.legend(frameon=False, fontsize=9, loc="lower left", ncols=2, labelcolor=INK)
    a1.set_title(f"{row.vehicle_type} {row.Lines} — average {row.avg_kmh:.1f} km/h incl. stops "
                 f"({row.moving_kmh:.1f} km/h while moving)", loc="left", color=INK, fontsize=12)

    bottom = np.zeros(len(split))
    for key, label, col in SPLIT:
        vals = split.get(key, pd.Series(0, index=split.index)).to_numpy()
        a2.bar(split.index, vals, bottom=bottom, color=col, width=0.75, edgecolor=SURFACE, linewidth=1.5, label=label)
        bottom += vals
    a2.set_ylim(0, 100); a2.set_ylabel("% of time", color=INK2, fontsize=9)
    a2.set_xticks(h.index, [f"{x}:00" for x in h.index], fontsize=8)
    a2.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.12), ncols=3, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYYMMDD (default: latest positions_*.csv)")
    ap.add_argument("--fresh", action="store_true", help="rebuild intervals instead of using the cache")
    a = ap.parse_args()
    csv = sp.DATA_DIR / f"positions_{a.date}.csv" if a.date else sorted(sp.DATA_DIR.glob("positions_*.csv"))[-1]
    day = csv.stem.split("_")[1]
    cache = sp.CACHE_DIR / f"intervals_{day}.pkl"
    if cache.exists() and not a.fresh:
        iv = pd.read_pickle(cache)
        print(f"Using cached intervals ({cache.name}) — rerun stuck_points.py or pass --fresh for newer data")
    else:
        iv, _ = sp.build(day)
    span = f"{iv.t.min():%H:%M}–{iv.t.max():%H:%M}"
    print(f"Intervals: {len(iv):,}  ({span})")

    tab = line_table(iv)
    s = iv[iv.why.isin(SERVICE)]
    sums = s.groupby(["vehicle_type", "Lines"]).agg(dist_sum=("dist", "sum"), dt_sum=("dt", "sum")).reset_index()
    tab = tab.merge(sums, on=["vehicle_type", "Lines"])
    csv_out = sp.DATA_DIR / f"line_speeds_{day}.csv"
    tab.drop(columns=["dist_sum", "dt_sum"]).to_csv(csv_out, index=False)

    png = sp.DATA_DIR / f"line_speed_ranking_{day}.png"
    ranking_chart(tab, png, f"Warsaw ZTM — average line speed from live GPS · {day} {span}")

    out_dir = sp.DATA_DIR / f"line_charts_{day}"
    out_dir.mkdir(exist_ok=True)
    s = s.assign(hour=s.t.dt.hour)
    mode_h = s.groupby(["vehicle_type", "hour"]).apply(lambda d: d.dist.sum() / d.dt.sum() * 3.6, include_groups=False)
    n = 0
    for row in tab.itertuples():
        d = s[(s.vehicle_type == row.vehicle_type) & (s.Lines == row.Lines)]
        n += line_chart(d, mode_h.loc[row.vehicle_type], row, out_dir / f"{row.vehicle_type.lower()}_{row.Lines}.png")

    ok = tab[tab.veh_hours >= MIN_VEH_HOURS]
    cols = ["Lines", "avg_kmh", "moving_kmh", "at_stop_pct", "stuck_pct", "veh_hours"]
    for mode in ["Tram", "Bus"]:
        m = ok[ok.vehicle_type == mode].sort_values("avg_kmh", ascending=False)
        print(f"\n{mode}s — fastest:\n{m[cols].head(8).to_string(index=False)}")
        print(f"{PLURAL[mode].title()} — slowest:\n{m[cols].tail(8).to_string(index=False)}")
    print(f"\nSaved {csv_out}\nSaved {png}\nSaved {n} line charts to {out_dir}")


if __name__ == "__main__":
    main()
