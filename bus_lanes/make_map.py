"""
Step D — Render the bus lane need map (PNG, 4:5 portrait for social media).

Outputs (bus_lanes/_output/):
    buspasy_warszawa.png     whole city
    buspasy_centrum.png      central Warsaw zoom
    buspasy_ranking.png      top streets without bus lanes
"""
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe

sys.path.insert(0, str(Path(__file__).parent))
from bl_config import DATA_DIR, NEED_HIGH, NEED_MED, OUTPUT_DIR

# --- style -----------------------------------------------------------------
BG = "#101218"
CITY = "#161a22"
STREET = "#252a35"
RIVER = "#1b2a3d"
INK = "#e8eaef"
INK2 = "#a3a9b6"
INK3 = "#6b7280"
COL = {"lane": "#3a82e0", "high": "#ff3d54", "medium": "#ffb020", "low": "#5a6170"}
ORDER = ["low", "lane", "medium", "high"]  # draw order (last on top)
FONT = "Segoe UI"

PL_MONTHS = ["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca",
             "sierpnia", "września", "października", "listopada", "grudnia"]


def date_pl(d: str) -> str:
    dt = datetime.strptime(d, "%Y%m%d")
    return f"środę {dt.day} {PL_MONTHS[dt.month - 1]} {dt.year}"


def width(per_h, scale):
    return scale * (0.3 + np.clip(per_h, 0, 70) / 16)


def offset_right(gdf: gpd.GeoDataFrame, dist_m: np.ndarray) -> np.ndarray:
    """Shift every way-direction to the right of travel (right-hand traffic) by its own
    half drawn width, so opposite directions sit side by side at any zoom level."""
    return shapely.offset_curve(gdf.geometry.values, -dist_m)


def draw_map(bus, streets, river, boundary, date, extent, out, title, scale, labels):
    fig = plt.figure(figsize=(10.8, 13.5), dpi=200, facecolor=BG)
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.72])
    ax.set_facecolor(BG)
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = extent
    # keep the axes aspect (0.94*10.8 : 0.72*13.5) so nothing is cropped oddly
    target = (0.94 * 10.8) / (0.72 * 13.5)
    cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    if w / h < target:
        w = h * target
    else:
        h = w / target
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)

    boundary.plot(ax=ax, color=CITY, edgecolor="#2c3240", linewidth=0.8, zorder=0)
    river.plot(ax=ax, color=RIVER, linewidth=0, zorder=1)
    streets.plot(ax=ax, color=STREET, linewidth=0.3, zorder=2)

    # bus loops / depot roads are bus-only but not street bus lanes — they only add blobs
    bus = bus[~((bus.bus_only == 1) & (bus.highway == "service"))].reset_index(drop=True)
    m_per_pt = w / (0.94 * 10.8 * 72)
    lw_all = width(bus.per_h.to_numpy(), scale) * np.where(bus.status == "low", 0.6, 1.0)
    geom = offset_right(bus, (lw_all / 2 + 0.3) * m_per_pt)
    for st in ORDER:
        m = (bus.status == st).to_numpy()
        if not m.any():
            continue
        # round caps join pieces smoothly; very short pieces (intersections) get flat caps,
        # otherwise their caps show up as beads along the street
        short_piece = shapely.length(geom) < 25
        for cap, sel in (("round", m & ~short_piece), ("butt", m & short_piece)):
            if sel.any():
                gpd.GeoSeries(geom[sel], crs=bus.crs).plot(
                    ax=ax, color=COL[st], linewidth=lw_all[sel], zorder=3 + ORDER.index(st),
                    capstyle=cap, joinstyle="round")

    halo = [pe.withStroke(linewidth=3, foreground=BG)]
    for name, (x, y) in labels.items():
        ax.text(x, y, name, fontsize=8.5 if scale > 1 else 7.5, color=INK, fontfamily=FONT,
                ha="center", va="center", zorder=10, path_effects=halo)

    # --- header ---
    fig.text(0.05, 0.955, title, fontsize=34, fontweight="bold", color=INK, fontfamily=FONT, va="top")
    fig.text(0.05, 0.895, "Liczba autobusów w godzinie szczytu, w jednym kierunku jazdy",
             fontsize=14, color=INK2, fontfamily=FONT, va="top")

    c = bus[bus.in_city & ~((bus.bus_only == 1) & (bus.highway == "service"))]
    km = c.geometry.length / 1000
    hi = c.per_h >= NEED_HIGH
    hi_km, hi_lane = km[hi].sum(), km[hi & (c.bus_lane == 1)].sum()
    fig.text(0.05, 0.862,
             f"Na {hi_km:.0f} km ulic jeździ ≥{NEED_HIGH} autobusów/h w jedną stronę. "
             f"Buspas ma tylko {hi_lane:.0f} km ({100 * hi_lane / hi_km:.0f}%).",
             fontsize=14, color=INK, fontfamily=FONT, va="top", fontweight="semibold")

    # --- legend ---
    items = [
        ("high", f"brak buspasu, ≥{NEED_HIGH} autobusów/h (co 2 min lub częściej)"),
        ("medium", f"brak buspasu, {NEED_MED}–{NEED_HIGH} autobusów/h"),
        ("lane", "jest buspas lub ulica tylko dla autobusów"),
        ("low", f"mniej niż {NEED_MED} autobusów/h"),
    ]
    handles = [Line2D([0], [0], color=COL[k], lw=5 if k != "low" else 3, solid_capstyle="round")
               for k, _ in items]
    leg = fig.legend(handles, [t for _, t in items], loc="upper left", bbox_to_anchor=(0.04, 0.845),
                     ncol=2, frameon=False, fontsize=11.5, labelcolor=INK2, handlelength=2.2,
                     columnspacing=1.5, prop={"family": FONT, "size": 11.5})
    fig.text(0.05, 0.047,
             f"Grubość linii = liczba autobusów/h (maks. ze szczytu 7–9 i 15–17). "
             f"Rozkład ZTM na {date_pl(date)}.",
             fontsize=10, color=INK3, fontfamily=FONT)
    fig.text(0.05, 0.027,
             "Dane: ZTM Warszawa (GTFS via mkuran.pl) · buspasy i ulice © współtwórcy OpenStreetMap (ODbL)",
             fontsize=10, color=INK3, fontfamily=FONT)
    fig.savefig(out, facecolor=BG, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def label_points(bus, names):
    """One label per street, at the middle of its longest 'high/medium' piece."""
    out = {}
    gaps = bus[bus.status.isin(["high", "medium"]) & bus.name.isin(names)]
    for name, g in gaps.groupby("name"):
        merged = shapely.line_merge(shapely.union_all(g.geometry.values))
        parts = list(getattr(merged, "geoms", [merged]))
        longest = max(parts, key=lambda p: p.length)
        p = longest.interpolate(0.5, normalized=True)
        out[short(name)] = (p.x, p.y + 180)
    return out


def short(name: str) -> str:
    rep = {"Aleje ": "Al. ", "Aleja ": "Al. ", "Generała ": "gen. ", "Tadeusza ": "", "Most ": "Most ",
           "Aleja Prymasa Tysiąclecia": "Al. Prymasa Tysiąclecia"}
    for a, b in rep.items():
        name = name.replace(a, b)
    return name


def draw_ranking(rank: pd.DataFrame, date: str, out: Path, n=15):
    r = rank[rank.len_km >= 1.0].sort_values("avg_per_h", ascending=False).head(n).iloc[::-1]
    fig = plt.figure(figsize=(10.8, 13.5), dpi=200, facecolor=BG)
    ax = fig.add_axes([0.40, 0.08, 0.52, 0.72])
    ax.set_facecolor(BG)
    y = np.arange(len(r))
    colors = [COL["high"] if v >= NEED_HIGH else COL["medium"] for v in r.avg_per_h]
    ax.barh(y, r.avg_per_h, height=0.62, color=colors, zorder=3)
    ax.set_yticks(y, [f"{short(u)} → {k}" for u, k in zip(r.ulica, r.kierunek)],
                  fontsize=12, color=INK, fontfamily=FONT)
    for yi, v, L in zip(y, r.avg_per_h, r.len_km):
        ax.text(v + 0.8, yi, f"{v:.0f}/h · {L:.1f} km", va="center", fontsize=11, color=INK2,
                fontfamily=FONT)
    ax.axvline(NEED_HIGH, color=INK3, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(NEED_HIGH, len(r) - 0.3, f" {NEED_HIGH}/h", color=INK3, fontsize=10, fontfamily=FONT)
    ax.set_xlim(0, r.avg_per_h.max() * 1.3)
    ax.tick_params(axis="x", colors=INK3, labelsize=10)
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="x", color="#232833", lw=0.8, zorder=0)
    ax.set_xlabel("średnio autobusów na godzinę w szczycie (w jednym kierunku)", color=INK2,
                  fontsize=11, fontfamily=FONT)
    fig.text(0.05, 0.955, "Najbardziej obciążone ulice\nbez buspasu", fontsize=32, fontweight="bold",
             color=INK, fontfamily=FONT, va="top", linespacing=1.1)
    fig.text(0.05, 0.855, "Odcinki bez buspasu o długości co najmniej 1 km, "
                          "średnia liczba autobusów/h na całym odcinku",
             fontsize=13, color=INK2, fontfamily=FONT, va="top")
    fig.text(0.05, 0.027, f"Rozkład ZTM na {date_pl(date)} · szczyt 7–9 i 15–17 · "
                          "buspasy © współtwórcy OpenStreetMap",
             fontsize=10, color=INK3, fontfamily=FONT)
    fig.savefig(out, facecolor=BG, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    date = (DATA_DIR / "analysis_date.txt").read_text(encoding="utf-8").split()[0]
    need = OUTPUT_DIR / "bus_lane_need.gpkg"
    bus = gpd.read_file(need, layer="bus_dir")
    boundary = gpd.read_file(need, layer="boundary")
    osm = DATA_DIR / "osm_streets.gpkg"
    streets = gpd.read_file(osm, layer="streets")
    streets = streets[streets.highway.isin([
        "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"])]
    river = gpd.read_file(osm, layer="river")
    rank = pd.read_csv(OUTPUT_DIR / "bus_lane_gaps_ranking.csv")

    city = boundary.total_bounds
    top = rank.drop_duplicates("ulica").head(12).ulica.tolist()
    draw_map(bus, streets, river, boundary, date, city, OUTPUT_DIR / "buspasy_warszawa.png",
             "Gdzie brakuje buspasów?", scale=1.3, labels=label_points(bus, top))

    # Central zoom: ~9 x 11 km around Śródmieście
    cx, cy = 638500, 486800   # EPSG:2180, near Rondo Dmowskiego
    ext = (cx - 5500, cy - 5500, cx + 5500, cy + 6500)
    in_ext = bus.cx[ext[0]:ext[2], ext[1]:ext[3]]
    top_c = (in_ext[in_ext.status == "high"].assign(L=lambda d: d.length)
             .groupby("name").L.sum().sort_values(ascending=False).head(12).index.tolist())
    lab = {k: v for k, v in label_points(in_ext, top_c).items()}
    draw_map(bus, streets, river, boundary, date, ext, OUTPUT_DIR / "buspasy_centrum.png",
             "Buspasy w centrum", scale=1.8, labels=lab)

    draw_ranking(rank, date, OUTPUT_DIR / "buspasy_ranking.png")


if __name__ == "__main__":
    main()
