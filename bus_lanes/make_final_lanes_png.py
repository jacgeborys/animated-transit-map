"""
Final bus-lane map (light): bus lanes counted in the analysis (OSM, verified by the community, Oct 2026)
in blue; short pieces dropped from the analysis (< MIN_LANE_STRETCH_M: bus-stop bays, terminus bits) in red.

Output: bus_lanes/_output/buspasy_mapa.png
"""
import json
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import make_map as mm
from bl_config import DATA_DIR, MIN_LANE_STRETCH_M, OUTPUT_DIR

BG, CITY, STREET, RIVER = "#f6f6f3", "#ffffff", "#cfd3cc", "#bcd5e8"
LANE, REMOVED = "#1f5fd1", "#e03131"
INK, INK2, INK3 = "#1b1f23", "#4f575e", "#7c858c"
FONT = mm.FONT
PL_MONTHS = ["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września",
             "października", "listopada", "grudnia"]


def main():
    bus = gpd.read_file(OUTPUT_DIR / "bus_lane_need.gpkg", layer="bus_dir")
    boundary = gpd.read_file(OUTPUT_DIR / "bus_lane_need.gpkg", layer="boundary")
    bus = bus[bus.in_city & (bus.loop == 0)].reset_index(drop=True)
    lanes = bus[bus.bus_lane == 1].reset_index(drop=True)
    removed = bus[bus.lane_removed == 1].reset_index(drop=True)
    lane_km, removed_km = lanes.length.sum() / 1000, removed.length.sum() / 1000

    osm = DATA_DIR / "osm_streets.gpkg"
    streets = gpd.read_file(osm, layer="streets")
    streets = streets[streets.highway.isin([
        "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"])]
    river = gpd.read_file(osm, layer="river")
    d = date.fromisoformat(json.loads((DATA_DIR / "osm_raw.json").read_text(encoding="utf-8"))
                           ["osm3s"]["timestamp_osm_base"][:10])

    fig = plt.figure(figsize=(10.8, 13.5), dpi=200, facecolor=BG)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.76])
    ax.set_facecolor(BG)
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = boundary.total_bounds
    target = (0.94 * 10.8) / (0.76 * 13.5)
    cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    w, h = (h * target, h) if w / h < target else (w, w / target)
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)
    m_per_pt = w / (0.94 * 10.8 * 72)

    boundary.plot(ax=ax, color=CITY, edgecolor="#b9beb6", linewidth=0.8, zorder=0)
    river.plot(ax=ax, color=RIVER, linewidth=0, zorder=1)
    streets.plot(ax=ax, color=STREET, linewidth=0.45, zorder=2)
    lw = 1.9
    for gdf, color, z in ((lanes, LANE, 5), (removed, REMOVED, 6)):   # removed on top so they are seen
        geom = shapely.offset_curve(gdf.geometry.values, -(lw / 2 + 0.4) * m_per_pt)
        gpd.GeoSeries(geom, crs=gdf.crs).plot(ax=ax, color=color, linewidth=lw, zorder=z,
                                              capstyle="round", joinstyle="round")

    # label the streets with the most lane length
    names = (lanes[lanes.name.notna()].assign(L=lambda g: g.length).groupby("name").L.sum()
             .sort_values(ascending=False).head(28).index.tolist())
    named = streets[streets.name.notna() & ~streets.highway.str.endswith("_link")]
    labels = mm.label_points(lanes.assign(status="lane"), named, names)
    mm.INK, mm.BG = INK, BG
    fs = 9.5
    gap_m = (lw / 2 + fs * 0.75) * m_per_pt
    frame = shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2).buffer(-4 * fs * m_per_pt)
    placed = []
    for name in labels:
        ways, tgt, side = labels[name]
        mm.curved_label(ax, name, ways, tgt, fs, m_per_pt, gap_m * side, frame, placed)

    fig.text(0.05, 0.965, "Buspasy w Warszawie", fontsize=30, fontweight="bold", color=INK, fontfamily=FONT, va="top")
    fig.text(0.05, 0.917, f"{lane_km:.0f} km buspasów na ulicach, którymi jeżdżą autobusy ZTM "
                          f"(każdy kierunek osobno). Stan na {d.day} {PL_MONTHS[d.month - 1]} {d.year}.",
             fontsize=14, color=INK, fontfamily=FONT, va="top", fontweight="semibold")
    handles = [Line2D([0], [0], color=c, lw=4, solid_capstyle="round") for c in (LANE, REMOVED)]
    fig.legend(handles, [f"buspas lub jezdnia tylko dla autobusów ({lane_km:.0f} km)",
                         f"pominięte: odcinki krótsze niż {MIN_LANE_STRETCH_M} m — zatoki przystankowe, "
                         f"końcówki przy pętlach ({removed_km:.1f} km)"],
               loc="upper left", bbox_to_anchor=(0.04, 0.89), frameon=False, labelcolor=INK2,
               prop={"family": FONT, "size": 12}, handlelength=2.2)
    fig.text(0.05, 0.03, "Buspasy: © współtwórcy OpenStreetMap (ODbL), zweryfikowane przez społeczność w październiku "
                         "2026 · każdy kierunek rysowany po prawej stronie jezdni", fontsize=9.5, color=INK3, fontfamily=FONT)
    out = OUTPUT_DIR / "buspasy_mapa.png"
    fig.savefig(out, facecolor=BG, dpi=200)
    plt.close(fig)
    print(f"Saved {out}  (lanes {lane_km:.1f} km, removed {removed_km:.1f} km)")


if __name__ == "__main__":
    main()
