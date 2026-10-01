"""
Simple bus-lane maps for colleagues to verify (light, print-friendly): streets + current bus lanes.

Each direction is drawn on its own side of the street (right of travel), so a one-way lane shows
as a single line on one side. Streets that have lanes are labelled so reviewers can name them.

Outputs (bus_lanes/_output/): buspasy_weryfikacja_warszawa.png, buspasy_weryfikacja_centrum.png
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import make_map as mm
from bl_config import DATA_DIR, OUTPUT_DIR

BG = "#f6f6f3"
CITY = "#ffffff"
STREET = "#cfd3cc"
RIVER = "#bcd5e8"
LANE = "#1f5fd1"
INK = "#1b1f23"
INK2 = "#4f575e"
INK3 = "#7c858c"
FONT = mm.FONT


def draw(lanes, streets, river, boundary, extent, out, title, scale, n_labels):
    fig = plt.figure(figsize=(10.8, 13.5), dpi=200, facecolor=BG)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.76])
    ax.set_facecolor(BG)
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = extent
    target = (0.94 * 10.8) / (0.76 * 13.5)
    cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    if w / h < target:
        w = h * target
    else:
        h = w / target
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)
    m_per_pt = w / (0.94 * 10.8 * 72)

    boundary.plot(ax=ax, color=CITY, edgecolor="#b9beb6", linewidth=0.8, zorder=0)
    river.plot(ax=ax, color=RIVER, linewidth=0, zorder=1)
    streets.plot(ax=ax, color=STREET, linewidth=0.45 * scale, zorder=2)

    lw = 1.7 * scale
    geom = shapely.offset_curve(lanes.geometry.values, -(lw / 2 + 0.4) * m_per_pt)
    short = shapely.length(geom) < 25
    for cap, sel in (("round", ~short), ("butt", short)):
        gpd.GeoSeries(geom[sel], crs=lanes.crs).plot(ax=ax, color=LANE, linewidth=lw, zorder=5,
                                                      capstyle=cap, joinstyle="round")

    # label the streets with the most lane length in view (light theme for the label renderer)
    view = shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    vis = lanes[lanes.intersects(view) & lanes.name.notna()]
    names = vis.assign(L=vis.length).groupby("name").L.sum().sort_values(ascending=False).head(n_labels).index.tolist()
    named = gpd.read_file(DATA_DIR / "osm_streets.gpkg", layer="streets", columns=["name", "highway"])
    named = named[named.name.notna() & ~named.highway.isin(["service", "living_street"])
                  & ~named.highway.str.endswith("_link")]
    labels = mm.label_points(lanes.assign(status="lane"), named, names)
    mm.INK, mm.BG = INK, BG
    fs = 10.5 if scale > 1.5 else 9.5
    gap_m = (lw / 2 + fs * 0.75) * m_per_pt
    frame = view.buffer(-4 * fs * m_per_pt)
    placed = []
    for name in labels:
        ways, tgt, side = labels[name]
        mm.curved_label(ax, name, ways, tgt, fs, m_per_pt, gap_m * side, frame, placed)

    fig.text(0.05, 0.965, title, fontsize=30, fontweight="bold", color=INK, fontfamily=FONT, va="top")
    fig.text(0.05, 0.915, "Czy coś się nie zgadza? Napisz: ulica, odcinek (od – do), kierunek jazdy.",
             fontsize=14, color=INK, fontfamily=FONT, va="top", fontweight="semibold")
    fig.text(0.05, 0.888, "Np. „ul. X, od ul. Y do ul. Z, w stronę centrum – nie ma tu buspasu”"
                          " albo „brakuje buspasu na …”.",
             fontsize=12, color=INK2, fontfamily=FONT, va="top")
    handles = [Line2D([0], [0], color=LANE, lw=4, solid_capstyle="round")]
    fig.legend(handles, ["buspas (lub ulica tylko dla autobusów)"], loc="upper left",
               bbox_to_anchor=(0.04, 0.862), frameon=False, labelcolor=INK2,
               prop={"family": FONT, "size": 12}, handlelength=2.2)
    fig.text(0.47, 0.851, "Każdy kierunek jest rysowany po prawej stronie jezdni (ruch prawostronny):\n"
                          "linia tylko z jednej strony ulicy = buspas tylko w jednym kierunku.",
             fontsize=11, color=INK2, fontfamily=FONT, va="center")
    fig.text(0.05, 0.03, "Buspasy: OpenStreetMap, mapa.um.warszawa.pl (stan 2021), komunikaty ZDM, zgłoszenia ·"
                         " ulice © współtwórcy OpenStreetMap (ODbL)", fontsize=10, color=INK3, fontfamily=FONT)
    fig.savefig(out, facecolor=BG, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    bus = gpd.read_file(OUTPUT_DIR / "bus_lane_need.gpkg", layer="bus_dir")
    lanes = bus[(bus.bus_lane == 1) & ~((bus.bus_only == 1) & (bus.highway == "service"))].reset_index(drop=True)
    boundary = gpd.read_file(OUTPUT_DIR / "bus_lane_need.gpkg", layer="boundary")
    osm = DATA_DIR / "osm_streets.gpkg"
    streets = gpd.read_file(osm, layer="streets")
    streets = streets[streets.highway.isin([
        "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"])]
    river = gpd.read_file(osm, layer="river")

    draw(lanes, streets, river, boundary, boundary.total_bounds, OUTPUT_DIR / "buspasy_weryfikacja_warszawa.png",
         "Buspasy w Warszawie – sprawdźmy!", scale=1.0, n_labels=30)
    cx, cy = 638500, 486800
    draw(lanes, streets, river, boundary, (cx - 5500, cy - 5500, cx + 5500, cy + 6500),
         OUTPUT_DIR / "buspasy_weryfikacja_centrum.png", "Buspasy w centrum – sprawdźmy!", scale=1.8, n_labels=40)


if __name__ == "__main__":
    main()
