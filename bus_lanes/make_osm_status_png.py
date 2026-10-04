"""
"State of bus lanes in OpenStreetMap" — one light PNG for a quick community sanity check.

Blue   = bus lane mapped in OSM (on a bus route; separate parallel bus roadways included, loops excluded)
Orange = lane per the city layer / ZDM announcements, but not in OSM (= what the OSM helper lists as "do dodania")

Output: bus_lanes/_output/buspasy_osm_stan.png
"""
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
import make_map as mm
import make_review_png as rp
from bl_config import DATA_DIR, OUTPUT_DIR

OSM = "#1f5fd1"
TODO = "#e8590c"
BASELINE_KM = 50.7   # OSM bus-lane km on bus routes on 1 Oct 2026, before the community edits
PL_MONTHS = ["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września",
             "października", "listopada", "grudnia"]


def main():
    bus = gpd.read_file(OUTPUT_DIR / "bus_lane_need.gpkg", layer="bus_dir")
    boundary = gpd.read_file(OUTPUT_DIR / "bus_lane_need.gpkg", layer="boundary")
    bus = bus[bus.in_city & (bus.loop == 0)].reset_index(drop=True)
    osm = bus[bus.bl_osm == 1]
    todo = bus[((bus.bl_city == 1) | (bus.bl_manual == 1)) & (bus.bl_osm == 0)]
    osm_km, todo_km = osm.length.sum() / 1000, todo.length.sum() / 1000

    streets = gpd.read_file(DATA_DIR / "osm_streets.gpkg", layer="streets")
    streets = streets[streets.highway.isin([
        "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"])]
    river = gpd.read_file(DATA_DIR / "osm_streets.gpkg", layer="river")
    osm_date = __import__("json").loads((DATA_DIR / "osm_raw.json").read_text(encoding="utf-8"))["osm3s"]["timestamp_osm_base"][:10]
    d = date.fromisoformat(osm_date)

    fig = plt.figure(figsize=(10.8, 13.5), dpi=200, facecolor=rp.BG)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.76])
    ax.set_facecolor(rp.BG)
    ax.set_axis_off()
    xmin, ymin, xmax, ymax = boundary.total_bounds
    target = (0.94 * 10.8) / (0.76 * 13.5)
    cx, cy, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
    w, h = (h * target, h) if w / h < target else (w, w / target)
    ax.set_xlim(cx - w / 2, cx + w / 2)
    ax.set_ylim(cy - h / 2, cy + h / 2)
    m_per_pt = w / (0.94 * 10.8 * 72)

    boundary.plot(ax=ax, color=rp.CITY, edgecolor="#b9beb6", linewidth=0.8, zorder=0)
    river.plot(ax=ax, color=rp.RIVER, linewidth=0, zorder=1)
    streets.plot(ax=ax, color=rp.STREET, linewidth=0.45, zorder=2)
    lw = 1.9
    for gdf, color, z in ((osm, OSM, 5), (todo, TODO, 6)):
        geom = shapely.offset_curve(gdf.geometry.values, -(lw / 2 + 0.4) * m_per_pt)
        gpd.GeoSeries(geom, crs=gdf.crs).plot(ax=ax, color=color, linewidth=lw, zorder=z,
                                              capstyle="round", joinstyle="round")

    ink, ink2, ink3, font = rp.INK, rp.INK2, rp.INK3, rp.FONT
    fig.text(0.05, 0.965, "Buspasy w OpenStreetMap", fontsize=30, fontweight="bold", color=ink,
             fontfamily=font, va="top")
    fig.text(0.05, 0.917, f"Stan na {d.day} {PL_MONTHS[d.month - 1]} {d.year}: {osm_km:.0f} km buspasów na trasach "
                          f"autobusowych (1 października: {BASELINE_KM:.0f} km).",
             fontsize=14, color=ink, fontfamily=font, va="top", fontweight="semibold")
    fig.text(0.05, 0.89, "Czy niebieskie się zgadzają? Czy pomarańczowe naprawdę istnieją? Daj znać: ulica, odcinek, kierunek.",
             fontsize=12, color=ink2, fontfamily=font, va="top")
    handles = [Line2D([0], [0], color=c, lw=4, solid_capstyle="round") for c in (OSM, TODO)]
    fig.legend(handles, [f"buspas zmapowany w OSM ({osm_km:.0f} km)",
                         f"wg mapy miasta (2021) lub komunikatów ZDM, brak w OSM ({todo_km:.0f} km)"],
               loc="upper left", bbox_to_anchor=(0.04, 0.868), frameon=False, labelcolor=ink2,
               prop={"family": font, "size": 12}, handlelength=2.2)
    fig.text(0.05, 0.03, "Km liczone osobno dla każdego kierunku, tylko ulice z autobusami ZTM. "
                         "Dane: © współtwórcy OpenStreetMap (ODbL), mapa.um.warszawa.pl, komunikaty ZDM, rozkład ZTM.",
             fontsize=9.5, color=ink3, fontfamily=font)
    out = OUTPUT_DIR / "buspasy_osm_stan.png"
    fig.savefig(out, facecolor=rp.BG, dpi=200)
    plt.close(fig)
    print(f"Saved {out}  (OSM {osm_km:.1f} km, missing {todo_km:.1f} km)")


if __name__ == "__main__":
    main()
