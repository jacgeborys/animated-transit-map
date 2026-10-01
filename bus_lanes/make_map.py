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
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath, TextToPath
from matplotlib.transforms import Affine2D
from shapely.geometry import LineString

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
COL = {"lane": "#4c9bff", "high": "#ff3d54", "medium": "#ffb020", "low": "#5a6170"}
ORDER = ["low", "lane", "medium", "high"]  # draw order (last on top)
FONT = "Segoe UI"
# Street labels: condensed face (loaded from file — matplotlib's font cache may not list it)
_NARROW = Path(r"C:\Windows\Fonts\LiberationSansNarrow-Regular.ttf")
LABEL_TRACKING = 0.13          # extra space between letters, in em
LABEL_SQUEEZE = 0.82           # horizontal glyph scale (< 1 = narrower than the font itself)
# Per-street label tweaks (keys = short names as printed on the map)
LABEL_SIDE = {"Radzymińska": -1, "Łopuszańska": -1, "Bora-Komorowskiego": -1, "Światowida": -1}   # -1 = right of / below the street
LABEL_AT = {"Puławska": "south"}                       # label the southern end instead
LABEL_NUDGE = {"Modlińska": (-600, 1000), "Czerniakowska": (0, -2500)}  # city map only: move anchor (dx, dy) m
CITY_EXTRA_LABELS = ["Łopuszańska", "Puławska", "Modlińska", "Aleja Armii Krajowej"]  # always labelled (if not in top 12)
LABEL_PRIORITY = ["Bora-Komorowskiego"]   # placed first, others dodge them


def label_font(size):
    if _NARROW.exists():
        return FontProperties(fname=str(_NARROW), size=size)
    return FontProperties(family=FONT, size=size)

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

    fs = 11 if scale > 1.5 else 10
    gap_m = (width(60, scale) / 2 + fs * 0.75) * m_per_pt  # clear the thickest line
    frame = shapely.box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2).buffer(-4 * fs * m_per_pt)
    placed = []  # glyph positions of labels already drawn (priority first, then ranking order)
    order = sorted(labels, key=lambda k: k not in LABEL_PRIORITY)
    for name in order:
        ways, target, side = labels[name]
        curved_label(ax, name, ways, target, fs, m_per_pt, gap_m * side, frame, placed)

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
             "Dane: ZTM (GTFS via mkuran.pl) · ulice i buspasy © OpenStreetMap (ODbL) · buspasy: mapa.um.warszawa.pl, komunikaty ZDM",
             fontsize=10, color=INK3, fontfamily=FONT)
    fig.savefig(out, facecolor=BG, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def label_points(bus, streets, names, nudge=False):
    """Per street: (all same-named OSM ways, point to label, side). The point sits on the
    street's high/medium stretch, or anywhere on its bus route if it has none left."""
    out = {}
    for name in dict.fromkeys(names):  # de-duplicate, keep order
        street = bus[bus.name == name]
        gaps = street[street.status.isin(["high", "medium"])]
        ways = streets[streets.name == name]
        if street.empty or ways.empty:
            continue
        if gaps.empty:
            gaps = street
        label = short(name)
        if LABEL_AT.get(label) == "south":
            g = street[street.in_city]  # southern end of the street, whatever its status
            xy = shapely.get_coordinates(g.geometry.values)
            target = shapely.Point(xy[np.argmin(xy[:, 1])])
        else:
            longest = gaps.geometry.iloc[int(np.argmax(gaps.length.to_numpy()))]
            target = longest.interpolate(0.5, normalized=True)
        if nudge and label in LABEL_NUDGE:
            dx, dy = LABEL_NUDGE[label]
            target = shapely.Point(target.x + dx, target.y + dy)
        out[label] = (list(ways.geometry.values), target, LABEL_SIDE.get(label, 1))
    return out




def _point_along(coords, dist):
    """Point `dist` metres along a coordinate array from its first vertex (or its end)."""
    line = LineString(coords)
    return np.asarray(line.interpolate(min(dist, line.length)).coords[0])


def centerline_path(ways, target, radius, frame=None, nbins=14):
    """Smooth street centreline around `target`: densified vertices of all same-named ways
    within `radius`, binned along the local main axis, median per bin. On dual carriageways
    this lands between the two roadways — where a street label belongs."""
    pts = np.vstack([shapely.get_coordinates(shapely.segmentize(w, 10)) for w in ways])
    t0 = np.asarray(target.coords[0])
    pts = pts[np.hypot(*(pts - t0).T) <= radius]
    if frame is not None and len(pts):
        pts = pts[shapely.contains_xy(frame, pts[:, 0], pts[:, 1])]
    if len(pts) < 10:
        return None
    ctr = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - ctr, full_matrices=False)
    t = (pts - ctr) @ vt[0]
    edges = np.linspace(t.min(), t.max(), nbins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, nbins - 1)
    cl = np.array([np.median(pts[idx == k], axis=0) for k in range(nbins) if (idx == k).any()])
    return LineString(cl) if len(cl) >= 3 else None


def _smooth(line: LineString, tol: float, iters=4) -> LineString:
    """Simplify, then Chaikin corner-cutting, so letters follow a calm curve."""
    c = np.asarray(line.simplify(tol).coords)
    for _ in range(iters):
        if len(c) < 3:
            break
        q = 0.75 * c[:-1] + 0.25 * c[1:]
        r = 0.25 * c[:-1] + 0.75 * c[1:]
        c = np.vstack([c[:1], np.column_stack([q, r]).reshape(-1, 2), c[-1:]])
    return LineString(c)


_T2P = TextToPath()


def _extend(line: LineString, d: float) -> LineString:
    """Prolong both ends straight along their ~25 m end direction by d metres."""
    c = np.asarray(line.coords)
    def tip(pts):
        a, b = _point_along(pts[::-1], 25), pts[-1]
        v = (b - a) / max(np.hypot(*(b - a)), 1e-9)
        return b + v * d
    return LineString(np.vstack([tip(c[::-1]), c, tip(c)]))


def curved_label(ax, text, ways, target, fs, m_per_pt, gap_m, frame, placed):
    """Street label set letter by letter along the street, parallel and offset to one side."""
    prop = label_font(fs)
    adv = lambda t: LABEL_SQUEEZE * _T2P.get_text_width_height_descent(t, prop, ismath=False)[0]  # pt
    track = LABEL_TRACKING * fs  # pt
    total = (adv(text) + track * (len(text) - 1)) * m_per_pt
    if frame is not None and not frame.contains(target):
        # anchor fell in the edge margin: move it onto the street, ~half a label inside the frame
        pts = np.vstack([shapely.get_coordinates(shapely.segmentize(w, 20)) for w in ways])
        inner = frame.buffer(-total * 0.6)
        pts = pts[shapely.contains_xy(inner, pts[:, 0], pts[:, 1])]
        if len(pts):
            t0 = np.asarray(target.coords[0])
            target = shapely.Point(pts[np.argmin(np.hypot(*(pts - t0).T))])
    line = centerline_path(ways, target, total * 0.8, frame)
    if line is None or line.length < total * 0.5:
        print(f"  label skipped (street too short in view): {text}")
        return
    if line.length < total * 1.3:  # short street: let the label overhang both ends
        line = _extend(line, (total * 1.3 - line.length) / 2)
    path = _smooth(line, tol=total * 0.06, iters=5)
    # centre the label on the target stretch, kept fully on the path
    mid = path.project(target)
    mid = min(max(mid, total / 2 + 1), path.length - total / 2 - 1)
    a, b = path.interpolate(mid - total / 4), path.interpolate(mid + total / 4)
    if b.x < a.x:  # read left → right
        path = path.reverse()
        mid = path.length - mid
    # shift to the left of travel (= above, after the flip) so the text clears the line
    off = path.offset_curve(gap_m)
    if off.geom_type == "MultiLineString":
        off = max(off.geoms, key=lambda g: g.length)
    if off.is_empty or off.length < total * 1.05:
        off = path
    mid = off.project(path.interpolate(mid))
    L = off.length

    def layout(start):
        glyphs = []
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            centre = start + (adv(text[:i]) + adv(ch) / 2 + track * i) * m_per_pt
            p0 = off.interpolate(max(centre - 0.6 * fs * m_per_pt, 0))
            p1 = off.interpolate(min(centre + 0.6 * fs * m_per_pt, L))
            p = off.interpolate(centre)
            glyphs.append((ch, p.x, p.y, np.degrees(np.arctan2(p1.y - p0.y, p1.x - p0.x))))
        return glyphs

    # try the target position first, then slide along the street to dodge earlier labels
    clear = 1.0 * fs * m_per_pt  # min distance between glyphs of different labels
    base = min(max(mid - total / 2, 0), max(L - total, 0))
    for shift in (0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5):
        start = base + shift * total
        if start < 0 or start > max(L - total, 0):
            continue
        glyphs = layout(start)
        xy = np.array([(g[1], g[2]) for g in glyphs])
        blockers = [n for n, q in placed
                    if np.min(np.hypot(*(xy[:, None, :] - q[None, :, :]).transpose(2, 0, 1))) <= clear]
        if not blockers:
            break
    else:
        print(f"  label skipped (collides with {', '.join(blockers)}): {text}")
        return
    placed.append((text, xy))
    # glyphs drawn as outlines so they can be squeezed horizontally (data units = metres);
    # round joins, otherwise the halo's miter joins spike out of sharp glyph corners
    halo = [pe.withStroke(linewidth=1.8, foreground=BG, joinstyle="round", capstyle="round")]
    for ch, x, y, ang in glyphs:
        tp = TextPath((0, 0), ch, prop=prop)  # units: points, baseline at y=0
        tr = (Affine2D().translate(-adv(ch) / LABEL_SQUEEZE / 2, -0.36 * fs)
              .scale(LABEL_SQUEEZE * m_per_pt, m_per_pt).rotate_deg(ang).translate(x, y))
        ax.add_patch(PathPatch(tr.transform_path(tp), facecolor=INK, edgecolor="none",
                               zorder=10, path_effects=halo))


def short(name: str) -> str:
    if "Armii Krajowej" in name:
        return "AK"
    if "Trasa Łazienkowska" in name:
        return "TŁ"
    if "Bora-Komorowskiego" in name:
        return "Bora-Komorowskiego"
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
                          "buspasy: OpenStreetMap, mapa.um.warszawa.pl, komunikaty ZDM",
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

    named = gpd.read_file(osm, layer="streets", columns=["name", "highway"])
    named = named[named.name.notna() & ~named.highway.isin(["service", "living_street"])
                  & ~named.highway.str.endswith("_link")]
    city = boundary.total_bounds
    top = rank.drop_duplicates("ulica").head(12).ulica.tolist() + CITY_EXTRA_LABELS
    draw_map(bus, streets, river, boundary, date, city, OUTPUT_DIR / "buspasy_warszawa.png",
             "Gdzie brakuje buspasów?", scale=1.3, labels=label_points(bus, named, top, nudge=True))

    # Central zoom: ~9 x 11 km around Śródmieście
    cx, cy = 638500, 486800   # EPSG:2180, near Rondo Dmowskiego
    ext = (cx - 5500, cy - 5500, cx + 5500, cy + 6500)
    in_ext = bus.cx[ext[0]:ext[2], ext[1]:ext[3]]
    top_c = (in_ext[in_ext.status == "high"].assign(L=lambda d: d.length)
             .groupby("name").L.sum().sort_values(ascending=False).head(12).index.tolist())
    lab = label_points(bus, named, top_c)
    draw_map(bus, streets, river, boundary, date, ext, OUTPUT_DIR / "buspasy_centrum.png",
             "Buspasy w centrum", scale=1.8, labels=lab)

    draw_ranking(rank, date, OUTPUT_DIR / "buspasy_ranking.png")


if __name__ == "__main__":
    main()
