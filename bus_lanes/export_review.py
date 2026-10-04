"""
Export the current bus-lane set for the colleague review page (bus_lanes/review/).

Lanes = street-directions on bus routes in Warsaw that have a lane in ANY source (OSM tags,
city layer, manual list), as decided by match_streets.py. Consecutive pieces with the same
street, sources and direction are merged into one stretch, so reviewers judge stretches
rather than hundreds of OSM fragments.

Output: bus_lanes/_output/review_data.json (WGS84, coordinates rounded to ~1 m)
        bus_lanes/_output/buspasy_do_weryfikacji.geojson  (for a no-login uMap, umap.openstreetmap.fr)
        bus_lanes/_output/weryfikacja_buspasow.html  (template + data, ready to publish)
"""
import hashlib
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).parent))
from bl_config import BL_DIR, DATA_DIR, OUTPUT_DIR, POLAND_CRS

TEMPLATE = BL_DIR / "review" / "review_template.html"
ARROW_EVERY_M = 120      # chevron spacing along a lane
ARROW_SIZE_M = 9         # chevron arm length
LANE_OFFSET_M = 4        # draw each direction slightly to the right of travel

COMPASS = ["północ", "płn.-wschód", "wschód", "płd.-wschód", "południe", "płd.-zachód", "zachód",
           "płn.-zachód"]
SOURCE_LABEL = {"osm": "OpenStreetMap", "city": "mapa miasta (stan 2021)", "manual": "komunikat ZDM / zgłoszenie"}


def heading(line: LineString) -> float:
    a, b = line.coords[0], line.coords[-1]
    return float(np.degrees(np.arctan2(b[0] - a[0], b[1] - a[1])) % 360)


def compass(h: float) -> str:
    return COMPASS[int(((h + 22.5) % 360) // 45)]


def to_ll(geoms, tol=1.0):
    """EPSG:2180 geometries → lists of [lat, lon] rounded to 5 decimals (~1 m)."""
    s = gpd.GeoSeries(geoms, crs=POLAND_CRS).simplify(tol).to_crs(4326)
    out = []
    for g in s:
        parts = list(getattr(g, "geoms", [g]))
        out.append([[[round(y, 5), round(x, 5)] for x, y in p.coords] for p in parts if not p.is_empty])
    return out


def chevrons(line: LineString):
    """Small arrowheads along a directional line (EPSG:2180)."""
    out = []
    L = line.length
    for d in np.arange(min(ARROW_EVERY_M / 2, L / 2), L, ARROW_EVERY_M):
        p = np.asarray(line.interpolate(d).coords[0])
        q = np.asarray(line.interpolate(max(d - 5, 0)).coords[0])
        v = p - q
        n = np.hypot(*v)
        if n == 0:
            continue
        v /= n
        perp = np.array([-v[1], v[0]])
        back = p - v * ARROW_SIZE_M
        out.append(LineString([back + perp * ARROW_SIZE_M * 0.7, p, back - perp * ARROW_SIZE_M * 0.7]))
    return out


def build_lanes():
    bus = gpd.read_file(OUTPUT_DIR / "bus_lane_need.gpkg", layer="bus_dir")
    bus = bus[bus.in_city & (bus.bus_lane == 1) & (bus.loop == 0)].copy()
    bus["name"] = bus["name"].fillna("(bez nazwy)")
    extra = gpd.read_file(DATA_DIR / "extra_lanes.gpkg", layer="lanes")
    extra_tree = shapely.STRtree(extra.geometry.values)

    rows = []
    # Split each street's pieces into its two travel directions (relative to the street's main
    # axis) BEFORE merging. Merging both at once — or via union_all — collapses a two-way way's
    # forward and backward pieces into one line and silently drops a direction.
    s0 = shapely.get_coordinates(shapely.get_point(bus.geometry.values, 0))
    s1 = shapely.get_coordinates(shapely.get_point(bus.geometry.values, -1))
    bus["hd"] = np.degrees(np.arctan2(s1[:, 0] - s0[:, 0], s1[:, 1] - s0[:, 1])) % 360
    ang2 = np.radians(2 * bus.hd)
    w = bus.geometry.length
    ax = (bus.assign(sx=np.sin(ang2) * w, cx=np.cos(ang2) * w).groupby("name")[["sx", "cx"]].sum())
    axis = bus.name.map((np.degrees(np.arctan2(ax.sx, ax.cx)) / 2) % 180)
    bus["along"] = (np.cos(np.radians(bus.hd - axis)) >= 0).astype(int)

    keys = ["name", "bl_osm", "bl_city", "bl_manual", "bus_only", "along"]
    for key, g in bus.groupby(keys):
        merged = shapely.line_merge(shapely.MultiLineString(list(g.geometry.values)), directed=True)
        mids = shapely.line_interpolate_point(g.geometry.values, 0.5, normalized=True)
        for chain in getattr(merged, "geoms", [merged]):
            if chain.length < 15:
                continue
            pieces = g[shapely.dwithin(mids, chain, 1.0)]
            h = heading(chain)
            # city / manual attributes from the nearest same-direction extra-lane line
            mid = chain.interpolate(0.5, normalized=True)
            info = {}
            for i in extra_tree.query(mid, predicate="dwithin", distance=30):
                e = extra.iloc[i]
                eh = heading(e.geometry)
                if abs((eh - h + 180) % 360 - 180) < 60:
                    info.setdefault(e.zrodlo, e)
            src = [s for s, f in zip(("osm", "city", "manual"), key[1:4]) if f]
            city = info.get("miasto")
            manual = info.get("reczne")
            lid = hashlib.sha1(f"{key[0]}|{np.round(chain.coords[0], 0)}|{np.round(chain.coords[-1], 0)}"
                               .encode()).hexdigest()[:10]
            rows.append({
                "id": lid,
                "street": key[0],
                "dir": compass(h),
                "len": int(round(chain.length)),
                "src": src,
                "busOnly": bool(key[4]),
                "perH": float(round(pieces.per_h.max(), 1)) if len(pieces) else None,
                "cityOdcinek": city.odcinek if city is not None else None,
                "cityDni": city.dni if city is not None else None,
                "cityGodziny": city.godziny if city is not None else None,
                "manualId": manual.get("id") if manual is not None else None,
                "manualNote": manual.uwagi if manual is not None else None,
                "opened": manual.otwarty if manual is not None else None,
                "geom": chain,
            })
    lanes = pd.DataFrame(rows)
    # draw slightly right of travel, plus chevrons
    drawn = [shapely.offset_curve(gm, -LANE_OFFSET_M) for gm in lanes.geom]
    drawn = [d if d.geom_type == "LineString" and not d.is_empty else gm for d, gm in zip(drawn, lanes.geom)]
    lanes["line"] = to_ll(drawn, tol=1.5)
    lanes["arrows"] = [[a[0] for a in to_ll(chevrons(d), tol=0)] for d in drawn]
    lanes = lanes.drop(columns="geom").replace({np.nan: None})
    return lanes


def build_base():
    osm = DATA_DIR / "osm_streets.gpkg"
    streets = gpd.read_file(osm, layer="streets", columns=["name", "highway"])
    boundary = gpd.read_file(osm, layer="boundary")
    river = gpd.read_file(osm, layer="river")
    city = boundary.geometry.iloc[0].buffer(1500)
    streets = streets[streets.intersects(city)]
    major = streets[streets.highway.isin(["motorway", "trunk", "primary", "secondary", "tertiary",
                                          "motorway_link", "trunk_link", "primary_link", "secondary_link"])]
    minor = streets[streets.highway.isin(["unclassified", "residential", "living_street", "tertiary_link"])]

    def lines(gdf, tol):
        u = shapely.line_merge(shapely.union_all(gdf.geometry.values))
        return [p for p in to_ll([u], tol=tol)[0] if len(p) > 1]

    # one label point per named major street (middle of its longest piece)
    labels = []
    named = major[major.name.notna() & ~major.highway.str.endswith("_link")]
    for name, g in named.groupby("name"):
        if g.length.sum() < 600:
            continue
        longest = g.geometry.iloc[int(np.argmax(g.length.to_numpy()))]
        p = gpd.GeoSeries([longest.interpolate(0.5, normalized=True)], crs=POLAND_CRS).to_crs(4326).iloc[0]
        labels.append([round(p.y, 5), round(p.x, 5), name])

    river_ll = []
    for poly in gpd.GeoSeries(river.geometry, crs=POLAND_CRS).simplify(5).to_crs(4326):
        for part in getattr(poly, "geoms", [poly]):
            river_ll.append([[round(y, 5), round(x, 5)] for x, y in part.exterior.coords])
    bnd = gpd.GeoSeries(boundary.geometry, crs=POLAND_CRS).simplify(10).to_crs(4326).iloc[0]
    bnd_ll = [[[round(y, 5), round(x, 5)] for x, y in p.exterior.coords] for p in getattr(bnd, "geoms", [bnd])]
    return {"major": lines(major, 2), "minor": lines(minor, 2), "river": river_ll,
            "boundary": bnd_ll, "labels": labels}


UMAP_COLORS = {  # how much we trust a stretch before review
    "multi": "#1f63d1",   # in 2+ sources
    "city": "#e08a00",    # only the city map (state 2021) — most likely outdated
    "osm": "#0f9b8e",     # only OpenStreetMap
    "manual": "#7b45c9",  # announcement / reported by us
}


def write_umap(lanes: pd.DataFrame, date: str, out: Path):
    """GeoJSON for uMap (umap.openstreetmap.fr): one feature per stretch, arrows included,
    per-feature style in `_umap_options`, empty `status` / `uwagi` fields for reviewers."""
    feats = []
    for l in lanes.itertuples():
        src = list(l.src)
        kind = "multi" if len(src) > 1 else src[0]
        desc = [f"Kierunek jazdy: **{l.dir}**", f"Długość: {l.len} m",
                "Źródło: " + ", ".join(SOURCE_LABEL[x] for x in src)]
        if l.busOnly:
            desc.append("Ulica tylko dla autobusów")
        if l.cityOdcinek:
            desc.append(f"Mapa miasta: {l.cityOdcinek}, {l.cityDni}, {l.cityGodziny}")
        if l.opened:
            desc.append(f"Otwarty: {l.opened}")
        if l.manualNote:
            desc.append(f"Uwagi: {l.manualNote}")
        if l.perH is not None:
            desc.append(f"Autobusy w szczycie: {l.perH}/h (rozkład ZTM {date[6:]}.{date[4:6]}.{date[:4]})")
        parts = [[[x, y] for y, x in part] for part in [l.line[0], *l.arrows]]
        feats.append({
            "type": "Feature",
            "geometry": {"type": "MultiLineString", "coordinates": parts},
            "properties": {
                "name": f"{l.street} → {l.dir}",
                "description": "\n".join(desc),
                "id": l.id,
                "status": "",
                "uwagi": "",
                "_umap_options": {"color": UMAP_COLORS[kind], "weight": 5, "opacity": 0.9},
            },
        })
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False),
                   encoding="utf-8")
    print(f"Saved {out} ({len(feats)} features)")


def main():
    date = (DATA_DIR / "analysis_date.txt").read_text(encoding="utf-8").split()[0]
    lanes = build_lanes()
    base = build_base()
    data = {"date": date, "sources": SOURCE_LABEL, "lanes": lanes.to_dict("records"), "base": base}
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    (OUTPUT_DIR / "review_data.json").write_text(js, encoding="utf-8")
    # uMap export retired (colleagues found uMap too complicated); call write_umap() manually if ever needed
    print(f"Lanes: {len(lanes)} stretches, {lanes.len.sum() / 1000:.1f} km; "
          f"by source: {pd.Series([s for l in lanes.src for s in l]).value_counts().to_dict()}")
    print(f"Base: {len(base['major'])} major + {len(base['minor'])} minor lines, {len(base['labels'])} labels")
    print(f"JSON size: {len(js) / 1e6:.2f} MB")
    if TEMPLATE.exists():
        html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", js)
        out = OUTPUT_DIR / "weryfikacja_buspasow.html"
        out.write_text(html, encoding="utf-8")
        print(f"Saved {out} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
