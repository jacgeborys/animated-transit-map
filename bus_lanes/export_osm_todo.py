"""
OSM editing helper: where Warsaw bus lanes are already tagged in OpenStreetMap (as a model) and
where other sources say there is a lane but OSM has none (to verify and tag).

Categories:
  osm      way-directions with a bus lane / bus-only road tagged in OSM (bus loops excluded)
  todo     bus-route way-directions with a lane in the city layer or manual list, none in OSM
  check    OSM lanes that a reviewer reported as non-existent (review page db export)

Direction is given relative to the OSM way ("forward"/"backward"), which decides the
:forward / :backward suffix of lane tags.

Outputs (bus_lanes/_output/osm/):
  buspasy_osm.html                  standalone page (Leaflet + own vector basemap), opens without login
  buspasy_osm_do_dodania.geojson    'todo' + 'check' features, loadable in iD as custom data
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).parent))
from bl_config import BL_DIR, DATA_DIR, OUTPUT_DIR, POLAND_CRS

OUT = OUTPUT_DIR / "osm"
TEMPLATE = BL_DIR / "osm" / "osm_template.html"
OFFSET_M = 3
TAG_KEYS = ("highway", "name", "oneway", "lanes", "lanes:forward", "lanes:backward", "access",
            "motor_vehicle", "motorcar", "junction")

# Reviewer verdicts "no such lane" (from the review page db), as (street, compass direction).
# City/manual ones are dropped from 'todo'; OSM ones become 'check'.
REPORTED_MISSING = [
    ("Kapucyńska", "płd.-zachód", "city"),
    ("Zygmunta Słomińskiego", "wschód", "osm"),
    ("Most Gdański", "wschód", "osm"),
]
COMPASS = ["północ", "płn.-wschód", "wschód", "płd.-wschód", "południe", "płd.-zachód", "zachód", "płn.-zachód"]


def compass(line) -> str:
    a, b = line.coords[0], line.coords[-1]
    h = np.degrees(np.arctan2(b[0] - a[0], b[1] - a[1])) % 360
    return COMPASS[int(((h + 22.5) % 360) // 45)]


def rel_tags(t: dict) -> dict:
    keep = {k: v for k, v in t.items()
            if k in TAG_KEYS or any(s in k for s in ("bus", "psv", "busway"))}
    return dict(sorted(keep.items()))


def to_ll(geom, offset=OFFSET_M):
    g = shapely.offset_curve(geom, -offset) if offset else geom
    if g.is_empty or g.geom_type != "LineString":
        g = geom
    g = gpd.GeoSeries([g], crs=POLAND_CRS).to_crs(4326).iloc[0]
    return [[round(y, 6), round(x, 6)] for x, y in g.coords]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads((DATA_DIR / "osm_raw.json").read_text(encoding="utf-8"))
    tags = {e["id"]: e.get("tags", {}) for e in raw["elements"] if e["type"] == "way"}
    st = gpd.read_file(DATA_DIR / "osm_streets.gpkg", layer="streets")
    city = gpd.read_file(OUTPUT_DIR / "bus_lane_need.gpkg", layer="boundary").geometry.iloc[0]
    st["name"] = st["name"].where(st["name"].notna(), "")
    bus = gpd.read_file(OUTPUT_DIR / "bus_lane_need.gpkg", layer="bus_dir")
    bus["name"] = bus["name"].where(bus["name"].notna(), "")
    extra = gpd.read_file(DATA_DIR / "extra_lanes.gpkg", layer="lanes")
    xtree = shapely.STRtree(extra.geometry.values)

    feats = []
    # --- existing OSM lanes (model) ---
    par_ids = set(gpd.read_file(OUTPUT_DIR / "bus_lane_need.gpkg", layer="osm_parallel_busways").osm_id)
    o = st[((st.bl_fwd == 1) | (st.bl_bwd == 1)) & st.intersects(city)
           & ~((st.bus_only == 1) & (st.highway == "service") & ~st.osm_id.isin(par_ids))]  # drop loops only
    for r in o.itertuples():
        for d, flag in ((1, r.bl_fwd), (-1, r.bl_bwd)):
            if not flag:
                continue
            g = r.geometry if d == 1 else r.geometry.reverse()
            feats.append({"cat": "osm", "way": int(r.osm_id), "name": r.name or "", "rel": "forward" if d == 1 else "backward",
                          "oneway": int(r.oneway), "dir": compass(g), "busOnly": bool(r.bus_only),
                          "tags": rel_tags(tags.get(r.osm_id, {})), "line": to_ll(g), "geom": g})

    # --- candidates: lane in city layer / manual list, none in OSM ---
    c = bus[bus.in_city & ((bus.bl_city == 1) | (bus.bl_manual == 1)) & (bus.bl_osm == 0)]
    for r in c.itertuples():
        g = r.geometry  # already drawn in travel direction
        dname = compass(g)
        if any(n == r.name and d == dname and s in ("city", "manual") for n, d, s in REPORTED_MISSING):
            continue
        hint = []
        mid = g.interpolate(0.5, normalized=True)
        for i in xtree.query(mid, predicate="dwithin", distance=30):
            e = extra.iloc[i]
            src = "mapa miasta (2021)" if e.zrodlo == "miasto" else "komunikat ZDM / zgłoszenie"
            txt = f"{src}: {e.ulica or ''}, {e.odcinek or ''}"
            if e.dni:
                txt += f" — {e.dni}, {e.godziny}"
            if e.otwarty:
                txt += f" (otwarty {e.otwarty})"
            if txt not in hint:
                hint.append(txt)
        feats.append({"cat": "todo", "way": int(r.osm_id), "name": r.name or "", "rel": "forward" if r.dir == 1 else "backward",
                      "oneway": int(r.oneway), "dir": dname, "perH": round(float(r.per_h), 1),
                      "hint": hint, "tags": rel_tags(tags.get(r.osm_id, {})), "line": to_ll(g), "geom": g})

    # --- OSM lanes reported as non-existent ---
    for f in feats:
        if f["cat"] == "osm" and any(n == f["name"] and d == f["dir"] and s == "osm" for n, d, s in REPORTED_MISSING):
            f["cat"] = "check"

    # nearest existing OSM lane as a tagging example for each candidate
    model = [f for f in feats if f["cat"] == "osm" and not f["busOnly"]]
    mtree = shapely.STRtree([f["geom"] for f in model])
    for f in feats:
        if f["cat"] == "todo" and model:
            same = [m for m in model if bool(m["oneway"]) == bool(f["oneway"])] or model
            near = min(same, key=lambda m: m["geom"].distance(f["geom"]))
            f["example"] = {"way": near["way"], "name": near["name"], "rel": near["rel"],
                            "dist": int(near["geom"].distance(f["geom"])), "tags": near["tags"]}
    for f in feats:
        f.pop("geom")

    counts = pd.Series([f["cat"] for f in feats]).value_counts().to_dict()
    print(f"Features: {counts}")
    date = (DATA_DIR / "analysis_date.txt").read_text(encoding="utf-8").split()[0]
    osm_date = pd.Timestamp((DATA_DIR / "osm_raw.json").stat().st_mtime, unit="s").strftime("%Y-%m-%d")
    # which lane-tagging schemes Warsaw mappers actually use (for the guide)
    schemes = pd.Series([k for f in feats if f["cat"] == "osm" and not f["busOnly"] for k in f["tags"]
                         if any(x in k for x in ("psv", "bus")) and k not in ("bus_bay", "maxspeed:bus", "toll:bus")])
    usage = schemes.value_counts().head(8).to_dict()
    print("Tag usage on existing lanes:", usage)
    from export_review import build_base  # same vector basemap as the review page
    data = {"osmDate": osm_date, "gtfsDate": date, "usage": usage, "features": feats, "base": build_base()}

    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "LineString", "coordinates": [[x, y] for y, x in f["line"]]},
         "properties": {"kategoria": "do dodania" if f["cat"] == "todo" else "do sprawdzenia",
                        "osm_way": f["way"], "ulica": f["name"], "kierunek": f["dir"],
                        "wzgledem_linii_osm": f["rel"], "wskazowki": " | ".join(f.get("hint", [])),
                        "stroke": "#e8590c" if f["cat"] == "todo" else "#c2255c", "stroke-width": 5}}
        for f in feats if f["cat"] in ("todo", "check")]}
    (OUT / "buspasy_osm_do_dodania.geojson").write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUT / 'buspasy_osm_do_dodania.geojson'} ({len(gj['features'])} features)")

    if TEMPLATE.exists():
        html = TEMPLATE.read_text(encoding="utf-8").replace(
            "/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        (OUT / "buspasy_osm.html").write_text(html, encoding="utf-8")
        print(f"Saved {OUT / 'buspasy_osm.html'} ({len(html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
