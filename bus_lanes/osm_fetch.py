"""
Step A — Fetch Warsaw street network + bus lane tags from OpenStreetMap (Overpass API).

Outputs (bus_lanes/_data/):
    osm_raw.json          raw Overpass response (cache)
Usage: osm_fetch.py            use the cache (download if missing)
       osm_fetch.py --update   merge in only ways changed since the cached snapshot (fast)
       osm_fetch.py --refresh  full re-download (9 tiles, resumable)
    osm_streets.gpkg      layer 'streets'  — drivable ways with parsed bus-lane flags
                          layer 'trams'    — railway=tram tracks
                          layer 'boundary' — Warsaw city boundary
                          layer 'river'    — river areas (Vistula), for the map background
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

sys.path.insert(0, str(Path(__file__).parent))
from bl_config import BBOX, DATA_DIR, POLAND_CRS

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",   # often months stale; the freshness check skips it
]

HIGHWAYS = (
    "motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|"
    "tertiary|tertiary_link|unclassified|residential|living_street|service|busway|bus_guideway|road"
)


def build_query(bbox=BBOX, boundary=True, changed_since=None):
    s, w, n, e = bbox
    ch = f'(changed:"{changed_since}")' if changed_since else ""
    q = f"""
[out:json][timeout:180];
(
  way["highway"~"^({HIGHWAYS})$"]{ch}({s},{w},{n},{e});
  way["railway"="tram"]{ch}({s},{w},{n},{e});
);
out tags geom qt;
"""
    if boundary:
        q += 'rel["boundary"="administrative"]["admin_level"="6"]["name"="Warszawa"];\nout geom;\n'
    return q


def tiles(nx=3, ny=3):
    s, w, n, e = BBOX
    for i in range(nx):
        for j in range(ny):
            yield (s + (n - s) * j / ny, w + (e - w) * i / nx, s + (n - s) * (j + 1) / ny, w + (e - w) * (i + 1) / nx)


def river_query():
    s, w, n, e = BBOX
    return f"""
[out:json][timeout:120];
(
  way["natural"="water"]["water"="river"]({s},{w},{n},{e});
  rel["natural"="water"]["water"="river"]({s},{w},{n},{e});
);
out geom;
"""


MAX_AGE_DAYS = 3   # mirrors can lag months behind; refuse data older than this


def _post(query: str, rounds=4) -> dict:
    """POST to the first server that answers with fresh data; back off between rounds
    (honouring Retry-After on 429), so a busy server is not hammered."""
    for rnd in range(rounds):
        for url in OVERPASS_URLS:
            try:
                print(f"  {url.split('/')[2]} ...", flush=True)
                r = requests.post(url, data={"data": query}, timeout=240,
                                  headers={"User-Agent": "warsaw-bus-lane-map/1.0 (gtfs_schedules_city)",
                                           "Accept": "application/json"})
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", "0") or 0)
                    print(f"    busy (429){f', retry after {wait}s' if wait else ''}", flush=True)
                    continue
                r.raise_for_status()
                data = r.json()
                ts = data.get("osm3s", {}).get("timestamp_osm_base", "")
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts.replace("Z", "+00:00"))).days if ts else 999
                if age > MAX_AGE_DAYS:
                    print(f"    stale data ({ts}) — skipped", flush=True)
                    continue
                return data
            except Exception as ex:
                print(f"    failed: {str(ex)[:120]}", flush=True)
        wait = 30 * (rnd + 1)
        print(f"  all servers busy, waiting {wait}s (round {rnd + 1}/{rounds})", flush=True)
        time.sleep(wait)
    raise RuntimeError("No Overpass server returned fresh data")


def _merge(parts):
    seen, elements = {}, []
    for part in parts:
        for el in part["elements"]:
            key = (el["type"], el["id"])
            if key in seen:
                elements[seen[key]] = el      # later part wins (newer data in update mode)
            else:
                seen[key] = len(elements)
                elements.append(el)
    return elements


def fetch(cache: Path, query: str = None, mode: str = "cache") -> dict:
    """mode: 'cache' (use cache if present), 'refresh' (full download), 'update' (only ways changed
    since the cached snapshot, merged in — seconds instead of minutes; deleted ways are not removed)."""
    if query:  # small one-off query (river)
        if cache.exists():
            print(f"Using cached {cache.name}")
            return json.loads(cache.read_text(encoding="utf-8"))
        data = _post(query)
    elif cache.exists() and mode == "cache":
        print(f"Using cached {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))
    elif cache.exists() and mode == "update":
        old = json.loads(cache.read_text(encoding="utf-8"))
        since = old["osm3s"]["timestamp_osm_base"]
        print(f"Update: ways changed since {since}")
        new = _post(build_query(BBOX, boundary=False, changed_since=since))
        print(f"  {len(new['elements'])} changed ways")
        data = {"osm3s": new["osm3s"], "elements": _merge([old, new])}
    else:
        # full download in 9 tiles, each saved on arrival so an interrupted run resumes
        tile_dir = cache.parent / "osm_tiles"
        tile_dir.mkdir(exist_ok=True)
        parts = []
        all_tiles = list(tiles())
        for k, bb in enumerate(all_tiles):
            tf = tile_dir / f"tile_{k}.json"
            if tf.exists():
                print(f"Tile {k + 1}/{len(all_tiles)}: cached")
                parts.append(json.loads(tf.read_text(encoding="utf-8")))
                continue
            print(f"Tile {k + 1}/{len(all_tiles)}", flush=True)
            part = _post(build_query(bb, boundary=(k == 0)))
            tf.write_text(json.dumps(part), encoding="utf-8")
            parts.append(part)
        # oldest tile timestamp = safe base for later incremental updates
        base = min((p["osm3s"] for p in parts), key=lambda o: o.get("timestamp_osm_base", ""))
        data = {"osm3s": base, "elements": _merge(parts)}
        for tf in tile_dir.glob("tile_*.json"):
            tf.unlink()
        tile_dir.rmdir()
    print(f"OSM data as of {data.get('osm3s', {}).get('timestamp_osm_base')}")
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


# ---------------------------------------------------------------------------
# Bus lane tag parsing
# ---------------------------------------------------------------------------
BUS_VALUES = {"designated", "lane", "yes"}


def _lanes_has_bus(value) -> bool:
    """'bus:lanes'/'psv:lanes' style value, e.g. 'yes|designated' → any designated lane."""
    if not value:
        return False
    return any(v.strip() == "designated" for v in value.split("|"))


def _count_positive(value) -> bool:
    """'lanes:bus' / 'lanes:psv' numeric value."""
    try:
        return int(str(value).split(";")[0]) > 0
    except ValueError:
        return False


def bus_lane_flags(tags: dict) -> tuple[bool, bool, bool]:
    """
    Returns (lane_forward, lane_backward, bus_only_road).

    forward/backward are relative to OSM way node order. Handles:
      busway=lane|opposite_lane, busway:both/left/right, bus:lanes(:forward/:backward),
      psv:lanes(...), lanes:bus(:forward/:backward), lanes:psv(...),
      highway=busway, and roads closed to general traffic but open to bus/psv.
    Right-hand traffic: busway:right → forward, busway:left → backward (unless oneway).
    """
    t = tags
    oneway = t.get("oneway") in ("yes", "1", "true") or t.get("highway") in ("motorway",) \
        or t.get("junction") == "roundabout"
    oneway_rev = t.get("oneway") == "-1"

    fwd = bwd = False

    # busway=*
    bw = t.get("busway")
    if bw == "lane":
        fwd = True
        if not oneway and not oneway_rev:
            bwd = True
    elif bw == "opposite_lane":
        bwd = True
    if t.get("busway:both") in BUS_VALUES:
        fwd = bwd = True
    if t.get("busway:right") in BUS_VALUES:
        if oneway_rev:
            bwd = True
        else:
            fwd = True
    if t.get("busway:left") in BUS_VALUES:
        # on a oneway, a left-side bus lane still goes with traffic
        if oneway:
            fwd = True
        else:
            bwd = True

    # lane-based schemes
    for mode in ("bus", "psv"):
        if _lanes_has_bus(t.get(f"{mode}:lanes")):
            if oneway_rev:
                bwd = True
            else:
                fwd = True
        if _lanes_has_bus(t.get(f"{mode}:lanes:forward")):
            fwd = True
        if _lanes_has_bus(t.get(f"{mode}:lanes:backward")):
            bwd = True
        if _count_positive(t.get(f"lanes:{mode}")):
            if oneway or oneway_rev:
                fwd = fwd or not oneway_rev
                bwd = bwd or oneway_rev
            else:
                fwd = bwd = True
        if _count_positive(t.get(f"lanes:{mode}:forward")):
            fwd = True
        if _count_positive(t.get(f"lanes:{mode}:backward")):
            bwd = True

    # whole road reserved for buses/PSV
    hw = t.get("highway")
    closed = t.get("access") in ("no", "private") or t.get("motor_vehicle") == "no" \
        or t.get("motorcar") == "no"
    bus_open = t.get("bus") in ("yes", "designated") or t.get("psv") in ("yes", "designated")
    # Warsaw often maps separate bus roadways as highway=service + psv/bus=yes|designated without an
    # explicit access=no; on a service road that clearly means "for buses". (Not applied to main
    # roads, where psv=yes only restates the default.) Loops vs parallel lanes: see match_streets.
    bus_only = hw in ("busway", "bus_guideway") or (closed and bus_open) or (hw == "service" and bus_open)
    if bus_only:
        # the whole roadway is for buses — but only in the directions buses may drive
        contra = t.get("oneway:bus") == "no" or t.get("oneway:psv") == "no"
        fwd = fwd or not oneway_rev or contra
        bwd = bwd or not oneway or contra

    return fwd, bwd, bus_only


# ---------------------------------------------------------------------------
def build_layers(data: dict):
    streets, trams, boundary_lines = [], [], []
    for el in data["elements"]:
        if el["type"] == "way" and "geometry" in el:
            coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(coords) < 2:
                continue
            tags = el.get("tags", {})
            geom = LineString(coords)
            if tags.get("railway") == "tram":
                trams.append({"osm_id": el["id"], "name": tags.get("name"), "geometry": geom})
                continue
            fwd, bwd, only = bus_lane_flags(tags)
            oneway = tags.get("oneway") in ("yes", "1", "true") or tags.get("junction") == "roundabout" \
                or tags.get("highway") == "motorway"
            streets.append({
                "osm_id": el["id"],
                "name": tags.get("name") or tags.get("ref"),
                "highway": tags.get("highway"),
                "oneway": -1 if tags.get("oneway") == "-1" else int(oneway),
                "lanes": tags.get("lanes"),
                "bl_fwd": int(fwd),
                "bl_bwd": int(bwd),
                "bus_only": int(only),
                "geometry": geom,
            })
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer" and "geometry" in m:
                    boundary_lines.append(LineString([(p["lon"], p["lat"]) for p in m["geometry"]]))

    streets = gpd.GeoDataFrame(streets, crs="EPSG:4326").to_crs(POLAND_CRS)
    trams = gpd.GeoDataFrame(trams, crs="EPSG:4326").to_crs(POLAND_CRS)
    polys = list(polygonize(unary_union(boundary_lines)))
    boundary = gpd.GeoDataFrame({"name": ["Warszawa"]}, geometry=[unary_union(polys)],
                                crs="EPSG:4326").to_crs(POLAND_CRS)
    return streets, trams, boundary


def build_river(data: dict) -> gpd.GeoDataFrame:
    """River area polygons (Vistula) — for map orientation only."""
    rings = []
    for el in data["elements"]:
        if el["type"] == "way" and "geometry" in el:
            rings.append(LineString([(p["lon"], p["lat"]) for p in el["geometry"]]))
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer" and "geometry" in m:
                    rings.append(LineString([(p["lon"], p["lat"]) for p in m["geometry"]]))
    polys = list(polygonize(unary_union(rings)))
    return gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_crs(POLAND_CRS)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mode = "update" if "--update" in sys.argv else "refresh" if "--refresh" in sys.argv else "cache"
    data = fetch(DATA_DIR / "osm_raw.json", mode=mode)
    streets, trams, boundary = build_layers(data)
    out = DATA_DIR / "osm_streets.gpkg"
    streets.to_file(out, layer="streets", driver="GPKG")
    trams.to_file(out, layer="trams", driver="GPKG")
    boundary.to_file(out, layer="boundary", driver="GPKG")
    river = build_river(fetch(DATA_DIR / "osm_river.json", river_query()))
    river.to_file(out, layer="river", driver="GPKG")
    n_lane = ((streets.bl_fwd == 1) | (streets.bl_bwd == 1)).sum()
    km = streets[(streets.bl_fwd == 1) | (streets.bl_bwd == 1)]
    km_len = (km.length * (km.bl_fwd + km.bl_bwd)).sum() / 1000
    print(f"Streets: {len(streets):,} ways | with bus lane/priority: {n_lane:,} "
          f"(~{km_len:.0f} km of directional bus lane)")
    print(f"Tram ways: {len(trams):,} | boundary area: {boundary.area.iloc[0] / 1e6:.0f} km²")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
