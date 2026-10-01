"""
Step A2 — Bus lanes from sources other than OSM.

1. City bus-lane layer (mapa.um.warszawa.pl, Oracle MapViewer theme K_BUSPASY_0_4).
   Lines are directional: vertex order = direction of travel; two-way lanes are two lines.
   NOTE: every record says "Aktualność: listopad 2021" — lanes opened later are missing.
2. bus_lanes/manual_bus_lanes.csv — newer lanes typed in from official announcements.
   Each row is routed along the named OSM street(s) between two endpoints:
       ulica:<OSM name>    junction with that street
       przystanek:<id>     GTFS ZTM stop
       granica             Warsaw city border
   kierunek: 'od-do' (one direction, od → do) or 'oba' (both directions).

Output: bus_lanes/_data/extra_lanes.gpkg, layer 'lanes' (EPSG:2180), one directional line per
lane with columns zrodlo ('miasto' | 'reczne'), ulica, odcinek, dni, godziny, otwarty, uwagi.
Edit/extend it in QGIS freely — match_streets.py only needs the directional lines.
"""
import heapq
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).parent))
from bl_config import BL_DIR, DATA_DIR, POLAND_CRS, RAW_DIR

CITY_URL = "https://mapa.um.warszawa.pl/mapviewer/dataserver/DANE_WAWA?t=K_BUSPASY_0_4"
MANUAL_CSV = BL_DIR / "manual_bus_lanes.csv"
# City-layer records to drop (street name contains any of these) — temporary lanes from the
# metro construction that no longer exist
CITY_EXCLUDE = ["Chodecka", "Wyszogrodzka"]
CITY_EXCLUDE_STATUS = ["metro"]   # all temporary metro-construction lanes are out of date
ENDPOINT_RADIUS_M = {"ulica": 30, "przystanek": 60, "granica": 80}
CONNECT_GAP_M = 25          # bridge small gaps between OSM pieces (junction areas)
CONTINUE_GAP_M = 100        # longer gaps bridged only when straight ahead (±30°)


def _bearing(a, b):
    return float(np.degrees(np.arctan2(b[0] - a[0], b[1] - a[1])) % 360)


def _angle(a, b):
    return abs((a - b + 180) % 360 - 180)


# ---------------------------------------------------------------------------
def fetch_city_layer(cache: Path) -> gpd.GeoDataFrame:
    if not cache.exists():
        print("Downloading city bus-lane layer ...")
        r = requests.get(CITY_URL, timeout=120, headers={"User-Agent": "warsaw-bus-lane-map/1.0"})
        r.raise_for_status()
        cache.write_text(r.text, encoding="utf-8")
    d = json.loads(cache.read_text(encoding="utf-8"))
    rows = []
    for f in d["features"]:
        if f["geometry"]["type"] != "LineString":
            continue
        p = f["properties"]
        if any(x in (p.get("NAZWA_SERWIS") or "") for x in CITY_EXCLUDE)                 or p.get("STATUS") in CITY_EXCLUDE_STATUS:
            continue
        # Oracle returns a flat [x1, y1, x2, y2, ...] list
        geom = LineString(np.asarray(f["geometry"]["coordinates"], float).reshape(-1, 2))
        rows.append({"zrodlo": "miasto", "ulica": p.get("NAZWA_SERWIS"), "odcinek": p.get("Odcinek"),
                     "dni": p.get("Dni"), "godziny": p.get("Godziny"), "otwarty": None,
                     "uwagi": f"status: {p.get('STATUS')}; aktualność: {p.get('Aktualność')}",
                     "geometry": geom})
    gdf = gpd.GeoDataFrame(rows, crs=f"EPSG:{d.get('srs', 2178)}").to_crs(POLAND_CRS)
    print(f"City layer: {len(gdf)} directional lines, {gdf.length.sum() / 1000:.1f} km")
    return gdf


# ---------------------------------------------------------------------------
class StreetGraph:
    """Directed graph over the vertices of the named OSM ways (oneway respected)."""

    def __init__(self, ways: gpd.GeoDataFrame):
        self.xy, self.adj = [], {}
        index = {}

        def node(p):
            k = (round(p[0], 2), round(p[1], 2))
            if k not in index:
                index[k] = len(self.xy)
                self.xy.append(p)
            return index[k]

        def edge(a, b, w):
            self.adj.setdefault(a, []).append((b, w))

        ends = []
        heads, tails = [], []  # (node, travel heading) where travel leaves / enters a way
        for geom, ow in zip(ways.geometry.values, ways.oneway.values):
            c = np.asarray(geom.coords)
            ids = [node(tuple(p)) for p in c]
            for a, b, pa, pb in zip(ids[:-1], ids[1:], c[:-1], c[1:]):
                w = float(np.hypot(*(pb - pa)))
                if ow != -1:
                    edge(a, b, w)
                if ow != 1:
                    edge(b, a, w)
            ends += [ids[0], ids[-1]]
            for cc, ii, ok in ((c, ids, ow != -1), (c[::-1], ids[::-1], ow != 1)):
                if ok and len(cc) > 1:
                    tails.append((ii[-1], _bearing(cc[-2], cc[-1])))
                    heads.append((ii[0], _bearing(cc[0], cc[1])))
        self.xy = np.asarray(self.xy)
        # junction areas are often separate, differently named ways → bridge short gaps
        tree = shapely.STRtree(shapely.points(self.xy))
        for e in set(ends):
            for j in tree.query(shapely.Point(self.xy[e]), predicate="dwithin", distance=CONNECT_GAP_M):
                if j != e and not any(n == j for n, _ in self.adj.get(e, [])):
                    w = float(np.hypot(*(self.xy[j] - self.xy[e]))) + 5  # small penalty
                    edge(e, j, w)
                    edge(j, e, w)
        # longer gaps (viaducts, junctions mapped as other ways): only straight-ahead continuations
        for t, ht in tails:
            for h, hh in heads:
                if h == t:
                    continue
                v = self.xy[h] - self.xy[t]
                d = float(np.hypot(*v))
                if CONNECT_GAP_M < d <= CONTINUE_GAP_M:
                    gap_dir = _bearing(self.xy[t], self.xy[h])
                    if _angle(gap_dir, ht) < 30 and _angle(hh, ht) < 30:
                        edge(t, h, d + 10)

    def near(self, geom, radius):
        d = shapely.distance(shapely.points(self.xy), geom)
        return set(np.flatnonzero(d <= radius).tolist())

    def shortest(self, sources, targets):
        dist = {s: 0.0 for s in sources}
        prev = {}
        pq = [(0.0, s) for s in sources]
        heapq.heapify(pq)
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf):
                continue
            if u in targets:
                path = [u]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                return LineString(self.xy[path[::-1]]) if len(path) > 1 else None
            for v, w in self.adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
        return None


def endpoint_geom(spec: str, streets, stops, boundary):
    kind, _, val = spec.partition(":")
    if kind == "ulica":
        g = streets[streets.name == val]
        if g.empty:
            raise ValueError(f"no OSM street named {val!r}")
        return g.geometry.union_all(), ENDPOINT_RADIUS_M["ulica"]
    if kind == "przystanek":
        s = stops[stops.stop_id == val]
        if s.empty:
            raise ValueError(f"no GTFS stop {val!r}")
        return s.geometry.iloc[0], ENDPOINT_RADIUS_M["przystanek"]
    if kind == "granica":
        return boundary.boundary, ENDPOINT_RADIUS_M["granica"]
    raise ValueError(f"unknown endpoint {spec!r}")


def build_manual(streets, stops, boundary) -> gpd.GeoDataFrame:
    rows = []
    for r in pd.read_csv(MANUAL_CSV, dtype=str).fillna("").itertuples():
        names = [n.strip() for n in r.ulice.split(";")]
        ways = streets[streets.name.isin(names)]
        if ways.empty:
            print(f"  [{r.id}] no OSM ways for {names}")
            continue
        g = StreetGraph(ways)
        a_geom, a_rad = endpoint_geom(r.od, streets, stops, boundary)
        b_geom, b_rad = endpoint_geom(r.do, streets, stops, boundary)
        A, B = g.near(a_geom, a_rad), g.near(b_geom, b_rad)
        pairs = [(A, B, f"{r.od} → {r.do}")]
        if r.kierunek == "oba":
            pairs.append((B, A, f"{r.do} → {r.od}"))
        for src, dst, label in pairs:
            line = g.shortest(src, dst) if src and dst else None
            if line is None:
                print(f"  [{r.id}] NO PATH {label}  (start nodes {len(src)}, end nodes {len(dst)})")
                continue
            a, b = line.coords[0], line.coords[-1]
            hd = np.degrees(np.arctan2(b[0] - a[0], b[1] - a[1])) % 360
            print(f"  [{r.id}] {label}: {line.length / 1000:.2f} km, heading {hd:.0f}°")
            rows.append({"zrodlo": "reczne", "ulica": " / ".join(names[:2]), "odcinek": label,
                         "dni": r.dni or None, "godziny": r.godziny or None, "otwarty": r.otwarty,
                         "uwagi": r.uwagi or None, "id": r.id, "geometry": line})
    return gpd.GeoDataFrame(rows, crs=POLAND_CRS)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    city = fetch_city_layer(DATA_DIR / "k_buspasy.json")

    osm = DATA_DIR / "osm_streets.gpkg"
    streets = gpd.read_file(osm, layer="streets")
    boundary = gpd.read_file(osm, layer="boundary").geometry.iloc[0]
    feed = sorted(d for d in RAW_DIR.iterdir() if d.name.startswith("warsaw_gtfs_") and "_with_" not in d.name)[-1]
    st = pd.read_csv(feed / "stops.txt", dtype={"stop_id": str})
    stops = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.stop_lon, st.stop_lat),
                             crs="EPSG:4326").to_crs(POLAND_CRS)

    print("Manual lanes:")
    manual = build_manual(streets, stops, boundary)
    print(f"Manual layer: {len(manual)} directional lines, {manual.length.sum() / 1000:.1f} km")

    lanes = pd.concat([city, manual], ignore_index=True)
    out = DATA_DIR / "extra_lanes.gpkg"
    gpd.GeoDataFrame(lanes, crs=POLAND_CRS).to_file(out, layer="lanes", driver="GPKG")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
