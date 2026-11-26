"""
Download traffic-signal nodes (OSM) for Warsaw once -> _cache/signals.json.
Used to tell signal-controlled stand-stills from queues / other causes. Signals change slowly, so a cached copy
is fine; delete the file to refresh.

Usage: python live/analysis/fetch_signals.py
"""

import json

import requests

import ta_config as C

MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter"]
BBOX = "52.09,20.85,52.37,21.28"   # S, W, N, E
Q = f"""[out:json][timeout:180];
(
  node["highway"="traffic_signals"]({BBOX});
  node["crossing"="traffic_signals"]({BBOX});
  node["railway"="crossing"]["crossing:light"!="no"]({BBOX});
  node["railway"="tram_crossing"]({BBOX});
  node["railway"="tram_level_crossing"]({BBOX});
);
out body;"""


def main():
    f = C.CACHE / "signals.json"
    C.CACHE.mkdir(exist_ok=True)
    for url in MIRRORS:
        try:
            r = requests.post(url, data={"data": Q}, timeout=300, headers={"User-Agent": "transit-analysis-script"})
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f"{url}: {e}")
    else:
        raise SystemExit("all Overpass mirrors failed")
    print(f"{len(data['elements'])} nodes, OSM data as of {data.get('osm3s', {}).get('timestamp_osm_base')}")
    f.write_text(json.dumps(data), encoding="utf-8")
    print(f"Saved {f}")


if __name__ == "__main__":
    main()
