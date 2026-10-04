# Warsaw bus lane maps (`bus_lanes/`)

Where do many buses run **without** a bus lane? Built for a Miasto Jest Nasze post (Oct 2026), then grew into
a community effort to fix bus lanes in OpenStreetMap. Independent of the animation pipeline: it only needs a
ZTM GTFS download (`core/gtfs_downloader.py`).

## Run order

```bash
python core/gtfs_downloader.py           # fresh ZTM GTFS (only when the feed is stale)
python bus_lanes/osm_fetch.py --update   # OSM: re-fetch only bus/psv-tagged ways, merge into the cache
                                         #   after a known edit: --ways 1421092757[,id...] (one tiny request)
python bus_lanes/official_lanes.py       # city layer + manual_bus_lanes.csv → _data/extra_lanes.gpkg
python bus_lanes/transit_counts.py       # random regular Wednesday → per-shape passing counts
python bus_lanes/match_streets.py        # shapes → OSM ways per direction, lanes, classification, ranking
python bus_lanes/make_map.py             # social-media maps (dark, 4:5)
python bus_lanes/export_osm_todo.py      # OSM editing helper (standalone HTML) + iD overlay GeoJSON
python bus_lanes/make_osm_status_png.py  # "bus lanes in OSM" status PNG for the community
python bus_lanes/make_final_lanes_png.py # final map: counted lanes (blue) + dropped short pieces (red)
```

`transit_counts.py` only needs re-running for a new date/feed. After OSM edits, the refresh is:
`osm_fetch.py --update` → `official_lanes.py` → `match_streets.py` → `make_map.py`, `export_osm_todo.py`,
`make_osm_status_png.py`. Retired (kept, not in the routine): `make_review_png.py` (verification PNGs, replaced by
the status PNG) and `export_review.py` (claude.ai review page — colleagues could not log in).

| Output (`_output/`) | What |
|---|---|
| `buspasy_warszawa.png`, `buspasy_centrum.png`, `buspasy_ranking.png` | the MJN post (city, centre, top streets) |
| `bus_lane_gaps_ranking.csv` | streets without lanes, ranked by bus-km/h |
| `bus_lane_need.gpkg` | `bus_dir` (per way-direction), `tram_dir`, `boundary`, `osm_parallel_busways` — for QGIS |
| `buspasy_mapa.png` | final bus-lane map: 76 km counted (blue), short bays/terminus bits dropped (red) |
| `buspasy_osm_stan.png` | **share this**: lanes in OSM (blue) vs still missing (orange), progress since 1 Oct |
| `osm/buspasy_osm.html`, `osm/buspasy_osm_do_dodania.geojson` | **share this**: OSM editing helper + overlay for iD |

## How the analysis works

- **Counts** use the time a bus *passes each stop*, not trip start. `per_h` = max(AM 7–9, PM 15–17) / 2, per
  direction. Bus = GTFS `route_type 3` (catches N/L/E/Z lines too; the 3-digit route-id rule does not).
- **Matching**: each shape is sampled every 10 m and snapped to the nearest OSM way within 15 m and ±35° whose
  `oneway` allows that direction. That one rule separates dual carriageways. Counts are summed per (way, direction).
- **Bus lane sources**: since the community OSM verification (4 Oct 2026) only OSM counts
  (`LANE_SOURCES = ("osm",)` in `bl_config.py`). City layer and manual list are still computed as `bl_city` /
  `bl_manual` for reference. Before that, a lane from any of the three counted (that gave 39 km instead of 26 km).
- **Short stretches don't count**: connected lane stretches per direction under `MIN_LANE_STRETCH_M` (80 m) are
  bus-stop bays / terminus bits (`lane_removed=1`). Stretches join across untagged junction areas (≤40 m, straight
  on); separate parallel bus roadways must themselves form ≥80 m to give the carriageway a lane.
- **Status**: `lane` / `high` (≥30/h, no lane) / `medium` (15–30/h) / `low`. Thresholds in `bl_config.py`.
- **Headline** (4 Oct 2026, OSM-only lanes): 120 km of street-direction carry ≥30 buses/h; 26 km (22%) have a lane. Km are counted
  per direction, the same way the city counts its "~70–80 km of bus lanes". Always say "of the busiest streets",
  never "Warsaw has only 39 km of bus lanes" — that contradicts the city's total and invites a rebuttal.

## Lessons learnt

### OSM tagging of bus lanes in Warsaw
- On-carriageway lanes: mostly `lanes:psv` + `psv:lanes` (also `bus:lanes`, `:forward`/`:backward` variants,
  `busway=lane`, `lanes:bus`). Parsed in `osm_fetch.bus_lane_flags`.
- **Separate parallel bus roadways** are very common: `highway=service` + `oneway=yes` + `access=no` +
  `psv=designated|yes` / `bus=yes`, often **without** `access=no`. Any `highway=service` with psv/bus
  yes|designated is bus-only. Such a way running alongside a street (≤35 m, ≤30°, ≥50% of its length) counts as
  that carriageway's lane (`bl_osm_par`); a bus-only service way not parallel to any street is a loop/depot
  (`loop=1`) and is excluded from stats and drawing. Do not drop all bus-only service roads; that removed
  Trasa AK, Sobieskiego, Polski Walczącej, Plac Bankowy.
- **Bus-only roads must respect `oneway`** (exception: `oneway:bus|psv=no`). Without it, 574 one-way bus
  roadways got a phantom reverse lane.
- **Shared tram-bus roadways** (e.g. way 1144662304: `highway=service`, `access=no`, `bus=yes`, `lanes=2`, tram
  tracks along it) are genuinely two-way, so don't flag them as "missing oneway".
- `psv=yes` on a *main* road (Solidarności, Toruńska, Jagiellońska) says nothing; it is on ordinary carriageways.
- The shared tram-bus track on al. Solidarności is not tagged as usable by buses anywhere in OSM (Oct 2026).

### Data sources
- **No authoritative, current bus-lane map exists.** The city layer `K_BUSPASY_0_4` (Oracle MapViewer, not WMS:
  `https://mapa.um.warszawa.pl/mapviewer/dataserver/DANE_WAWA?t=K_BUSPASY_0_4`) returns GeoJSON in EPSG:2178
  with a *flat* coordinate list; lines are directional (vertex order = travel direction, two-way = two lines)
  and carry days/hours, but every record says "Aktualność: listopad 2021". All 17 `STATUS=metro` lanes are out of
  date (dropped via `CITY_EXCLUDE_STATUS`); Chodecka/Wyszogrodzka too.
- Post-2021 lanes come from warszawa19115.pl announcements (`manual_bus_lanes.csv`, each row quotes its source).
  Its "Alkuzyjna" is a typo for Aluzyjna. Rows marked DO WERYFIKACJI have no direction in the source.
- **Licence**: the city layer and announcements are hints only. Never trace them into OSM; verify on the ground
  or on imagery OSM may use (Geoportal orthophoto, Mapillary, Panoramax).
- **Overpass**: some mirrors lag months (kumi.systems served May data in October), so `osm_fetch.py` rejects data
  older than 3 days. The main server rate-limits (429) and times out (504) in the evening. Use `--update`: one
  index-friendly query for bus/psv-tagged roads (a clause per key; a key *regex* or a `changed` history search over
  the whole city is too heavy) plus a by-id refetch of ways that lost their tags; non-road ways are dropped. Never delete `_data/osm_raw.json` to refresh: it is the update baseline,
  and a full `--refresh` (9 resumable tiles) can take 40 min when Overpass is busy.

### Cartography (make_map.py)
- Draw each direction offset to the right of travel by half its *drawn* width (pt → m), or both directions sit on
  top of each other and one-sided lanes are invisible.
- Bus loops/depots as round-capped lines become blobs; tiny pieces (<25 m) get butt caps to avoid beading.
- Street labels: curved along a binned-median centreline of the same-named OSM ways (lands between carriageways),
  letters as squeezed outlines (82% width, Liberation Sans Narrow, +0.13 em tracking, thin 1.8 pt round-join halo;
  miter joins spike). Placed in ranking order with collision avoidance; per-street tweaks in `LABEL_SIDE`,
  `LABEL_AT`, `LABEL_NUDGE` (city map only), `LABEL_PRIORITY`, `CITY_EXTRA_LABELS`, `CENTRUM_EXTRA_LABELS`.
- Palette checked for colour-blind separation; lane blue brightened to #4c9bff for Instagram compression.

### Sharing with colleagues (MJN)
- claude.ai artifacts need a login, and saving to their db needs Contributor/Editor access inside the owner's
  organisation; outside colleagues could only view. Nobody but the owner left a review.
- uMap (no-login editing) was "too complicated". **What worked: a plain PNG** they describe in words, and for the
  OSM community a **standalone HTML** with an own vector basemap (OSM tiles are blocked for local/embedded use
  by the tile usage policy).
- A PNG can be open in an image viewer, which locks it on Windows (`OSError: [Errno 22]` on save). Close the viewer
  and re-run.

## Adding or removing a lane by hand

- Missing lane: add a row to `manual_bus_lanes.csv` with endpoints `ulica:<OSM name>`, `przystanek:<GTFS stop_id>`
  or `granica`, `kierunek` = `od-do` or `oba`, and the source; then `official_lanes.py` → `match_streets.py`.
- Wrong city-layer lane: add the street to `CITY_EXCLUDE` in `official_lanes.py`.
- Wrong OSM lane: fix it in OSM (the helper's pink "do sprawdzenia" layer), then `osm_fetch.py --update`.
- Reviewer verdicts from the review page db: `REPORTED_MISSING` in `export_osm_todo.py`.
