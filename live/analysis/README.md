# Tram time-loss analysis (`live/analysis/`)

Live GPS (`live/data/positions_<date>.csv`) tied trip-by-trip to the GTFS timetable of the same day.
Answers: where and **in which direction** trams stand still, whether a corridor has working signal priority,
how far reality is from the timetable, which tram models ran, and what working priority would save
(tram-hours, passenger-hours, trams freed in the peak).

Run with **Python 3.11** (`py -3.11`; the default `python` on this machine is 3.8 with an old pandas). Order:

| # | Script | Output | Time |
|---|--------|--------|------|
| 0 | edit `ta_config.py` → `DATE` | | |
| 1 | `match_trips.py` | `_cache/fixes,intervals,passages,trips_obs_<date>.parquet` (+ GTFS/fix caches) | ~3 min |
| 2 | `places.py` | `_output/places_<date>.gpkg/.csv`: stand-still places per travel heading | ~2 min |
| 3 | `fetch_signals.py` (once) → `signals.py` | `_cache/signals.json`; `_output/signals_<date>.csv/.gpkg`: stop rate per line/dir/signal | ~3 min |
| 4 | `priority_scenarios.py` | `_output/priority_lines_<date>.csv`, `priority_approaches_<date>.csv` | <1 min |
| 5 | `timetable.py` | `_output/tt_segments/headways/models/delay_profile_<date>.csv` | ~1 min |
| 6 | `report.py` | `_output/tram_report_<date>.html` (standalone page from `report_template.html`) | <1 min |

## Method in one paragraph
Live data carries line + brigade; GTFS `block_short_name` is the brigade. Each fix is projected onto every trip of that
brigade scheduled around that time (on the trip's own shape) and assigned to the trip with the most consistent lateness —
that separates the two directions on a shared street (97 % of fixes, 6,914 / 7,139 trips matched on 24.11.2026).
Intervals between fixes are classified: moving / standing at a platform (−50 m…+25 m) / standing elsewhere / terminus
(first & last 150 m); stand-stills > 8 min are incidents. **Stop rate** at a signal approach (OSM signal nodes ≤ 18 m from
the route, 90 m approach) = share of passing trams that stood still there. Fixed-time signal ≈ red share (40–70 %);
working priority ≈ 5–20 %. Approaches with a platform inside hide the red phase in the dwell → flagged, lower bound.

## Fleet
`ta_config.FLEET`: fleet-number ranges → type, places, length (tramwar.pl / pl.wikipedia, read 26.11.2026). 105N-family
and 123N run as 2-car sets but report one car number → 2 × 97 places. tramwar.pl mixes capacity norms (±15 %).
GTFS `fleet_type` is the *planned* type; actual type comes from the vehicle number.

## Assumptions (only affect step 4 + report)
`LOAD_FACTOR` (50 % peak / 25 % off-peak — not measured), `TARGET_STOP_RATE` 10 % (level of the good Wilanów signals),
`ACCEL_PENALTY_S` 8 s per avoided stop, `VALUE_OF_TIME_PLN_H` 25. Passenger-hours scale linearly with the load factor.

## Findings, Tue 24.11.2026 (04:27–23:12)
- 5,050 tram-hours in service: 66 % moving, 24 % at platforms, **9 % (467 h) standing elsewhere**; share 8 % off-peak → 10 % peak.
- Worst places: Rakowiecka × Puławska (N), Grójecka × Banacha (both dirs), Targowa × Cyryla i Metodego, Okopowa × Solidarności,
  Rondo ONZ, Rondo Czterdziestolatka, Pl. Zawiszy, Goworka × Puławska (N), Marszałkowska (Hoża → Pl. Unii).
- One-direction-only junctions: Al. Solidarności (Wola-Ratusz) 95 % vs 1 %, Puławska at Niedźwiedzia/Dworkowa N 91 % vs S 6 %,
  Banacha × Grójecka, Grochowska, Jagiellońska, Marynarska × Postępu.
- Wilanów (Goworka – Miasteczko Wilanów): priority works on most of Sobieskiego / Al. Rzeczypospolitej (2–12 % stop rate);
  bad: Goworka × Puławska northbound (~80 %, 40+ s/tram), Belwederska × Spacerowa/Gagarina (55–60 % both ways),
  Św. Bonifacego (43–58 %), Witosa (20–25 %).
- Scenario (all signals → 10 %): 192 tram-h/day, ~13.6k passenger-h/day, 13.8 trams of peak cycle time
  (≈ 1 each on lines 17, 1, 16, 9; 3–4 whole brigades). Many lines already run 1–7 min/round trip behind the timetable in the peak.
- Timetable too tight: Dworkowa → Goworka (4, 10, 18: 2 min planned, 4.3 real), Muranowska → Muranów (15, 18: 1 → 3 min).
- Mean excess waiting time from irregular headways +0.4 min; worst 19, 23, 20.
- Tram model barely matters for dwell (all within ~2 s of the stop median). 232 of 1,825 trips planned for 105N sets ran with low-floor trams.

Report: https://claude.ai/artifact/Wrj38hkTvnX8opgUAxGT7w (private artifact) — standalone copy `_output/tram_report_<date>.html`.

## Next
Buses through the same pipeline; a full week incl. Saturday; before/after around a junction retune; ZTM stop-level
boardings instead of the load-factor assumption.
