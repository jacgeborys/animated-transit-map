"""
Tram delay map — no schedule needed.

For each pair of consecutive GPS fixes on the same vehicle:
  - Computes speed (m/s)
  - Projects the midpoint onto the nearest tram junction segment
  - Accumulates "time lost" = time spent below free-flow speed

Output: live/data/delay_segments_<date>.geojson
  Attributes per segment:
    n_obs         — number of GPS movement samples mapped here
    mean_speed    — average speed in km/h
    slow_sec      — cumulative seconds trams spent below FREE_FLOW_KMH
    slow_pct      — % of observed time that was slow (0–100)
    passes        — number of distinct tram passes through this segment

Usage: python live/delay_map.py
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Point
from collections import defaultdict

DATA_DIR   = Path(__file__).parent / "data"
SEGMENTS   = Path(__file__).parent.parent / "data" / "processed" / "tram_junction_segments.shp"

FREE_FLOW_KMH = 25.0          # trams doing >= this are considered unimpeded
FREE_FLOW_MPS = FREE_FLOW_KMH / 3.6
MAX_GAP_SEC   = 180            # skip pairs with gap > 3 min (parked / data hole)
MIN_GAP_SEC   = 8              # skip pairs too close together (GPS jitter)
MAX_SNAP_M    = 150            # discard if nearest segment is further than this


def load_movements(csv_path: Path) -> pd.DataFrame:
    """Return one row per consecutive GPS pair per vehicle, with speed computed."""
    df = pd.read_csv(csv_path, parse_dates=['poll_time'])
    df = df[df['vehicle_type'] == 'Tram'].sort_values(['VehicleNumber', 'poll_time'])

    rows = []
    for vid, vdf in df.groupby('VehicleNumber'):
        vdf = vdf.reset_index(drop=True)
        for i in range(len(vdf) - 1):
            r0, r1 = vdf.iloc[i], vdf.iloc[i + 1]
            dt = (r1['poll_time'] - r0['poll_time']).total_seconds()
            if dt < MIN_GAP_SEC or dt > MAX_GAP_SEC:
                continue

            # Haversine distance (metres)
            lat0, lon0 = np.radians(r0['Lat']), np.radians(r0['Lon'])
            lat1, lon1 = np.radians(r1['Lat']), np.radians(r1['Lon'])
            dlat, dlon = lat1 - lat0, lon1 - lon0
            a = np.sin(dlat/2)**2 + np.cos(lat0)*np.cos(lat1)*np.sin(dlon/2)**2
            dist_m = 6_371_000 * 2 * np.arcsin(np.sqrt(a))

            speed_mps = dist_m / dt
            if speed_mps > 30:          # > 108 km/h — bad GPS jump
                continue

            mid_lat = (r0['Lat'] + r1['Lat']) / 2
            mid_lon = (r0['Lon'] + r1['Lon']) / 2

            rows.append({
                'VehicleNumber': vid,
                'mid_lat': mid_lat,
                'mid_lon': mid_lon,
                'dt_sec':  dt,
                'dist_m':  dist_m,
                'speed_mps': speed_mps,
            })

    return pd.DataFrame(rows)


def main():
    csvs = sorted(DATA_DIR.glob('positions_*.csv'))
    if not csvs:
        print("No data in live/data/")
        return
    csv_path = csvs[-1]
    date_str = csv_path.stem.replace('positions_', '')
    print(f"Loading {csv_path.name}...")

    moves = load_movements(csv_path)
    print(f"  {len(moves)} valid GPS movement pairs from {moves['VehicleNumber'].nunique()} vehicles")
    print(f"  Speed range: {moves['speed_mps'].min()*3.6:.1f} – {moves['speed_mps'].max()*3.6:.1f} km/h")

    # Load junction segments
    print(f"Loading tram junction segments...")
    segs = gpd.read_file(SEGMENTS).to_crs('EPSG:2180')
    segs['seg_idx'] = segs.index
    print(f"  {len(segs)} segments")

    # Convert movement midpoints to GeoDataFrame (EPSG:2180)
    moves_gdf = gpd.GeoDataFrame(
        moves,
        geometry=gpd.points_from_xy(moves['mid_lon'], moves['mid_lat']),
        crs='EPSG:4326'
    ).to_crs('EPSG:2180')

    # Snap each midpoint to nearest junction segment
    print("Snapping GPS midpoints to nearest segment...")
    seg_geoms = segs.geometry.values
    seg_idx_arr = segs['seg_idx'].values

    # Build spatial index
    from shapely.strtree import STRtree
    tree = STRtree(seg_geoms)

    snapped_idx = []
    snap_dist   = []
    for pt in moves_gdf.geometry:
        nearest_i = tree.nearest(pt)
        dist = pt.distance(seg_geoms[nearest_i])
        snapped_idx.append(seg_idx_arr[nearest_i])
        snap_dist.append(dist)

    moves_gdf['snapped_seg'] = snapped_idx
    moves_gdf['snap_dist_m'] = snap_dist
    moves_gdf = moves_gdf[moves_gdf['snap_dist_m'] <= MAX_SNAP_M]
    print(f"  {len(moves_gdf)} pairs snapped within {MAX_SNAP_M}m ({len(moves)-len(moves_gdf)} discarded)")

    # Aggregate per segment
    print("Aggregating delay per segment...")
    stats = defaultdict(lambda: {
        'total_sec': 0.0,
        'slow_sec':  0.0,
        'speeds':    [],
        'passes':    set(),
    })

    for _, row in moves_gdf.iterrows():
        s = stats[row['snapped_seg']]
        s['total_sec'] += row['dt_sec']
        s['speeds'].append(row['speed_mps'] * 3.6)
        s['passes'].add(row['VehicleNumber'])
        if row['speed_mps'] < FREE_FLOW_MPS:
            expected_sec = row['dist_m'] / FREE_FLOW_MPS if row['dist_m'] > 0 else 0
            s['slow_sec'] += max(0, row['dt_sec'] - expected_sec)

    # Build output GeoDataFrame
    out_rows = []
    for seg_i, s in stats.items():
        seg_row = segs.loc[seg_i]
        mean_spd = np.mean(s['speeds'])
        slow_pct = 100 * s['slow_sec'] / s['total_sec'] if s['total_sec'] > 0 else 0
        out_rows.append({
            'seg_idx':    int(seg_i),
            'n_obs':      len(s['speeds']),
            'mean_kmh':   round(mean_spd, 1),
            'slow_sec':   round(s['slow_sec']),
            'slow_min':   round(s['slow_sec'] / 60, 1),
            'slow_pct':   round(slow_pct, 1),
            'passes':     len(s['passes']),
            'geometry':   seg_row.geometry,
        })

    out_gdf = gpd.GeoDataFrame(out_rows, crs=segs.crs).to_crs('EPSG:4326')
    out_gdf = out_gdf.sort_values('slow_sec', ascending=False)

    out_path = DATA_DIR / f"delay_segments_{date_str}.geojson"
    out_gdf.to_file(out_path, driver='GeoJSON')
    print(f"\nSaved: {out_path}")
    print(f"  {len(out_gdf)} segments with observations")

    print("\n=== Top 15 worst segments (most cumulative slow time) ===")
    print(f"{'seg':>6}  {'passes':>6}  {'n_obs':>5}  {'mean km/h':>9}  {'slow_min':>8}  {'slow_pct':>8}")
    for _, r in out_gdf.head(15).iterrows():
        print(f"{int(r['seg_idx']):>6}  {int(r['passes']):>6}  {int(r['n_obs']):>5}  "
              f"{r['mean_kmh']:>9.1f}  {r['slow_min']:>8.1f}  {r['slow_pct']:>7.1f}%")


if __name__ == "__main__":
    main()
