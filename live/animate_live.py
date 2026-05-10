"""
Animation of live GPS positions collected by collect.py.

Much simpler than animate_full_density.py — no GTFS schedule, no route snapping.
Vehicles move between their actual recorded GPS fixes, interpolated smoothly.

Output: live/data/live_animation_<date>.mp4
Usage:  python live/animate_live.py
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR   = Path(__file__).parent / "data"
OSM_DIR    = Path(r"D:\QGIS\mapy_warszawy_misc\data\osm")
FFMPEG     = r"C:\ffmpeg\bin\ffmpeg.exe"

FPS        = 20
DURATION   = 60      # seconds of output video
TRAIL_SECS = 120     # how many seconds of trail to show behind each vehicle

TRAM_COLOR  = '#FF7075'
BUS_COLOR   = '#B46EFC'


def load_data(csv_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path, parse_dates=['poll_time', 'gps_time'])
    df = df.sort_values(['VehicleNumber', 'poll_time'])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['Lon'], df['Lat']),
        crs='EPSG:4326'
    ).to_crs('EPSG:2180')
    gdf['x'] = gdf.geometry.x
    gdf['y'] = gdf.geometry.y
    return gdf


def build_vehicle_timelines(gdf: gpd.GeoDataFrame) -> dict:
    """For each vehicle: sorted list of (timestamp, x, y)."""
    timelines = {}
    for vid, vdf in gdf.groupby('VehicleNumber'):
        vdf = vdf.sort_values('poll_time').drop_duplicates('poll_time')
        timelines[vid] = {
            'times': vdf['poll_time'].values,  # numpy datetime64
            'xs':    vdf['x'].values,
            'ys':    vdf['y'].values,
            'vtype': vdf['vehicle_type'].iloc[0],
            'line':  str(vdf['Lines'].iloc[0]),
        }
    return timelines


def interpolate_position(timeline: dict, t: pd.Timestamp):
    """Linear interpolation of x,y for vehicle at time t."""
    times = timeline['times']
    t64 = np.datetime64(t)
    if t64 < times[0] or t64 > times[-1]:
        return None
    idx = np.searchsorted(times, t64)
    if idx == 0:
        return timeline['xs'][0], timeline['ys'][0]
    if idx >= len(times):
        return timeline['xs'][-1], timeline['ys'][-1]
    t0, t1 = times[idx - 1], times[idx]
    span = (t1 - t0) / np.timedelta64(1, 's')
    if span <= 0 or span > 120:   # gap > 2 min = vehicle was off, skip
        return None
    frac = (t64 - t0) / np.timedelta64(1, 's') / span
    x = timeline['xs'][idx - 1] + frac * (timeline['xs'][idx] - timeline['xs'][idx - 1])
    y = timeline['ys'][idx - 1] + frac * (timeline['ys'][idx] - timeline['ys'][idx - 1])
    return x, y


def main():
    csvs = sorted(DATA_DIR.glob('positions_*.csv'))
    if not csvs:
        print("No data found in live/data/")
        return
    csv_path = csvs[-1]
    print(f"Loading {csv_path.name}...")

    gdf = load_data(csv_path)
    timelines = build_vehicle_timelines(gdf)

    t_start = gdf['poll_time'].min()
    t_end   = gdf['poll_time'].max()
    real_duration = (t_end - t_start).total_seconds()
    print(f"  Data: {t_start.strftime('%H:%M')} → {t_end.strftime('%H:%M')} "
          f"({real_duration/60:.0f} min), {len(timelines)} vehicles")

    # Map bounds
    cx, cy = gdf['x'].mean(), gdf['y'].mean()
    spread_x = (gdf['x'].max() - gdf['x'].min()) * 0.55
    spread_y = (gdf['y'].max() - gdf['y'].min()) * 0.55
    r = max(spread_x, spread_y, 5000)
    xmin, xmax = cx - r, cx + r
    ymin, ymax = cy - r, cy + r

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=0.96, bottom=0)

    # Background
    from shapely.geometry import box as shapely_box
    clip_box = gpd.GeoDataFrame(geometry=[shapely_box(xmin, ymin, xmax, ymax)], crs='EPSG:2180')

    for layer_path, fc, ec, lw, zo in [
        (OSM_DIR / 'water.gpkg',  '#0f2e45', 'none',    0,   1),
        (OSM_DIR / 'roads.shp',   'none',    '#2a2a2a', 0.3, 2),
    ]:
        if layer_path.exists():
            lyr = gpd.read_file(layer_path).to_crs('EPSG:2180')
            lyr['geometry'] = lyr.geometry.make_valid()
            lyr = lyr[lyr.geometry.is_valid & ~lyr.geometry.is_empty]
            lyr = gpd.clip(lyr, clip_box)
            if not lyr.empty:
                lyr.plot(ax=ax, fc=fc, ec=ec, lw=lw, zorder=zo)

    # Dynamic artists
    tram_sc  = ax.scatter([], [], s=18, c=TRAM_COLOR,  marker='D', zorder=10,
                          edgecolors='#730011', linewidths=1.2, alpha=0.9)
    bus_sc   = ax.scatter([], [], s=14, c=BUS_COLOR,   marker='o', zorder=9,
                          edgecolors='#430073', linewidths=1.2, alpha=0.9)
    trail_lc = LineCollection([], colors=TRAM_COLOR, linewidths=0.8, alpha=0.3, zorder=8)
    ax.add_collection(trail_lc)

    time_text  = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=18,
                         color='white', va='top', fontweight='bold', fontfamily='Segoe UI')
    count_text = ax.text(0.02, 0.93, '', transform=ax.transAxes, fontsize=12,
                         color='#aaaaaa', va='top', fontfamily='Segoe UI')

    total_frames = FPS * DURATION

    def update(frame):
        # Map frame → real data time
        frac = frame / total_frames
        t_cur = t_start + timedelta(seconds=frac * real_duration)
        t_trail = t_cur - timedelta(seconds=TRAIL_SECS)

        tram_pts, bus_pts, trails = [], [], []

        for vid, tl in timelines.items():
            pos = interpolate_position(tl, t_cur)
            if pos is None:
                continue
            x, y = pos
            if tl['vtype'] == 'Tram':
                tram_pts.append((x, y))
            else:
                bus_pts.append((x, y))

            # Trail: collect recent positions
            mask = (tl['times'] >= np.datetime64(t_trail)) & (tl['times'] <= np.datetime64(t_cur))
            if mask.sum() > 1:
                trail_x = tl['xs'][mask]
                trail_y = tl['ys'][mask]
                trails.append(np.column_stack([trail_x, trail_y]))

        tram_sc.set_offsets(np.array(tram_pts) if tram_pts else np.empty((0, 2)))
        bus_sc.set_offsets(np.array(bus_pts)   if bus_pts  else np.empty((0, 2)))
        trail_lc.set_segments(trails)

        time_text.set_text(t_cur.strftime('%H:%M:%S'))
        count_text.set_text(f"{len(tram_pts)} trams  ·  {len(bus_pts)} buses")

        if frame % 40 == 0:
            print(f"  Frame {frame}/{total_frames}  {t_cur.strftime('%H:%M:%S')}  "
                  f"trams={len(tram_pts)} buses={len(bus_pts)}")

    out = DATA_DIR / f"live_animation_{csv_path.stem.replace('positions_', '')}.mp4"
    out.parent.mkdir(exist_ok=True)

    plt.rcParams['animation.ffmpeg_path'] = FFMPEG
    writer = animation.FFMpegWriter(fps=FPS, codec='libx264', bitrate=3000,
                                    extra_args=['-pix_fmt', 'yuv420p'])

    print(f"Rendering {total_frames} frames → {out.name}...")
    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/FPS)
    try:
        anim.save(str(out), writer=writer)
        print(f"Saved: {out}")
    except KeyboardInterrupt:
        print("Interrupted.")
    plt.close()


if __name__ == "__main__":
    main()
