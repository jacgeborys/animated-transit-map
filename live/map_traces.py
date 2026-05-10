"""
Static map of GPS traces from collected live positions.

Shows each vehicle's recorded path — the scatter/wobble around tram rails
is a direct visual of GPS error (typically 5–20m for Warsaw trams).

Output:
  live/data/traces_<date>.png               — map image
  live/data/traces_points_<date>.geojson    — every GPS fix as a point
  live/data/traces_lines_<date>.geojson     — one LineString per vehicle

Usage:  python live/map_traces.py
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from shapely.geometry import LineString, box as shapely_box

DATA_DIR  = Path(__file__).parent / "data"
OSM_DIR   = Path(r"D:\QGIS\mapy_warszawy_misc\data\osm")

BACKGROUND_LAYERS = {
    'water': OSM_DIR / 'water.gpkg',
    'roads': OSM_DIR / 'roads.shp',
}

RADIUS_DEG = 0.12   # ~10 km radius from data centroid


def load_trams(csv_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    df = df[df['vehicle_type'] == 'Tram'].copy()
    df['poll_time'] = pd.to_datetime(df['poll_time'])
    df = df.sort_values(['VehicleNumber', 'poll_time'])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['Lon'], df['Lat']),
        crs='EPSG:4326'
    ).to_crs('EPSG:2180')
    return gdf


def load_layer(path, clip_box):
    lyr = gpd.read_file(path).to_crs('EPSG:2180')
    lyr['geometry'] = lyr.geometry.make_valid()
    lyr = lyr[lyr.geometry.is_valid & ~lyr.geometry.is_empty]
    return gpd.clip(lyr, clip_box)


def main():
    csvs = sorted(DATA_DIR.glob('positions_*.csv'))
    if not csvs:
        print("No data found in live/data/")
        return
    csv_path = csvs[-1]
    print(f"Loading {csv_path.name}...")

    gdf = load_trams(csv_path)
    date_str = csv_path.stem.replace('positions_', '')
    print(f"  {len(gdf)} tram positions, {gdf['VehicleNumber'].nunique()} vehicles, "
          f"{gdf['Lines'].nunique()} lines")

    # --- Export GeoJSON for QGIS ---
    # Points: every raw GPS fix
    points_out = DATA_DIR / f"traces_points_{date_str}.geojson"
    gdf[['poll_time', 'vehicle_type', 'Lines', 'Brigade', 'VehicleNumber', 'gps_time', 'geometry']] \
        .to_crs('EPSG:4326').to_file(points_out, driver='GeoJSON')
    print(f"Saved: {points_out}")

    # Lines: one LineString per vehicle
    trace_rows = []
    for vehicle_id, vdf in gdf.groupby('VehicleNumber'):
        vdf = vdf.sort_values('poll_time')
        if len(vdf) < 2:
            continue
        coords = list(zip(vdf.geometry.x, vdf.geometry.y))
        trace_rows.append({
            'VehicleNumber': vehicle_id,
            'Lines':         vdf['Lines'].iloc[0],
            'vehicle_type':  vdf['vehicle_type'].iloc[0],
            'n_fixes':       len(vdf),
            'geometry':      LineString(coords),
        })
    traces_gdf = gpd.GeoDataFrame(trace_rows, crs='EPSG:2180').to_crs('EPSG:4326')
    traces_out = DATA_DIR / f"traces_lines_{date_str}.geojson"
    traces_gdf.to_file(traces_out, driver='GeoJSON')
    print(f"Saved: {traces_out}")

    # --- Static map ---
    cx, cy = gdf.geometry.x.mean(), gdf.geometry.y.mean()
    r_m = RADIUS_DEG * 100_000
    xmin, xmax = cx - r_m, cx + r_m
    ymin, ymax = cy - r_m, cy + r_m

    fig, ax = plt.subplots(figsize=(14, 14), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    clip_box = gpd.GeoDataFrame(geometry=[shapely_box(xmin, ymin, xmax, ymax)], crs='EPSG:2180')

    if BACKGROUND_LAYERS['water'].exists():
        water = load_layer(BACKGROUND_LAYERS['water'], clip_box)
        if not water.empty:
            water.plot(ax=ax, fc='#0f2e45', ec='none', zorder=1)

    if BACKGROUND_LAYERS['roads'].exists():
        roads = load_layer(BACKGROUND_LAYERS['roads'], clip_box)
        if not roads.empty:
            roads.plot(ax=ax, fc='none', ec='#333333', lw=0.3, zorder=2)

    lines = sorted(gdf['Lines'].unique())
    cmap = plt.colormaps.get_cmap('tab20')
    line_colors = {line: cmap(i % 20) for i, line in enumerate(lines)}

    for vehicle_id, vdf in gdf.groupby('VehicleNumber'):
        vdf = vdf.sort_values('poll_time')
        xs = vdf.geometry.x.values
        ys = vdf.geometry.y.values
        color = line_colors[vdf['Lines'].iloc[0]]
        ax.plot(xs, ys, color=color, lw=0.6, alpha=0.5, zorder=3)
        ax.scatter(xs, ys, color=color, s=3, alpha=0.6, zorder=4, linewidths=0)

    top_lines = gdf.groupby('Lines')['VehicleNumber'].nunique().sort_values(ascending=False).head(15).index
    handles = [plt.Line2D([0], [0], color=line_colors[l], lw=2, label=f"Tram {l}") for l in top_lines]
    ax.legend(handles=handles, loc='lower right', fontsize=7, framealpha=0.3,
              facecolor='#111111', edgecolor='none', labelcolor='white', ncol=2)

    time_range = f"{gdf['poll_time'].min().strftime('%H:%M')} – {gdf['poll_time'].max().strftime('%H:%M')}"
    ax.set_title(f"Warsaw Tram GPS Traces  ·  {date_str}  ·  {time_range}",
                 color='white', fontsize=13, pad=8)

    png_out = DATA_DIR / f"traces_{date_str}.png"
    fig.savefig(png_out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print(f"Saved: {png_out}")
    plt.show()


if __name__ == "__main__":
    main()
