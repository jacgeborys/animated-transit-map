"""
Static map of GPS traces from collected live positions.

Shows each vehicle's recorded path — the scatter/wobble around tram rails
is a direct visual of GPS error (typically 5–20m for Warsaw trams).

Output: live/data/traces_<date>.png
Usage:  python live/map_traces.py
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from shapely.geometry import Point

DATA_DIR  = Path(__file__).parent / "data"
OSM_DIR   = Path(r"D:\QGIS\mapy_warszawy_misc\data\osm")

BACKGROUND_LAYERS = {
    'water': OSM_DIR / 'water.gpkg',
    'roads': OSM_DIR / 'roads.shp',
}

# Focus area — adjust if your data covers a different part of the city
CENTER_LAT, CENTER_LON = 52.23, 21.01
RADIUS_DEG = 0.12   # ~10 km radius


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


def main():
    csvs = sorted(DATA_DIR.glob('positions_*.csv'))
    if not csvs:
        print("No data found in live/data/")
        return
    csv_path = csvs[-1]
    print(f"Loading {csv_path.name}...")

    gdf = load_trams(csv_path)
    print(f"  {len(gdf)} tram positions, {gdf['VehicleNumber'].nunique()} vehicles, "
          f"{gdf['Lines'].nunique()} lines")

    # Compute map bounds
    cx, cy = gdf.geometry.x.mean(), gdf.geometry.y.mean()
    # rough degree→metre: 1° lat ≈ 111 km, 1° lon ≈ 70 km at Warsaw latitude
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

    # Background layers
    from shapely.geometry import box as shapely_box
    clip_box = gpd.GeoDataFrame(geometry=[shapely_box(xmin, ymin, xmax, ymax)], crs='EPSG:2180')

    if BACKGROUND_LAYERS['water'].exists():
        water = gpd.read_file(BACKGROUND_LAYERS['water']).to_crs('EPSG:2180')
        water = gpd.clip(water, clip_box)
        if not water.empty:
            water.plot(ax=ax, fc='#0f2e45', ec='none', zorder=1)

    if BACKGROUND_LAYERS['roads'].exists():
        roads = gpd.read_file(BACKGROUND_LAYERS['roads']).to_crs('EPSG:2180')
        roads = gpd.clip(roads, clip_box)
        if not roads.empty:
            roads.plot(ax=ax, fc='none', ec='#333333', lw=0.3, zorder=2)

    # Assign a colour per line
    lines = sorted(gdf['Lines'].unique())
    cmap = cm.get_cmap('tab20', len(lines))
    line_colors = {line: cmap(i) for i, line in enumerate(lines)}

    # Draw traces: one polyline per vehicle, dots at each GPS fix
    for vehicle_id, vdf in gdf.groupby('VehicleNumber'):
        vdf = vdf.sort_values('poll_time')
        xs = vdf.geometry.x.values
        ys = vdf.geometry.y.values
        line_name = vdf['Lines'].iloc[0]
        color = line_colors[line_name]

        # Trace line
        ax.plot(xs, ys, color=color, lw=0.6, alpha=0.5, zorder=3)
        # GPS fix dots
        ax.scatter(xs, ys, color=color, s=3, alpha=0.6, zorder=4, linewidths=0)

    # Legend — top 15 lines by vehicle count
    top_lines = gdf.groupby('Lines')['VehicleNumber'].nunique().sort_values(ascending=False).head(15).index
    handles = [plt.Line2D([0], [0], color=line_colors[l], lw=2, label=f"Tram {l}") for l in top_lines]
    ax.legend(handles=handles, loc='lower right', fontsize=7, framealpha=0.3,
              facecolor='#111111', edgecolor='none', labelcolor='white', ncol=2)

    date_str = csv_path.stem.replace('positions_', '')
    time_range = f"{gdf['poll_time'].min().strftime('%H:%M')} – {gdf['poll_time'].max().strftime('%H:%M')}"
    ax.set_title(f"Warsaw Tram GPS Traces  ·  {date_str}  ·  {time_range}",
                 color='white', fontsize=13, pad=8)

    out = DATA_DIR / f"traces_{date_str}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
