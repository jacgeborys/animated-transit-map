"""
Test Frame Generator - Noon Snapshot
=====================================
Generates a single PNG at 12:00 noon to preview visual settings
before running the full animation.

Shows:
  - Dark background layers (water, forests, roads)
  - Transit segments with density-based coloring
  - Vehicle dots at their noon positions

Usage:
  python animate/test_frame_noon.py
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
from shapely.geometry import box
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED     = PROJECT_ROOT / "data" / "processed"
ROUTE_LINES   = PROCESSED / "route_lines_continuous.shp"
SCHEDULE      = PROCESSED / "schedule_for_animation.csv"

OSM = Path(r"D:\QGIS\mapy_warszawy_misc\data\osm")
BACKGROUND_LAYERS = {
    "water":      OSM / "water.gpkg",
    "waterways":  OSM / "waterways.gpkg",
    "forests":    OSM / "forests.gpkg",
    "roads":      OSM / "roads.shp",
}

OUTPUT = PROJECT_ROOT / "animate" / "_output" / "test_frame_noon.png"

# ─── Map settings ─────────────────────────────────────────────────────────────

TARGET_CRS        = "EPSG:2180"
CENTRAL_X         = 638000
CENTRAL_Y         = 487000
FRAME_SIZE        = 20000          # same as animation
TARGET_TIME_SEC   = 12 * 3600     # noon

# ─── Visual settings ──────────────────────────────────────────────────────────

BG_COLOR = "#000000"

# Background layer colors (keep very dark)
LAYER_COLORS = {
    "forests":    {"fc": "#060d06", "ec": "none",    "lw": 0,   "zorder": 1},
    "water":      {"fc": "#05101a", "ec": "none",    "lw": 0,   "zorder": 2},
    "waterways":  {"fc": "none",    "ec": "#05101a", "lw": 0.8, "zorder": 3},
    "roads":      {"fc": "none",    "ec": "#161616", "lw": 0.5, "zorder": 4},
}

# Transit colors (same as animation)
COLOR_GRADIENTS = {
    "Tram":  {"dark": "#160000", "bright": "#ed1f1f"},
    "Bus":   {"dark": "#1b0020", "bright": "#b613d7"},
    "Train": {"dark": "#0a2e2a", "bright": "#3d9a8f"},
}
LINE_WIDTHS      = {"Tram": 1.5, "Bus": 1.2, "Train": 1.5}
GLOW_WIDTH       = 4.0
GLOW_ALPHA       = 0.7
BASE_BRIGHTNESS  = 0.15
MAX_BRIGHTNESS   = 1.0
VEHICLE_COLORS   = {"Tram": "#FF7075", "Bus": "#B46EFC", "Train": "#6BC9C6"}
VEHICLE_SIZES    = {"Tram": 16, "Bus": 12, "Train": 19}
Z_ORDERS = {
    "Train": {"glow": 10, "line": 20},
    "Bus":   {"glow": 30, "line": 40},
    "Tram":  {"glow": 50, "line": 60},
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def hex_lerp(c1, c2, t):
    """Linearly interpolate between two hex colors."""
    r1, g1, b1 = (int(c1.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    r2, g2, b2 = (int(c2.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    half = FRAME_SIZE / 2
    xmin, xmax = CENTRAL_X - half, CENTRAL_X + half
    ymin, ymax = CENTRAL_Y - half, CENTRAL_Y + half
    clip_box = box(xmin, ymin, xmax, ymax)

    # ── 1. Figure setup ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 12), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ── 2. Background layers ──────────────────────────────────────────────────
    for name, path in BACKGROUND_LAYERS.items():
        if not path.exists():
            logger.warning(f"Background layer not found, skipping: {path}")
            continue

        logger.info(f"Loading {name}...")
        gdf = gpd.read_file(path)

        if gdf.crs is None:
            logger.warning(f"  {name} has no CRS, assuming EPSG:4326")
            gdf = gdf.set_crs("EPSG:4326")

        logger.info(f"  {name}: CRS={gdf.crs.to_epsg()}, bounds={gdf.total_bounds.round(1)}")

        if gdf.crs.to_epsg() != 2180:
            gdf = gdf.to_crs(TARGET_CRS)
            logger.info(f"  {name}: reprojected bounds={gdf.total_bounds.round(1)}")

        # Fix invalid geometries (buffer(0) only safe for polygons — destroys lines)
        if gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
        clip_gdf = gpd.GeoDataFrame(geometry=[clip_box], crs=TARGET_CRS)
        gdf = gpd.clip(gdf, clip_gdf)

        if gdf.empty:
            logger.info(f"  {name}: no features in frame extent, skipping")
            continue

        style = LAYER_COLORS[name]
        gdf.plot(
            ax=ax,
            facecolor=style["fc"],
            edgecolor=style["ec"],
            linewidth=style["lw"],
            zorder=style["zorder"],
        )
        logger.info(f"  {name}: {len(gdf)} features drawn")

    # ── 3. Load transit data ──────────────────────────────────────────────────
    logger.info("Loading junction segments...")
    all_segments = []
    for vtype in ["tram", "bus", "train"]:
        seg_file = PROCESSED / f"{vtype}_junction_segments.shp"
        if seg_file.exists():
            s = gpd.read_file(seg_file)
            s["vehicle_ty"] = vtype.capitalize()
            all_segments.append(s)
            logger.info(f"  {vtype}: {len(s)} segments")
        else:
            logger.warning(f"  {seg_file.name} not found, skipping")

    if not all_segments:
        logger.error("No junction segment files found. Run core/junction_segmenter.py first.")
        return 1

    segments = gpd.GeoDataFrame(pd.concat(all_segments, ignore_index=True))
    segments["route_ids_list"] = segments["route_ids"].apply(
        lambda x: x.split(",") if isinstance(x, str) else []
    )

    logger.info("Loading routes and schedule...")
    routes   = gpd.read_file(ROUTE_LINES)
    routes["route_id"] = routes["route_id"].astype(str)
    schedule = pd.read_csv(SCHEDULE)
    schedule["route_id"] = schedule["route_id"].astype(str)

    schedule["departure_seconds"] = schedule["departure_time_str"].apply(
        lambda t: sum(int(x) * s for x, s in zip(t.split(":"), [3600, 60, 1]))
        if isinstance(t, str) else np.nan
    )
    schedule = schedule.dropna(subset=["departure_seconds"])

    # ── 4. Route lookup ───────────────────────────────────────────────────────
    route_lookup = {
        row["shape_id"]: {
            "geometry": row.geometry,
            "vehicle":  row["vehicle"],
            "route_id": row["route_id"],
        }
        for _, row in routes.iterrows()
    }

    route_to_segments = {}
    for idx, segment in segments.iterrows():
        for route_id in segment["route_ids_list"]:
            for shape_id in routes[routes["route_id"] == route_id]["shape_id"].tolist():
                route_to_segments.setdefault(shape_id, [])
                if idx not in route_to_segments[shape_id]:
                    route_to_segments[shape_id].append(idx)

    # ── 5. Noon density snapshot ──────────────────────────────────────────────
    logger.info(f"Computing density snapshot at {TARGET_TIME_SEC // 3600:02d}:00...")

    first_stops = schedule[schedule["stop_sequence"] == 1].copy()
    segment_vehicle_count = np.zeros(len(segments))

    active_vehicles = []
    for _, trip in first_stops.iterrows():
        shape_id = trip["shape_id"]
        if shape_id not in route_lookup:
            continue

        route_geom   = route_lookup[shape_id]["geometry"]
        trip_duration = route_geom.length / 4.7   # ~4.7 m/s average speed
        dep           = trip["departure_seconds"]

        if dep <= TARGET_TIME_SEC <= dep + trip_duration:
            progress = (TARGET_TIME_SEC - dep) / trip_duration
            position = route_geom.interpolate(progress, normalized=True)
            active_vehicles.append({
                "shape_id": shape_id,
                "vehicle":  route_lookup[shape_id]["vehicle"],
                "position": position,
            })

            # Credit segments this vehicle is near
            if shape_id in route_to_segments:
                for seg_idx in route_to_segments[shape_id]:
                    segment_vehicle_count[seg_idx] += 1

    logger.info(f"  Active vehicles at noon: {len(active_vehicles)}")

    # Calibrate brightness against observed peak
    densities = np.zeros(len(segments))
    for i, seg in segments.iterrows():
        lkm = seg.get("length_km", 0)
        if lkm and lkm > 0:
            densities[i] = segment_vehicle_count[i] / lkm

    density_peak = np.percentile(densities[densities > 0], 90) if (densities > 0).any() else 1.0
    density_for_max = density_peak * 0.3   # 30% of 90th-percentile = full brightness
    logger.info(f"  90th-pct density: {density_peak:.1f} veh/km  →  max-brightness at {density_for_max:.1f} veh/km")

    # ── 6. Draw base transit segments ─────────────────────────────────────────
    logger.info("Drawing base transit segments...")
    seg_by_type = {"Tram": [], "Bus": [], "Train": []}
    for _, seg in segments.iterrows():
        geom = seg.geometry
        if hasattr(geom, "coords"):
            x, y = geom.xy
            seg_by_type[seg["vehicle_ty"]].append(np.column_stack([x, y]))

    for vtype in ["Train", "Bus", "Tram"]:
        if not seg_by_type[vtype]:
            continue
        dark = COLOR_GRADIENTS[vtype]["dark"]
        lc = LineCollection(
            seg_by_type[vtype],
            colors=dark,
            linewidths=LINE_WIDTHS.get(vtype, 1.5),
            zorder=Z_ORDERS[vtype]["line"],
            capstyle="round", joinstyle="round",
        )
        ax.add_collection(lc)

    # ── 7. Draw bright segments based on density ───────────────────────────────
    logger.info("Drawing density-lit segments...")
    for i, seg in segments.iterrows():
        geom = seg.geometry
        if not hasattr(geom, "coords"):
            continue

        brightness_ratio = densities[i] / density_for_max if density_for_max > 0 else 0
        brightness = BASE_BRIGHTNESS + (MAX_BRIGHTNESS - BASE_BRIGHTNESS) * brightness_ratio
        brightness = min(brightness, MAX_BRIGHTNESS)

        if brightness <= BASE_BRIGHTNESS + 0.001:
            continue   # at base level, already drawn above

        x, y   = geom.xy
        vtype  = seg["vehicle_ty"]
        grad   = COLOR_GRADIENTS.get(vtype, COLOR_GRADIENTS["Tram"])
        color  = hex_lerp(grad["dark"], grad["bright"], brightness)
        zorder = Z_ORDERS.get(vtype, Z_ORDERS["Tram"])

        glow_progress = (brightness - BASE_BRIGHTNESS) / (MAX_BRIGHTNESS - BASE_BRIGHTNESS)
        glow_alpha    = max(0, glow_progress * GLOW_ALPHA)

        ax.plot(x, y, color=grad["bright"], linewidth=GLOW_WIDTH,
                alpha=glow_alpha, solid_capstyle="round", zorder=zorder["glow"])
        ax.plot(x, y, color=color, linewidth=LINE_WIDTHS.get(vtype, 1.5),
                alpha=1.0, solid_capstyle="round", zorder=zorder["line"])

    # ── 8. Draw vehicle dots ───────────────────────────────────────────────────
    logger.info("Drawing vehicle dots...")
    by_type = {"Tram": [], "Bus": [], "Train": []}
    for v in active_vehicles:
        by_type[v["vehicle"]].append(v["position"])

    for vtype, positions in by_type.items():
        if not positions:
            continue
        xs = [p.x for p in positions]
        ys = [p.y for p in positions]
        ax.scatter(xs, ys, s=VEHICLE_SIZES[vtype], color=VEHICLE_COLORS[vtype],
                   alpha=0.85, zorder=80)

    # ── 9. Labels ─────────────────────────────────────────────────────────────
    ax.text(0.02, 0.98, "Warsaw Public Transit — 12:00",
            transform=ax.transAxes, fontsize=22, color="white",
            verticalalignment="top", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6), zorder=100)
    ax.text(0.02, 0.92, f"Active vehicles: {len(active_vehicles)}",
            transform=ax.transAxes, fontsize=15, color="white",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6), zorder=100)
    ax.text(0.98, 0.02, "© Jacek Gęborys",
            transform=ax.transAxes, fontsize=11, color="white",
            verticalalignment="bottom", horizontalalignment="right",
            alpha=0.5, zorder=100)

    # ── 10. Save ──────────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving → {OUTPUT}")
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    exit(main())
