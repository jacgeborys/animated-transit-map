"""
Create diagnostic GeoJSON showing route segments with expected trip counts
"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from pathlib import Path
import json

# Paths
PROJECT_ROOT = Path(__file__).parent
ROUTE_LINES = PROJECT_ROOT / "data" / "processed" / "route_lines_continuous.shp"
SCHEDULE = PROJECT_ROOT / "data" / "processed" / "schedule_for_animation.csv"
OUTPUT = PROJECT_ROOT / "out" / "maps" / "segment_diagnostic.geojson"

TARGET_ROUTES = ['4', '13', '20', '23', '26']

def substring(geom, start_dist, end_dist):
    """Extract substring of LineString between start and end distances"""
    if start_dist < 0 or end_dist > geom.length:
        return None
    
    coords = []
    current_dist = 0
    
    for i in range(len(geom.coords) - 1):
        p1 = Point(geom.coords[i])
        p2 = Point(geom.coords[i + 1])
        segment_length = p1.distance(p2)
        
        if current_dist + segment_length >= start_dist and current_dist <= end_dist:
            if current_dist < start_dist:
                fraction = (start_dist - current_dist) / segment_length
                start_pt = geom.interpolate(start_dist)
                coords.append((start_pt.x, start_pt.y))
            else:
                coords.append(geom.coords[i])
            
            if current_dist + segment_length > end_dist:
                end_pt = geom.interpolate(end_dist)
                coords.append((end_pt.x, end_pt.y))
                break
            else:
                coords.append(geom.coords[i + 1])
        
        current_dist += segment_length
    
    if len(coords) >= 2:
        return LineString(coords)
    return None

# Load data
print("Loading routes...")
routes = gpd.read_file(ROUTE_LINES)
routes['route_id'] = routes['route_id'].astype(str)
routes = routes[routes['route_id'].isin(TARGET_ROUTES)]
print(f"Loaded {len(routes)} routes")

print("Loading schedule...")
schedule = pd.read_csv(SCHEDULE)
schedule['route_id'] = schedule['route_id'].astype(str)
schedule = schedule[schedule['route_id'].isin(TARGET_ROUTES)]

# Count trips per route
trips_per_route = {}
for route_id in TARGET_ROUTES:
    count = schedule[schedule['route_id'] == route_id]['trip_id'].nunique()
    trips_per_route[route_id] = count
    print(f"  Route {route_id}: {count} trips")

# Find junction points
print("\nFinding junction points...")
junction_points = []

for i in range(len(routes)):
    for j in range(i + 1, len(routes)):
        geom_i = routes.iloc[i].geometry
        geom_j = routes.iloc[j].geometry
        
        intersection = geom_i.intersection(geom_j.buffer(15))
        
        if intersection.is_empty:
            continue
            
        if intersection.geom_type == 'LineString':
            coords = list(intersection.coords)
            if len(coords) >= 2:
                junction_points.append(Point(coords[0]))
                junction_points.append(Point(coords[-1]))
        elif intersection.geom_type == 'MultiLineString':
            for line in intersection.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    junction_points.append(Point(coords[0]))
                    junction_points.append(Point(coords[-1]))

# Remove duplicate points
unique_junctions = []
for pt in junction_points:
    is_duplicate = False
    for existing in unique_junctions:
        if pt.distance(existing) < 20:
            is_duplicate = True
            break
    if not is_duplicate:
        unique_junctions.append(pt)

print(f"Found {len(unique_junctions)} unique junction points")

# Split routes into segments
print("\nSplitting routes at junctions...")
all_segments = []

for idx, route in routes.iterrows():
    route_geom = route.geometry
    route_id = route['route_id']
    shape_id = route['shape_id']
    
    # Find junctions on this route
    junctions_on_route = []
    for junction in unique_junctions:
        if route_geom.distance(junction) < 20:
            junctions_on_route.append(junction)
    
    if not junctions_on_route:
        all_segments.append({
            'geometry': route_geom,
            'route_id': route_id,
            'shape_id': shape_id,
            'route_ids': [route_id],
            'length_m': int(route_geom.length)
        })
        continue
    
    # Sort by distance along route
    junctions_with_dist = []
    for junction in junctions_on_route:
        dist = route_geom.project(junction)
        junctions_with_dist.append((dist, junction))
    junctions_with_dist.sort(key=lambda x: x[0])
    
    # Create segments
    current_start = 0
    
    for i, (junction_dist, junction_pt) in enumerate(junctions_with_dist):
        if junction_dist - current_start > 50:
            segment_geom = substring(route_geom, current_start, junction_dist)
            if segment_geom and hasattr(segment_geom, 'length') and segment_geom.length > 50:
                all_segments.append({
                    'geometry': segment_geom,
                    'route_id': route_id,
                    'shape_id': shape_id,
                    'route_ids': [route_id],
                    'length_m': int(segment_geom.length)
                })
        current_start = junction_dist
    
    # Final segment
    if route_geom.length - current_start > 50:
        segment_geom = substring(route_geom, current_start, route_geom.length)
        if segment_geom and hasattr(segment_geom, 'length') and segment_geom.length > 50:
            all_segments.append({
                'geometry': segment_geom,
                'route_id': route_id,
                'shape_id': shape_id,
                'route_ids': [route_id],
                'length_m': int(segment_geom.length)
            })

print(f"Created {len(all_segments)} segments")

# Find overlaps
print("\nFinding overlapping segments...")
for i in range(len(all_segments)):
    for j in range(i + 1, len(all_segments)):
        if all_segments[i]['route_id'] == all_segments[j]['route_id']:
            continue
            
        geom_i = all_segments[i]['geometry']
        geom_j = all_segments[j]['geometry']
        
        overlap = geom_i.buffer(15).intersection(geom_j.buffer(15))
        if hasattr(overlap, 'length') and overlap.length > 100:
            if all_segments[j]['route_id'] not in all_segments[i]['route_ids']:
                all_segments[i]['route_ids'].append(all_segments[j]['route_id'])
            if all_segments[i]['route_id'] not in all_segments[j]['route_ids']:
                all_segments[j]['route_ids'].append(all_segments[i]['route_id'])

# Calculate expected trips per segment
for seg in all_segments:
    seg['num_routes'] = len(seg['route_ids'])
    seg['expected_trips'] = sum(trips_per_route.get(rid, 0) for rid in seg['route_ids'])
    seg['routes_using'] = ','.join(sorted(seg['route_ids']))

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(all_segments, crs=routes.crs)

# Statistics
print("\nSegment statistics:")
overlap_counts = gdf['num_routes'].value_counts().sort_index()
for num_routes, count in overlap_counts.items():
    print(f"  {count} segments with {num_routes} route(s)")

print(f"\nExpected trips: min={gdf['expected_trips'].min()}, max={gdf['expected_trips'].max()}")

# Show busiest segments
print("\nTop 10 busiest segments:")
top_segments = gdf.nlargest(10, 'expected_trips')[['route_id', 'num_routes', 'routes_using', 'expected_trips', 'length_m']]
for idx, row in top_segments.iterrows():
    print(f"  {row['routes_using']}: {row['expected_trips']} trips, {row['length_m']}m, {row['num_routes']} routes")

# Save
print(f"\nSaving to {OUTPUT}...")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
gdf.to_file(OUTPUT, driver='GeoJSON')
print("Done!")
