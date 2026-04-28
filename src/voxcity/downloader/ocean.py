"""
Ocean detection using OSM coastlines via Overpass API.

OSM handles oceans by the "absence of land" principle:
1. The renderer starts with a blue canvas
2. Land polygons (derived from natural=coastline) are drawn on top
3. Anything not covered by land is ocean

This module queries coastlines from Overpass API and determines land/ocean
based on the coastline orientation rule: land is on the LEFT of the coastline.

For areas without coastlines, we check if the point is in the ocean using
a simple heuristic based on nearby land features.
"""
import os
import tempfile
import hashlib
import time
from pathlib import Path
from typing import List, Tuple, Optional
import requests
import numpy as np
from ..utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP

# Cache directory for ocean detection results (optional)
CACHE_DIR = Path(tempfile.gettempdir()) / "voxcity_ocean_cache"


def get_land_polygon_for_area(rectangle_vertices: List[Tuple[float, float]], use_cache: bool = False):
    """
    Get the land polygon for a given area using OSM coastlines.
    
    This is the main entry point for ocean detection. It queries coastlines
    from Overpass API and builds a polygon representing land areas.
    
    Args:
        rectangle_vertices: List of (lon, lat) tuples defining the area
        use_cache: Whether to use disk cache (default False)
        
    Returns:
        Shapely Polygon/MultiPolygon representing land, or None if no coastlines found
    """
    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)
    
    bbox = (min_lon, min_lat, max_lon, max_lat)
    
    # Query coastlines from Overpass API
    overpass_data, ok = query_coastlines_from_overpass(min_lat, min_lon, max_lat, max_lon)

    if not ok:
        # API failure - signal to caller; do not silently treat as "no coastlines"
        return None

    # Count coastlines
    coastline_count = sum(1 for e in overpass_data.get('elements', []) 
                         if e.get('type') == 'way' and e.get('tags', {}).get('natural') == 'coastline')
    
    if coastline_count == 0:
        return None
    
    # Build land polygon from coastlines
    land_polygon = build_coastline_polygons(overpass_data, bbox)
    
    return land_polygon


def get_cache_path(rectangle_vertices: List[Tuple[float, float]], grid_shape: Tuple[int, int]) -> Path:
    """Generate a cache filename based on the bounding box and grid shape."""
    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)
    
    bbox_str = f"{min_lon:.6f}_{min_lat:.6f}_{max_lon:.6f}_{max_lat:.6f}_{grid_shape[0]}_{grid_shape[1]}"
    cache_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:12]
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"ocean_mask_{cache_hash}.npy"


def query_coastlines_from_overpass(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float,
    buffer_deg: float = 0.1,
    max_retries: int = 2,
) -> Tuple[dict, bool]:
    """
    Query coastline ways from Overpass API.

    Args:
        min_lat, min_lon, max_lat, max_lon: Bounding box
        buffer_deg: Buffer around bbox to catch coastlines that might affect the area
        max_retries: Per-endpoint retry count for transient errors (e.g., 429/5xx)

    Returns:
        Tuple ``(data, ok)`` where ``data`` is the Overpass JSON response
        (``{'elements': []}`` on failure) and ``ok`` is True only if the API
        actually returned a successful response. Callers should treat
        ``ok == False`` as "API failure" rather than "no coastlines in area".
    """
    # Expand bbox slightly to catch nearby coastlines
    query_min_lat = min_lat - buffer_deg
    query_max_lat = max_lat + buffer_deg
    query_min_lon = min_lon - buffer_deg
    query_max_lon = max_lon + buffer_deg
    
    query = f"""
    [out:json][timeout:30];
    (
      way["natural"="coastline"]({query_min_lat},{query_min_lon},{query_max_lat},{query_max_lon});
    );
    out body;
    >;
    out skel qt;
    """
    
    overpass_endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    
    headers = {"User-Agent": "voxcity/1.0 (https://github.com/kunifujiwara/voxcity)"}

    last_error = None
    for endpoint in overpass_endpoints:
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(
                    endpoint, params={'data': query}, headers=headers, timeout=60
                )
                if response.status_code == 200:
                    try:
                        return response.json(), True
                    except ValueError as e:
                        last_error = f"invalid JSON from {endpoint}: {e}"
                        break  # don't retry on bad JSON; try next endpoint

                # Transient errors: retry with backoff (honor Retry-After if present)
                if response.status_code in (429, 502, 503, 504):
                    last_error = f"{endpoint} returned HTTP {response.status_code}"
                    if attempt < max_retries:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            delay = float(retry_after) if retry_after else 2.0 * (attempt + 1)
                        except ValueError:
                            delay = 2.0 * (attempt + 1)
                        time.sleep(min(delay, 30.0))
                        continue
                    break  # exhausted retries; try next endpoint

                # Other non-200: don't retry, try next endpoint
                last_error = f"{endpoint} returned HTTP {response.status_code}"
                break
            except requests.RequestException as e:
                last_error = f"{endpoint} request error: {e}"
                if attempt < max_retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                break

    if last_error:
        print(f"  Warning: Overpass coastline query failed ({last_error}).")
    return {'elements': []}, False


def build_coastline_polygons(overpass_data: dict, bbox: Tuple[float, float, float, float]):
    """
    Build land polygons by splitting bbox with coastlines.
    
    Algorithm:
    1. Merge bbox boundary with all clipped coastlines to form a network
    2. Use polygonize to create all possible polygons
    3. For each polygon, determine if it's land by checking the coastline direction
       (land is on the LEFT of coastline when walking along it)
    
    Returns:
        Shapely polygon representing land area, or None if processing fails
    """
    from shapely.geometry import LineString, Polygon, box, Point, MultiLineString
    from shapely.ops import linemerge, unary_union, polygonize, split
    import math
    
    elements = overpass_data.get('elements', [])
    
    # Build node lookup
    nodes = {}
    for elem in elements:
        if elem.get('type') == 'node':
            nodes[elem['id']] = (elem['lon'], elem['lat'])
    
    # Build coastline linestrings preserving direction
    coastlines = []
    for elem in elements:
        if elem.get('type') == 'way' and elem.get('tags', {}).get('natural') == 'coastline':
            node_ids = elem.get('nodes', [])
            coords = [nodes[nid] for nid in node_ids if nid in nodes]
            if len(coords) >= 2:
                coastlines.append(LineString(coords))
    
    if not coastlines:
        return None
    
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)
    bbox_boundary = bbox_polygon.exterior
    
    try:
        # Clip each coastline to bbox and collect segments with their direction
        clipped_segments = []  # List of (LineString, original_direction_preserved)
        
        for coastline in coastlines:
            if not coastline.intersects(bbox_polygon):
                continue
            
            # Get intersection with bbox
            clipped = coastline.intersection(bbox_polygon)
            if clipped.is_empty:
                continue
            
            # Extract LineStrings
            if clipped.geom_type == 'LineString':
                if len(clipped.coords) >= 2:
                    clipped_segments.append(clipped)
            elif clipped.geom_type == 'MultiLineString':
                for line in clipped.geoms:
                    if len(line.coords) >= 2:
                        clipped_segments.append(line)
            elif clipped.geom_type == 'GeometryCollection':
                for geom in clipped.geoms:
                    if geom.geom_type == 'LineString' and len(geom.coords) >= 2:
                        clipped_segments.append(geom)
        
        if not clipped_segments:
            return None
        
        # Combine bbox boundary with coastlines to create a line network
        all_lines = [bbox_boundary] + clipped_segments
        merged_lines = unary_union(all_lines)
        
        # Polygonize the network
        polygons = list(polygonize(merged_lines))
        
        if not polygons:
            return None
        
        # For each polygon, determine if it's land or water
        # A polygon is land if it's on the LEFT side of the coastlines that bound it
        land_polygons = []
        
        for poly in polygons:
            if not poly.is_valid or poly.is_empty:
                continue
            
            # Check if this polygon is inside the bbox
            if not poly.intersects(bbox_polygon):
                continue
            
            # Clip to bbox
            poly_clipped = poly.intersection(bbox_polygon)
            if poly_clipped.is_empty or poly_clipped.area < 1e-12:
                continue
            
            # Determine if this polygon is land
            # Sample a point inside the polygon and check if it's on the land side
            # of the nearby coastlines
            is_land = is_polygon_land(poly_clipped, clipped_segments)
            
            if is_land:
                land_polygons.append(poly_clipped)
        
        if land_polygons:
            result = unary_union(land_polygons)
            if not result.is_empty and result.area > 0:
                return result
        
        return None
        
    except Exception as e:
        return None


def is_polygon_land(polygon, coastline_segments):
    """
    Determine if a polygon is land based on coastline orientation.

    OSM Rule: Land is on the LEFT of the coastline direction.

    Algorithm (orientation-match, robust to non-convex / harbor-shaped polygons):
      1. Force the polygon's exterior to CCW orientation. With CCW orientation,
         the polygon's interior lies on the LEFT of every exterior edge as you
         walk around it.
      2. Walk consecutive vertex pairs ``(a, b)`` of the exterior.
      3. For each edge whose midpoint lies on a coastline (within a tiny
         tolerance), look up the coastline segment containing that midpoint
         and compare its direction to the polygon edge direction via dot
         product:
            - dot > 0  -> edge follows coastline forward -> interior on
              LEFT of coastline -> LAND vote.
            - dot < 0  -> edge runs against coastline -> interior on
              RIGHT of coastline -> WATER vote.
      4. Majority vote across all coastline-coincident edges. Defaults to
         land on tie or when no coastline-coincident edges exist (e.g., a
         polygon bounded entirely by the bbox boundary).
    """
    from shapely.geometry import LineString
    from shapely.geometry.polygon import orient

    try:
        polygon = orient(polygon, sign=1.0)  # CCW exterior
    except Exception:
        pass

    try:
        coords = list(polygon.exterior.coords)
    except Exception:
        return True

    if len(coords) < 2:
        return True

    # Tolerance: roughly the polygonize/snap precision in degrees.
    tol = 1e-9

    votes_land = 0
    votes_water = 0

    for i in range(len(coords) - 1):
        ax, ay = coords[i]
        bx, by = coords[i + 1]
        edx = bx - ax
        edy = by - ay
        if edx == 0 and edy == 0:
            continue
        mx = (ax + bx) / 2.0
        my = (ay + by) / 2.0

        # Find a coastline whose geometry passes through this edge midpoint.
        matching_coastline = None
        for cl in coastline_segments:
            try:
                if cl.distance(LineString([(ax, ay), (bx, by)])) <= tol:
                    matching_coastline = cl
                    break
            except Exception:
                continue
        if matching_coastline is None:
            continue  # bbox-boundary edge or unrelated edge

        # Locate the coastline segment containing the midpoint.
        cl_coords = list(matching_coastline.coords)
        best_dist = float('inf')
        best_k = 0
        for k in range(len(cl_coords) - 1):
            seg = LineString([cl_coords[k], cl_coords[k + 1]])
            d = seg.distance(LineString([(ax, ay), (bx, by)]))
            if d < best_dist:
                best_dist = d
                best_k = k
        cdx = cl_coords[best_k + 1][0] - cl_coords[best_k][0]
        cdy = cl_coords[best_k + 1][1] - cl_coords[best_k][1]

        dot = cdx * edx + cdy * edy
        if dot > 0:
            votes_land += 1
        elif dot < 0:
            votes_water += 1

    if votes_land == 0 and votes_water == 0:
        return True  # no coastline-coincident edges; default to land
    return votes_land >= votes_water


def check_if_area_is_ocean_via_land_features(
    rectangle_vertices: List[Tuple[float, float]]
) -> bool:
    """
    Quick check: if an area has buildings/roads/land-use features, it's not pure ocean.
    
    Returns True if the area appears to be mostly ocean (few land features).
    """
    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)
    
    # Quick query for any land features
    query = f"""
    [out:json][timeout:10];
    (
      way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["landuse"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out count;
    """
    
    try:
        response = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={'data': query},
            headers={"User-Agent": "voxcity/1.0"},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            # If there are substantial land features, it's not ocean
            count = data.get('elements', [{}])[0].get('tags', {}).get('total', 0)
            if isinstance(count, str):
                count = int(count)
            return count < 10  # Very few land features = likely ocean
    except Exception:
        pass
    
    return False  # Default to not ocean if we can't determine


def get_land_mask_from_coastlines(
    rectangle_vertices: List[Tuple[float, float]],
    grid_shape: Tuple[int, int],
    use_cache: bool = True
) -> np.ndarray:
    """
    Create a boolean mask where True = land, False = ocean.
    
    Uses Overpass API to query coastlines and determine land/ocean areas.
    Much faster than downloading the full 600MB land polygons file.
    
    Args:
        rectangle_vertices: List of (lon, lat) tuples defining the area
        grid_shape: (rows, cols) of the output grid
        use_cache: Whether to cache the result
        
    Returns:
        np.ndarray: Boolean array where True = land, False = ocean
    """
    from shapely.geometry import box, Point
    from rasterio import features
    from affine import Affine
    
    cache_path = get_cache_path(rectangle_vertices, grid_shape)
    
    # Check cache
    if use_cache and cache_path.exists():
        try:
            cached = np.load(cache_path)
            if cached.shape == grid_shape:
                return cached
        except Exception:
            pass
    
    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)
    
    rows, cols = grid_shape
    
    # Query coastlines
    print("  Querying coastlines from Overpass API...")
    overpass_data, ok = query_coastlines_from_overpass(min_lat, min_lon, max_lat, max_lon)

    if not ok:
        # API failure - do not silently mislabel the area via heuristic.
        print("  Warning: Overpass API failed; skipping ocean detection (treating area as land).")
        land_mask = np.ones(grid_shape, dtype=bool)
        if use_cache:
            try:
                np.save(cache_path, land_mask)
            except Exception:
                pass
        return land_mask

    coastline_count = sum(1 for e in overpass_data.get('elements', [])
                         if e.get('type') == 'way' and e.get('tags', {}).get('natural') == 'coastline')
    
    if coastline_count == 0:
        # No coastlines in area - check if it's inland or open ocean
        print("  No coastlines found in area.")
        
        # Quick heuristic: if there are land features, assume all land
        # If no land features, check if we're in the middle of the ocean
        is_mostly_ocean = check_if_area_is_ocean_via_land_features(rectangle_vertices)
        
        if is_mostly_ocean:
            print("  Area appears to be open ocean (few land features).")
            land_mask = np.zeros(grid_shape, dtype=bool)
        else:
            print("  Area appears to be inland (has land features).")
            land_mask = np.ones(grid_shape, dtype=bool)
    else:
        print(f"  Found {coastline_count} coastline segments.")
        
        # Build land polygons from coastlines
        land_polygon = build_coastline_polygons(
            overpass_data, 
            (min_lon, min_lat, max_lon, max_lat)
        )
        
        if land_polygon is None:
            # Coastline processing failed - use heuristic
            print("  Could not build land polygon from coastlines, using land feature heuristic.")
            is_mostly_ocean = check_if_area_is_ocean_via_land_features(rectangle_vertices)
            land_mask = np.zeros(grid_shape, dtype=bool) if is_mostly_ocean else np.ones(grid_shape, dtype=bool)
        else:
            # Rasterize land polygon
            pixel_width = (max_lon - min_lon) / cols
            pixel_height = (max_lat - min_lat) / rows
            transform = Affine(pixel_width, 0, min_lon, 0, -pixel_height, max_lat)
            
            land_mask = np.zeros(grid_shape, dtype=np.uint8)
            
            try:
                if land_polygon.geom_type == 'Polygon':
                    geometries = [(land_polygon, 1)]
                else:  # MultiPolygon
                    geometries = [(geom, 1) for geom in land_polygon.geoms]
                
                features.rasterize(
                    shapes=geometries,
                    out=land_mask,
                    transform=transform,
                    all_touched=False
                )
                land_mask = land_mask.astype(bool)
            except Exception as e:
                print(f"  Warning: Rasterization failed: {e}")
                land_mask = np.ones(grid_shape, dtype=bool)
    
    # Cache the result
    if use_cache:
        try:
            np.save(cache_path, land_mask)
        except Exception:
            pass
    
    return land_mask


# Alias for backward compatibility
get_land_mask_from_osm_land_polygons = get_land_mask_from_coastlines


def get_ocean_class_for_source(source: str) -> str:
    """Get the appropriate ocean/water class name for a given land cover source."""
    if source == "Urbanwatch":
        return "Sea"
    elif source in ["OpenStreetMap", "Standard", "OpenEarthMapJapan"]:
        return "Water"
    elif source == "ESA WorldCover":
        return "Open water"
    elif source == "Dynamic World V1":
        return "Water"
    elif source == "ESRI 10m Annual Land Cover":
        return "Water"
    else:
        return "Water"


def apply_ocean_mask_to_grid(
    grid: np.ndarray,
    rectangle_vertices: List[Tuple[float, float]],
    source: str = "OpenStreetMap",
    ocean_class: Optional[str] = None
) -> np.ndarray:
    """
    Apply ocean detection to an existing land cover grid.
    
    Cells that are:
    1. Currently set to the default class (e.g., 'Developed space')
    2. Located in ocean areas (outside OSM land polygons)
    
    Will be changed to the ocean/water class.
    
    Args:
        grid: Land cover grid (2D array of class names)
        rectangle_vertices: Area coordinates
        source: Land cover source name
        ocean_class: Override for ocean class name (auto-detected if None)
        
    Returns:
        Updated grid with ocean areas classified as water
    """
    if ocean_class is None:
        ocean_class = get_ocean_class_for_source(source)
    
    # Get default class for this source
    default_classes = {
        "OpenStreetMap": "Developed space",
        "Standard": "Developed space", 
        "OpenEarthMapJapan": "Developed space",
        "Urbanwatch": "Unknown",
        "ESA WorldCover": "Barren / sparse vegetation",
        "Dynamic World V1": "Bare",
        "ESRI 10m Annual Land Cover": "Bare Ground",
    }
    default_class = default_classes.get(source, "Developed space")
    
    # Get land mask
    land_mask = get_land_mask_from_osm_land_polygons(
        rectangle_vertices, 
        grid.shape
    )
    
    # Flip land mask to match grid orientation (grid is flipped at the end of creation)
    land_mask = ensure_orientation(land_mask, ORIENTATION_SOUTH_UP, ORIENTATION_NORTH_UP)
    
    # Apply ocean class to cells that are:
    # 1. Not land (ocean according to OSM land polygons)
    # 2. Currently classified as the default class
    ocean_cells = ~land_mask & (grid == default_class)
    
    grid_updated = grid.copy()
    grid_updated[ocean_cells] = ocean_class
    
    ocean_count = np.sum(ocean_cells)
    if ocean_count > 0:
        total_cells = grid.size
        pct = 100 * ocean_count / total_cells
        print(f"  Ocean detection: {ocean_count:,} cells ({pct:.1f}%) classified as '{ocean_class}'")
    
    return grid_updated
