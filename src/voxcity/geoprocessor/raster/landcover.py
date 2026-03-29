import numpy as np
from typing import List, Tuple, Dict, Any
from shapely.geometry import Polygon
from affine import Affine
import rasterio

from ..utils import initialize_geod
from ...utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP

from ...utils.lc import (
    get_class_priority,
    create_land_cover_polygons,
    get_dominant_class,
)
from .core import translate_array


def tree_height_grid_from_land_cover(land_cover_grid_ori: np.ndarray) -> np.ndarray:
    """
    Convert a land cover grid to a tree height grid.
    
    Expects 1-based land cover indices where class 5 is Tree.
    """
    # 1-based indices: 1=Bareland, 2=Rangeland, 3=Shrub, 4=Agriculture, 5=Tree, etc.
    tree_translation_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 10, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0}
    # Double-flip was a no-op; operate directly on the original grid.
    tree_height_grid = translate_array(land_cover_grid_ori, tree_translation_dict).astype(int)
    return tree_height_grid


def create_land_cover_grid_from_geotiff_polygon(
    tiff_path: str,
    mesh_size: float,
    land_cover_classes: Dict[str, Any],
    polygon: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Create a land cover grid from a GeoTIFF file within a polygon boundary.

    Uses :func:`compute_cell_center_coords` so rotated rectangles are handled
    correctly.
    """
    from .core import compute_cell_center_coords

    cc = compute_cell_center_coords(polygon, mesh_size)
    nx, ny = cc["grid_size"]
    center_lons = cc["lons"].ravel()
    center_lats = cc["lats"].ravel()

    with rasterio.open(tiff_path) as src:
        img = src.read((1, 2, 3))

        # Transform cell centres to raster CRS if needed
        if src.crs and src.crs.to_epsg() != 4326:
            from pyproj import Transformer
            tfm = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            sx, sy = tfm.transform(center_lons, center_lats)
        else:
            sx, sy = center_lons.copy(), center_lats.copy()

        row, col = rasterio.transform.rowcol(src.transform, sx, sy)
        row, col = np.array(row), np.array(col)

        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)

        grid = np.full(nx * ny, 'No Data', dtype=object)
        for idx in np.where(valid)[0]:
            cell_data = img[:, row[idx], col[idx]]
            dominant_class = get_dominant_class(cell_data, land_cover_classes)
            grid[idx] = dominant_class

        grid = grid.reshape(nx, ny)

    return grid


def create_land_cover_grid_from_gdf_polygon(
    gdf,
    meshsize: float,
    source: str,
    rectangle_vertices: List[Tuple[float, float]],
    default_class: str = 'Developed space',
    detect_ocean: bool = True,
    land_polygon = "NOT_PROVIDED"
) -> np.ndarray:
    """
    Create a grid of land cover classes from GeoDataFrame polygon data.
    
    Uses vectorized rasterization for ~100x speedup over cell-by-cell intersection.
    Correctly handles rotated rectangles by rasterizing onto a bounding-box
    grid and then sampling at the rotated cell centre coordinates.
    
    Args:
        gdf: GeoDataFrame with land cover polygons and 'class' column
        meshsize: Grid cell size in meters
        source: Land cover data source name (e.g., 'OpenStreetMap')
        rectangle_vertices: List of (lon, lat) tuples defining the area
        default_class: Default class for cells not covered by any polygon
        detect_ocean: If True, use OSM land polygons to detect ocean areas.
                     Areas outside land polygons will be classified as 'Water'
                     instead of the default class.
        land_polygon: Optional pre-computed land polygon from OSM coastlines.
                     If provided (including None), this is used directly.
                     If "NOT_PROVIDED", coastlines will be queried when detect_ocean=True.
    
    Returns:
        2D numpy array of land cover class names
    """
    import numpy as np
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import box, Polygon as ShapelyPolygon

    class_priority = get_class_priority(source)

    from ..utils import (
        initialize_geod,
        calculate_distance,
        normalize_to_one_meter,
    )
    from .core import calculate_grid_size, compute_grid_geometry, compute_cell_center_coords

    # Calculate grid dimensions (correct for rotated rectangles)
    geom = compute_grid_geometry(rectangle_vertices, meshsize)
    origin = geom["origin"]
    side_1, side_2 = geom["side_1"], geom["side_2"]
    u_vec, v_vec = geom["u_vec"], geom["v_vec"]
    grid_size, adjusted_meshsize = geom["grid_size"], geom["adj_mesh"]
    rows, cols = grid_size

    # Get bounding box for the rasterization working grid
    min_lon = min(coord[0] for coord in rectangle_vertices)
    max_lon = max(coord[0] for coord in rectangle_vertices)
    min_lat = min(coord[1] for coord in rectangle_vertices)
    max_lat = max(coord[1] for coord in rectangle_vertices)

    # Compute bounding-box grid dimensions (may differ from rows/cols for rotated rects)
    geod_inst = initialize_geod()
    _, _, bb_width_m = geod_inst.inv(min_lon, min_lat, max_lon, min_lat)
    _, _, bb_height_m = geod_inst.inv(min_lon, min_lat, min_lon, max_lat)
    bb_cols = max(1, int(bb_width_m / meshsize + 0.5))
    bb_rows = max(1, int(bb_height_m / meshsize + 0.5))

    # Create affine transform for the bounding-box working grid
    pixel_width = (max_lon - min_lon) / bb_cols
    pixel_height = (max_lat - min_lat) / bb_rows
    transform = Affine(pixel_width, 0, min_lon, 0, -pixel_height, max_lat)

    # Build class name to priority mapping, then sort classes by priority (highest priority = lowest number = rasterize last)
    unique_classes = gdf['class'].unique().tolist()
    if default_class not in unique_classes:
        unique_classes.append(default_class)
    
    # Map class names to integer codes
    class_to_code = {cls: i for i, cls in enumerate(unique_classes)}
    code_to_class = {i: cls for cls, i in class_to_code.items()}
    default_code = class_to_code[default_class]

    # Initialize bounding-box grid with default class code
    bb_grid = np.full((bb_rows, bb_cols), default_code, dtype=np.int32)

    # Sort classes by priority (highest priority last so they overwrite lower priority)
    # Lower priority number = higher priority = should be drawn last
    sorted_classes = sorted(unique_classes, key=lambda c: class_priority.get(c, 999), reverse=True)

    # Rasterize each class in priority order (lowest priority first, highest priority last overwrites)
    for lc_class in sorted_classes:
        if lc_class == default_class:
            continue  # Already filled as default
        
        class_gdf = gdf[gdf['class'] == lc_class]
        if class_gdf.empty:
            continue
        
        # Get all geometries for this class
        geometries = class_gdf.geometry.tolist()
        
        # Filter out invalid geometries and fix them
        valid_geometries = []
        for geom in geometries:
            if geom is None or geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_valid and not geom.is_empty:
                valid_geometries.append(geom)
        
        if not valid_geometries:
            continue
        
        # Create shapes for rasterization: (geometry, value) pairs
        class_code = class_to_code[lc_class]
        shapes = [(geom, class_code) for geom in valid_geometries]
        
        # Rasterize this class onto the bounding-box grid (overwrites previous values)
        try:
            features.rasterize(
                shapes=shapes,
                out=bb_grid,
                transform=transform,
                all_touched=False,  # Only cells whose center is inside
            )
        except Exception:
            # Fallback: try each geometry individually
            for geom, val in shapes:
                try:
                    features.rasterize(
                        shapes=[(geom, val)],
                        out=bb_grid,
                        transform=transform,
                        all_touched=False,
                    )
                except Exception:
                    continue

    # Apply ocean detection on the bounding-box grid BEFORE resampling
    if detect_ocean:
        try:
            from ...downloader.ocean import get_land_polygon_for_area, get_ocean_class_for_source
            
            ocean_class = get_ocean_class_for_source(source)
            if ocean_class not in class_to_code:
                class_to_code[ocean_class] = len(class_to_code)
                code_to_class[class_to_code[ocean_class]] = ocean_class
            ocean_code = class_to_code[ocean_class]
            
            # Use provided land_polygon or query from coastlines if not provided
            if land_polygon == "NOT_PROVIDED":
                land_polygon = get_land_polygon_for_area(rectangle_vertices, use_cache=False)
            
            if land_polygon is not None:
                land_mask = np.zeros((bb_rows, bb_cols), dtype=np.uint8)
                
                try:
                    if land_polygon.geom_type == 'Polygon':
                        land_geometries = [(land_polygon, 1)]
                    else:  # MultiPolygon
                        land_geometries = [(geom, 1) for geom in land_polygon.geoms]
                    
                    features.rasterize(
                        shapes=land_geometries,
                        out=land_mask,
                        transform=transform,
                        all_touched=False
                    )
                    
                    ocean_cells = (land_mask == 0) & (bb_grid == default_code)
                    ocean_count = np.sum(ocean_cells)
                    
                    if ocean_count > 0:
                        bb_grid[ocean_cells] = ocean_code
                        pct = 100 * ocean_count / bb_grid.size
                        print(f"  Ocean detection: {ocean_count:,} cells ({pct:.1f}%) classified as '{ocean_class}'")
                        
                except Exception as e:
                    print(f"  Warning: Ocean rasterization failed: {e}")
            else:
                from ...downloader.ocean import check_if_area_is_ocean_via_land_features
                is_ocean = check_if_area_is_ocean_via_land_features(rectangle_vertices)
                if is_ocean:
                    ocean_cells = (bb_grid == default_code)
                    ocean_count = np.sum(ocean_cells)
                    if ocean_count > 0:
                        bb_grid[ocean_cells] = ocean_code
                        pct = 100 * ocean_count / bb_grid.size
                        print(f"  Ocean detection: {ocean_count:,} cells ({pct:.1f}%) classified as '{ocean_class}' (open ocean)")
                        
        except Exception as e:
            print(f"  Warning: Ocean detection failed: {e}")

    # Sample the bounding-box grid at rotated cell centre coordinates
    cc = compute_cell_center_coords(rectangle_vertices, meshsize)
    center_lons = cc["lons"].ravel()
    center_lats = cc["lats"].ravel()

    col_idx = np.clip(((center_lons - min_lon) / pixel_width).astype(int), 0, bb_cols - 1)
    row_idx = np.clip(((max_lat - center_lats) / pixel_height).astype(int), 0, bb_rows - 1)

    grid_int = bb_grid[row_idx, col_idx].reshape(rows, cols)

    # Convert integer codes back to class names
    grid = np.empty((rows, cols), dtype=object)
    for code, cls_name in code_to_class.items():
        grid[grid_int == code] = cls_name

    return grid




