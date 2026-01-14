import numpy as np
from typing import List, Tuple, Dict, Any
from shapely.geometry import Polygon
from affine import Affine
import rasterio

from pyproj import Geod

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
    land_cover_grid = np.flipud(land_cover_grid_ori)
    # 1-based indices: 1=Bareland, 2=Rangeland, 3=Shrub, 4=Agriculture, 5=Tree, etc.
    tree_translation_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 10, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0}
    tree_height_grid = translate_array(np.flipud(land_cover_grid), tree_translation_dict).astype(int)
    return tree_height_grid


def create_land_cover_grid_from_geotiff_polygon(
    tiff_path: str,
    mesh_size: float,
    land_cover_classes: Dict[str, Any],
    polygon: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Create a land cover grid from a GeoTIFF file within a polygon boundary.
    """
    with rasterio.open(tiff_path) as src:
        img = src.read((1, 2, 3))
        left, bottom, right, top = src.bounds
        poly = Polygon(polygon)
        left_wgs84, bottom_wgs84, right_wgs84, top_wgs84 = poly.bounds

        geod = Geod(ellps="WGS84")
        _, _, width = geod.inv(left_wgs84, bottom_wgs84, right_wgs84, bottom_wgs84)
        _, _, height = geod.inv(left_wgs84, bottom_wgs84, left_wgs84, top_wgs84)

        num_cells_x = int(width / mesh_size + 0.5)
        num_cells_y = int(height / mesh_size + 0.5)

        adjusted_mesh_size_x = (right - left) / num_cells_x
        adjusted_mesh_size_y = (top - bottom) / num_cells_y

        new_affine = Affine(adjusted_mesh_size_x, 0, left, 0, -adjusted_mesh_size_y, top)

        cols, rows = np.meshgrid(np.arange(num_cells_x), np.arange(num_cells_y))
        xs, ys = new_affine * (cols, rows)
        xs_flat, ys_flat = xs.flatten(), ys.flatten()

        row, col = rasterio.transform.rowcol(src.transform, xs_flat, ys_flat)
        row, col = np.array(row), np.array(col)

        valid = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        row, col = row[valid], col[valid]

        grid = np.full((num_cells_y, num_cells_x), 'No Data', dtype=object)
        for i, (r, c) in enumerate(zip(row, col)):
            cell_data = img[:, r, c]
            dominant_class = get_dominant_class(cell_data, land_cover_classes)
            grid_row, grid_col = np.unravel_index(i, (num_cells_y, num_cells_x))
            grid[grid_row, grid_col] = dominant_class

    return np.flipud(grid)


def create_land_cover_grid_from_gdf_polygon(
    gdf,
    meshsize: float,
    source: str,
    rectangle_vertices: List[Tuple[float, float]],
    default_class: str = 'Developed space'
) -> np.ndarray:
    """
    Create a grid of land cover classes from GeoDataFrame polygon data.
    
    Uses vectorized rasterization for ~100x speedup over cell-by-cell intersection.
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
    from .core import calculate_grid_size

    # Calculate grid dimensions
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]
    dist_side_1 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1])
    dist_side_2 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_3[0], vertex_3[1])

    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)

    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)
    rows, cols = grid_size

    # Get bounding box for the raster
    min_lon = min(coord[0] for coord in rectangle_vertices)
    max_lon = max(coord[0] for coord in rectangle_vertices)
    min_lat = min(coord[1] for coord in rectangle_vertices)
    max_lat = max(coord[1] for coord in rectangle_vertices)

    # Create affine transform (top-left origin, pixel size)
    pixel_width = (max_lon - min_lon) / cols
    pixel_height = (max_lat - min_lat) / rows
    transform = Affine(pixel_width, 0, min_lon, 0, -pixel_height, max_lat)

    # Build class name to priority mapping, then sort classes by priority (highest priority = lowest number = rasterize last)
    unique_classes = gdf['class'].unique().tolist()
    if default_class not in unique_classes:
        unique_classes.append(default_class)
    
    # Map class names to integer codes
    class_to_code = {cls: i for i, cls in enumerate(unique_classes)}
    code_to_class = {i: cls for cls, i in class_to_code.items()}
    default_code = class_to_code[default_class]

    # Initialize grid with default class code
    grid_int = np.full((rows, cols), default_code, dtype=np.int32)

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
        
        # Rasterize this class onto the grid (overwrites previous values)
        try:
            features.rasterize(
                shapes=shapes,
                out=grid_int,
                transform=transform,
                all_touched=False,  # Only cells whose center is inside
            )
        except Exception:
            # Fallback: try each geometry individually
            for geom, val in shapes:
                try:
                    features.rasterize(
                        shapes=[(geom, val)],
                        out=grid_int,
                        transform=transform,
                        all_touched=False,
                    )
                except Exception:
                    continue

    # Convert integer codes back to class names
    grid = np.empty((rows, cols), dtype=object)
    for code, cls_name in code_to_class.items():
        grid[grid_int == code] = cls_name

    # Flip to match expected orientation (north-up)
    return np.flipud(grid)




