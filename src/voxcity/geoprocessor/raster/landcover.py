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
    """
    land_cover_grid = np.flipud(land_cover_grid_ori) + 1
    tree_translation_dict = {1: 0, 2: 0, 3: 0, 4: 10, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
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

        row, col = src.index(xs_flat, ys_flat)
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
    """Create a grid of land cover classes from GeoDataFrame polygon data."""
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.errors import GEOSException
    from rtree import index

    class_priority = get_class_priority(source)

    from ..utils import (
        initialize_geod,
        calculate_distance,
        normalize_to_one_meter,
    )
    from .core import calculate_grid_size, create_cell_polygon

    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]
    dist_side_1 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1])
    dist_side_2 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_3[0], vertex_3[1])

    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)

    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)

    grid = np.full(grid_size, default_class, dtype=object)

    extent = [min(coord[1] for coord in rectangle_vertices), max(coord[1] for coord in rectangle_vertices),
              min(coord[0] for coord in rectangle_vertices), max(coord[0] for coord in rectangle_vertices)]
    plotting_box = box(extent[2], extent[0], extent[3], extent[1])

    land_cover_polygons = []
    idx = index.Index()
    for i, row in gdf.iterrows():
        polygon = row.geometry
        land_cover_class = row['class']
        land_cover_polygons.append((polygon, land_cover_class))
        idx.insert(i, polygon.bounds)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            land_cover_class = default_class
            cell = create_cell_polygon(origin, i, j, adjusted_meshsize, u_vec, v_vec)
            for k in idx.intersection(cell.bounds):
                polygon, land_cover_class_temp = land_cover_polygons[k]
                try:
                    if cell.intersects(polygon):
                        intersection = cell.intersection(polygon)
                        if intersection.area > cell.area / 2:
                            rank = class_priority[land_cover_class]
                            rank_temp = class_priority[land_cover_class_temp]
                            if rank_temp < rank:
                                land_cover_class = land_cover_class_temp
                                grid[i, j] = land_cover_class
                except GEOSException as e:
                    try:
                        fixed_polygon = polygon.buffer(0)
                        if cell.intersects(fixed_polygon):
                            intersection = cell.intersection(fixed_polygon)
                            if intersection.area > cell.area / 2:
                                rank = class_priority[land_cover_class]
                                rank_temp = class_priority[land_cover_class_temp]
                                if rank_temp < rank:
                                    land_cover_class = land_cover_class_temp
                                    grid[i, j] = land_cover_class
                    except Exception:
                        continue
    return grid




