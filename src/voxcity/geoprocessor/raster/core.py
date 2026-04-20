import warnings
import numpy as np
from typing import List, Tuple, Dict, Any
from shapely.geometry import Polygon, box

from ..utils import initialize_geod, calculate_distance, normalize_to_one_meter


def apply_operation(arr: np.ndarray, meshsize: float) -> np.ndarray:
    """
    Applies a sequence of operations to an array based on a mesh size to normalize and discretize values.

    1) Divide by meshsize, 2) +0.5, 3) floor, 4) rescale by meshsize
    """
    step1 = arr / meshsize
    step2 = step1 + 0.5
    step3 = np.floor(step2)
    return step3 * meshsize


def translate_array(input_array: np.ndarray, translation_dict: Dict[Any, Any]) -> np.ndarray:
    """
    Translate values in an array using a dictionary mapping (vectorized).

    Any value not found in the mapping is replaced with an empty string.
    Returns an object-dtype ndarray preserving the input shape.
    """
    if not isinstance(input_array, np.ndarray):
        input_array = np.asarray(input_array)

    getter = np.vectorize(lambda v: translation_dict.get(v, ''), otypes=[object])
    return getter(input_array)


def group_and_label_cells(array: np.ndarray) -> np.ndarray:
    """
    Convert non-zero numbers in a 2D numpy array to sequential IDs starting from 1.
    """
    result = array.copy()
    unique_values = sorted(set(array.flatten()) - {0})
    value_to_id = {value: idx + 1 for idx, value in enumerate(unique_values)}
    for value in unique_values:
        result[array == value] = value_to_id[value]
    return result


def process_grid_optimized(grid_bi: np.ndarray, dem_grid: np.ndarray) -> np.ndarray:
    """
    Optimized version that computes per-building averages without allocating huge arrays
    when building IDs are large and sparse.
    """
    result = dem_grid.copy()
    if np.any(grid_bi != 0):
        if grid_bi.dtype.kind == 'f':
            grid_bi_int = np.nan_to_num(grid_bi, nan=0).astype(np.int64)
        else:
            grid_bi_int = grid_bi.astype(np.int64)

        flat_ids = grid_bi_int.ravel()
        flat_dem = dem_grid.ravel()
        nz_mask = flat_ids != 0
        if np.any(nz_mask):
            ids_nz = flat_ids[nz_mask]
            vals_nz = flat_dem[nz_mask]
            unique_ids, inverse_idx = np.unique(ids_nz, return_inverse=True)
            sums = np.bincount(inverse_idx, weights=vals_nz)
            counts = np.bincount(inverse_idx)
            counts[counts == 0] = 1
            means = sums / counts
            result.ravel()[nz_mask] = means[inverse_idx]
    return result - np.min(result)


def process_grid(grid_bi: np.ndarray, dem_grid: np.ndarray) -> np.ndarray:
    """
    Safe version that tries optimization first, then falls back to original method.
    """
    try:
        return process_grid_optimized(grid_bi, dem_grid)
    except Exception as e:
        print(f"Optimized process_grid failed: {e}, using original method")
        unique_ids = np.unique(grid_bi[grid_bi != 0])
        result = dem_grid.copy()
        for id_num in unique_ids:
            mask = (grid_bi == id_num)
            avg_value = np.mean(dem_grid[mask])
            result[mask] = avg_value
        return result - np.min(result)


def calculate_grid_size(
    side_1: np.ndarray,
    side_2: np.ndarray,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    meshsize: float
) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    """
    Calculate grid size and adjusted mesh size based on input parameters.
    Returns ((nx, ny), (dx, dy))
    """
    dist_side_1_m = np.linalg.norm(side_1) / (np.linalg.norm(u_vec) + 1e-12)
    dist_side_2_m = np.linalg.norm(side_2) / (np.linalg.norm(v_vec) + 1e-12)

    grid_size_0 = max(1, int(dist_side_1_m / meshsize + 0.5))
    grid_size_1 = max(1, int(dist_side_2_m / meshsize + 0.5))

    adjusted_mesh_size_0 = dist_side_1_m / grid_size_0
    adjusted_mesh_size_1 = dist_side_2_m / grid_size_1

    return (grid_size_0, grid_size_1), (adjusted_mesh_size_0, adjusted_mesh_size_1)


def create_coordinate_mesh(
    origin: np.ndarray,
    grid_size: Tuple[int, int],
    adjusted_meshsize: Tuple[float, float],
    u_vec: np.ndarray,
    v_vec: np.ndarray
) -> np.ndarray:
    """
    Create a coordinate mesh based on input parameters.
    Returns array of shape (coord_dim, ny, nx)
    """
    x = np.linspace(0, grid_size[0], grid_size[0])
    y = np.linspace(0, grid_size[1], grid_size[1])
    xx, yy = np.meshgrid(x, y)
    cell_coords = origin[:, np.newaxis, np.newaxis] + \
                  xx[np.newaxis, :, :] * adjusted_meshsize[0] * u_vec[:, np.newaxis, np.newaxis] + \
                  yy[np.newaxis, :, :] * adjusted_meshsize[1] * v_vec[:, np.newaxis, np.newaxis]
    return cell_coords


def create_cell_polygon(
    origin: np.ndarray,
    i: int,
    j: int,
    adjusted_meshsize: Tuple[float, float],
    u_vec: np.ndarray,
    v_vec: np.ndarray
):
    """
    Create a polygon representing a grid cell.
    """
    bottom_left = origin + i * adjusted_meshsize[0] * u_vec + j * adjusted_meshsize[1] * v_vec
    bottom_right = origin + (i + 1) * adjusted_meshsize[0] * u_vec + j * adjusted_meshsize[1] * v_vec
    top_right = origin + (i + 1) * adjusted_meshsize[0] * u_vec + (j + 1) * adjusted_meshsize[1] * v_vec
    top_left = origin + i * adjusted_meshsize[0] * u_vec + (j + 1) * adjusted_meshsize[1] * v_vec
    return Polygon([bottom_left, bottom_right, top_right, top_left])


def compute_grid_geometry(rectangle_vertices, meshsize: float) -> dict:
    """
    Compute full grid geometry from rectangle vertices and mesh size.

    Returns a dict with keys: origin, side_1, side_2, u_vec, v_vec,
    grid_size, adj_mesh — or *None* if inputs are insufficient.
    """
    if rectangle_vertices is None or len(rectangle_vertices) < 4:
        return None

    v0 = np.array(rectangle_vertices[0])
    v1 = np.array(rectangle_vertices[1])
    v3 = np.array(rectangle_vertices[3])
    side_1 = v1 - v0
    side_2 = v3 - v0

    geod = initialize_geod()
    dist_1 = calculate_distance(geod, v0[0], v0[1], v1[0], v1[1])
    dist_2 = calculate_distance(geod, v0[0], v0[1], v3[0], v3[1])
    u_vec = normalize_to_one_meter(side_1, dist_1)
    v_vec = normalize_to_one_meter(side_2, dist_2)
    grid_size, adj_mesh = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)

    return {
        "origin": v0,
        "side_1": side_1,
        "side_2": side_2,
        "u_vec": u_vec,
        "v_vec": v_vec,
        "grid_size": grid_size,
        "adj_mesh": adj_mesh,
    }


def compute_cell_center_coords(rectangle_vertices, meshsize: float) -> dict:
    """
    Compute (lon, lat) coordinates for each grid cell center.

    Works correctly for both axis-aligned and rotated rectangles.
    Cell (i, j) centre = origin + (i+0.5)*dx*u_vec + (j+0.5)*dy*v_vec.

    Returns a dict with all keys from :func:`compute_grid_geometry` plus:
    - ``lons`` : ndarray of shape *grid_size* with cell-centre longitudes
    - ``lats`` : ndarray of shape *grid_size* with cell-centre latitudes

    Returns *None* when *rectangle_vertices* is insufficient.
    """
    geom = compute_grid_geometry(rectangle_vertices, meshsize)
    if geom is None:
        return None

    origin = geom["origin"]
    u_vec = geom["u_vec"]
    v_vec = geom["v_vec"]
    nx, ny = geom["grid_size"]
    dx, dy = geom["adj_mesh"]

    ii, jj = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="ij")
    lons = origin[0] + ii * dx * u_vec[0] + jj * dy * v_vec[0]
    lats = origin[1] + ii * dx * u_vec[1] + jj * dy * v_vec[1]

    return {**geom, "lons": lons, "lats": lats}


def compute_grid_shape(rectangle_vertices, meshsize: float) -> Tuple[int, int]:
    """
    Compute the grid dimensions (rows, cols) for a given rectangle and mesh size.
    
    This is useful when you need to know the output grid shape without
    actually creating the grid (e.g., for pre-allocating arrays or fallback shapes).
    
    Args:
        rectangle_vertices: List of 4 vertices [(lon, lat), ...] defining the rectangle.
        meshsize: Grid cell size in meters.
        
    Returns:
        Tuple of (grid_size_0, grid_size_1) representing grid dimensions.
    """
    geom = compute_grid_geometry(rectangle_vertices, meshsize)
    if geom is None:
        return (1, 1)
    return geom["grid_size"]


def normalize_gdf_crs(gdf, assume_epsg: int = 4326):
    """Return *gdf* re-projected to *assume_epsg*.

    If the GeoDataFrame has no CRS, EPSG:4326 is assumed with a warning.
    """
    if gdf.crs is None:
        warnings.warn(
            f"GeoDataFrame has no CRS. Assuming EPSG:{assume_epsg}.",
            UserWarning, stacklevel=2,
        )
        return gdf.set_crs(epsg=assume_epsg)
    if gdf.crs.to_epsg() != assume_epsg:
        return gdf.to_crs(epsg=assume_epsg)
    return gdf


def bbox_from_vertices(rectangle_vertices: List[Tuple[float, float]]) -> "box":
    """Return a shapely box covering *rectangle_vertices* (lon, lat) pairs."""
    lons = [c[0] for c in rectangle_vertices]
    lats = [c[1] for c in rectangle_vertices]
    return box(min(lons), min(lats), max(lons), max(lats))




