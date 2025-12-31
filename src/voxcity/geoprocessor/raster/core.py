import numpy as np
from typing import Tuple, Dict, Any
from shapely.geometry import Polygon

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
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]
    
    dist_side_1 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1])
    dist_side_2 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_3[0], vertex_3[1])
    
    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)
    
    grid_size, _ = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)
    return grid_size




