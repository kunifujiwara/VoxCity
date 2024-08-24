import numpy as np
import math
from shapely.geometry import Polygon
from scipy.ndimage import label, generate_binary_structure


def apply_operation(arr, meshsize):
    step1 = arr / meshsize
    step2 = step1 + 0.5
    step3 = np.floor(step2)
    return step3 * meshsize

def translate_array(input_array, translation_dict):
    translated_array = np.empty_like(input_array, dtype=object)
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            value = input_array[i, j]
            translated_array[i, j] = translation_dict.get(value, '')
    return translated_array

def group_and_label_cells(input_array):
    binary_array = input_array > 0
    structure = generate_binary_structure(2, 1)
    labeled_array, num_features = label(binary_array, structure=structure)
    result = np.zeros_like(input_array, dtype=int)
    for i in range(1, num_features + 1):
        result[labeled_array == i] = i
    return result

def process_grid(grid_bi, dem_grid):
    unique_ids = np.unique(grid_bi[grid_bi != 0])
    result = dem_grid.copy()
    for id_num in unique_ids:
        mask = (grid_bi == id_num)
        avg_value = np.mean(dem_grid[mask])
        result[mask] = avg_value
    return result - np.min(result)

def calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize):
    grid_size_0 = math.ceil(np.linalg.norm(side_1) / np.linalg.norm(meshsize * u_vec))
    grid_size_1 = math.ceil(np.linalg.norm(side_2) / np.linalg.norm(meshsize * v_vec))
    return (grid_size_0, grid_size_1)

def create_coordinate_mesh(origin, grid_size, meshsize, u_vec, v_vec):
    x = np.linspace(0, grid_size[0], grid_size[0])
    y = np.linspace(0, grid_size[1], grid_size[1])
    xx, yy = np.meshgrid(x, y)

    cell_coords = origin[:, np.newaxis, np.newaxis] + \
                  xx[np.newaxis, :, :] * meshsize * u_vec[:, np.newaxis, np.newaxis] + \
                  yy[np.newaxis, :, :] * meshsize * v_vec[:, np.newaxis, np.newaxis]

    return cell_coords

def create_cell_polygon(origin, i, j, meshsize, u_vec, v_vec):
    bottom_left = origin + i * meshsize * u_vec + j * meshsize * v_vec
    bottom_right = origin + (i + 1) * meshsize * u_vec + j * meshsize * v_vec
    top_right = origin + (i + 1) * meshsize * u_vec + (j + 1) * meshsize * v_vec
    top_left = origin + i * meshsize * u_vec + (j + 1) * meshsize * v_vec
    return Polygon([bottom_left, bottom_right, top_right, top_left])

def tree_height_grid_from_land_cover(land_cover_grid_ori):

    land_cover_grid = np.flipud(land_cover_grid_ori) + 1

    tree_translation_dict = {
        1: 0,
        2: 0,
        3: 0,
        4: 10,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0
    }
    tree_height_grid = translate_array(np.flipud(land_cover_grid), tree_translation_dict).astype(int)

    return tree_height_grid