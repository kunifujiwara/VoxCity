import pytest
import numpy as np
from voxcity.geo.grid import (
    apply_operation,
    translate_array,
    group_and_label_cells,
    process_grid
)

def test_apply_operation():
    test_array = np.array([[1.2, 2.7], [3.4, 4.8]])
    meshsize = 1.0
    result = apply_operation(test_array, meshsize)
    assert isinstance(result, np.ndarray)
    assert result.shape == test_array.shape

def test_translate_array():
    input_array = np.array([[1, 2], [3, 4]])
    translation_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    result = translate_array(input_array, translation_dict)
    assert result.shape == input_array.shape
    assert result[0, 0] == 'A'

def test_group_and_label_cells():
    input_array = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    result = group_and_label_cells(input_array)
    assert result.shape == input_array.shape
    assert np.max(result) <= np.sum(input_array > 0)

def test_process_grid():
    grid_bi = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    dem_grid = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    result = process_grid(grid_bi, dem_grid)
    assert isinstance(result, np.ndarray)
    assert result.shape == grid_bi.shape 