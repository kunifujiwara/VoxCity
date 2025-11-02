import numpy as np

from voxcity.geoprocessor.raster import (
    apply_operation,
    translate_array,
    group_and_label_cells,
    process_grid,
)


def test_apply_operation_basic():
    test_array = np.array([[1.2, 2.7], [3.4, 4.8]])
    meshsize = 1.0
    result = apply_operation(test_array, meshsize)
    assert isinstance(result, np.ndarray)
    assert result.shape == test_array.shape


def test_apply_operation_zero():
    test_array = np.zeros((3, 3))
    meshsize = 1.0
    result = apply_operation(test_array, meshsize)
    assert np.array_equal(result, test_array)


def test_translate_array_empty_dict():
    input_array = np.array([[1, 2], [3, 4]])
    translation_dict = {}
    result = translate_array(input_array, translation_dict)
    assert result.shape == input_array.shape
    assert np.all(result == '')


def test_group_and_label_cells_empty():
    input_array = np.zeros((3, 3))
    result = group_and_label_cells(input_array)
    assert np.array_equal(result, input_array)


def test_process_grid():
    grid_bi = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    dem_grid = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    result = process_grid(grid_bi, dem_grid)
    assert isinstance(result, np.ndarray)
    assert result.shape == grid_bi.shape

