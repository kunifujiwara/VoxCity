"""Tests for voxcity.geoprocessor.raster.core module."""
import pytest
import numpy as np


class TestApplyOperation:
    """Tests for apply_operation function."""

    def test_zero_array(self):
        """Zero values should remain zero."""
        from voxcity.geoprocessor.raster.core import apply_operation
        
        arr = np.zeros((3, 3))
        result = apply_operation(arr, meshsize=1.0)
        
        np.testing.assert_array_equal(result, np.zeros((3, 3)))

    def test_meshsize_scaling(self):
        """Values should be discretized by meshsize."""
        from voxcity.geoprocessor.raster.core import apply_operation
        
        arr = np.array([0.3, 0.6, 0.9, 1.2, 1.5])
        result = apply_operation(arr, meshsize=0.5)
        
        # Each value gets rounded to nearest meshsize
        # 0.3/0.5 + 0.5 = 1.1 -> floor = 1 -> 0.5
        # 0.6/0.5 + 0.5 = 1.7 -> floor = 1 -> 0.5
        # etc.
        assert all(r % 0.5 == pytest.approx(0, abs=1e-10) or 
                   r % 0.5 == pytest.approx(0.5, abs=1e-10) for r in result)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        from voxcity.geoprocessor.raster.core import apply_operation
        
        arr = np.random.rand(5, 7) * 10
        result = apply_operation(arr, meshsize=2.0)
        
        assert result.shape == arr.shape


class TestTranslateArray:
    """Tests for translate_array function."""

    def test_simple_translation(self):
        """Should translate values using dictionary."""
        from voxcity.geoprocessor.raster.core import translate_array
        
        arr = np.array([1, 2, 3])
        trans = {1: 'a', 2: 'b', 3: 'c'}
        result = translate_array(arr, trans)
        
        assert result[0] == 'a'
        assert result[1] == 'b'
        assert result[2] == 'c'

    def test_missing_key_returns_empty(self):
        """Values not in dict should become empty string."""
        from voxcity.geoprocessor.raster.core import translate_array
        
        arr = np.array([1, 2, 99])
        trans = {1: 'a', 2: 'b'}
        result = translate_array(arr, trans)
        
        assert result[0] == 'a'
        assert result[1] == 'b'
        assert result[2] == ''

    def test_empty_dict(self):
        """Empty dict should result in all empty strings."""
        from voxcity.geoprocessor.raster.core import translate_array
        
        arr = np.array([1, 2, 3])
        result = translate_array(arr, {})
        
        assert all(v == '' for v in result)

    def test_preserves_shape(self):
        """Output shape should match input shape."""
        from voxcity.geoprocessor.raster.core import translate_array
        
        arr = np.array([[1, 2], [3, 4]])
        trans = {1: 'a', 2: 'b', 3: 'c', 4: 'd'}
        result = translate_array(arr, trans)
        
        assert result.shape == arr.shape


class TestGroupAndLabelCells:
    """Tests for group_and_label_cells function."""

    def test_sequential_labeling(self):
        """Non-zero values should become sequential IDs."""
        from voxcity.geoprocessor.raster.core import group_and_label_cells
        
        arr = np.array([[10, 0, 20], [0, 10, 0], [20, 0, 30]])
        result = group_and_label_cells(arr)
        
        # Values 10, 20, 30 should become 1, 2, 3
        assert result[0, 0] == 1  # was 10
        assert result[0, 2] == 2  # was 20
        assert result[2, 2] == 3  # was 30
        assert result[1, 0] == 0  # zeros stay zero

    def test_zeros_preserved(self):
        """Zero values should remain zero."""
        from voxcity.geoprocessor.raster.core import group_and_label_cells
        
        arr = np.array([[0, 5, 0], [5, 0, 5]])
        result = group_and_label_cells(arr)
        
        assert result[0, 0] == 0
        assert result[0, 2] == 0
        assert result[1, 1] == 0

    def test_all_zeros(self):
        """All-zero array should remain all zeros."""
        from voxcity.geoprocessor.raster.core import group_and_label_cells
        
        arr = np.zeros((3, 3))
        result = group_and_label_cells(arr)
        
        np.testing.assert_array_equal(result, arr)

    def test_single_value(self):
        """Single non-zero value should become 1."""
        from voxcity.geoprocessor.raster.core import group_and_label_cells
        
        arr = np.array([[0, 0], [100, 0]])
        result = group_and_label_cells(arr)
        
        assert result[1, 0] == 1


class TestProcessGrid:
    """Tests for process_grid function."""

    def test_no_building_ids(self):
        """With all-zero building IDs, should return DEM minus minimum."""
        from voxcity.geoprocessor.raster.core import process_grid
        
        grid_bi = np.zeros((3, 3))
        dem_grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        result = process_grid(grid_bi, dem_grid)
        
        # Should subtract minimum (1)
        expected = dem_grid - 1
        np.testing.assert_array_almost_equal(result, expected)

    def test_building_averaging(self):
        """Building cells should have average DEM value."""
        from voxcity.geoprocessor.raster.core import process_grid
        
        # Building 1 covers two cells with DEM values 2 and 4
        grid_bi = np.array([[1, 0], [1, 0]], dtype=float)
        dem_grid = np.array([[2.0, 10.0], [4.0, 20.0]])
        
        result = process_grid(grid_bi, dem_grid)
        
        # Building 1 cells should have average of 2 and 4 = 3
        # After subtracting minimum
        building_vals = result[grid_bi == 1]
        assert building_vals[0] == building_vals[1]  # Same value for same building

    def test_multiple_buildings(self):
        """Multiple buildings should each have their own average."""
        from voxcity.geoprocessor.raster.core import process_grid
        
        grid_bi = np.array([[1, 2], [1, 2]], dtype=float)
        dem_grid = np.array([[10.0, 20.0], [30.0, 40.0]])
        
        result = process_grid(grid_bi, dem_grid)
        
        # Building 1: average of 10, 30 = 20
        # Building 2: average of 20, 40 = 30
        # After min subtraction, values should be 0 and 10
        building1_vals = result[grid_bi == 1]
        building2_vals = result[grid_bi == 2]
        
        assert building1_vals[0] == building1_vals[1]
        assert building2_vals[0] == building2_vals[1]


class TestCalculateGridSize:
    """Tests for calculate_grid_size function."""

    def test_returns_tuple_of_tuples(self):
        """Should return ((nx, ny), (dx, dy)) structure."""
        from voxcity.geoprocessor.raster.core import calculate_grid_size
        
        side_1 = np.array([100.0, 0.0])
        side_2 = np.array([0.0, 100.0])
        u_vec = np.array([1.0, 0.0])
        v_vec = np.array([0.0, 1.0])
        meshsize = 10.0
        
        result = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], tuple)
        assert isinstance(result[1], tuple)


class TestCreateCoordinateMesh:
    """Tests for create_coordinate_mesh function."""

    def test_returns_correct_shape(self):
        """Should return shape (coord_dim, ny, nx)."""
        from voxcity.geoprocessor.raster.core import create_coordinate_mesh
        
        origin = np.array([0.0, 0.0])
        grid_size = (5, 4)
        adjusted_meshsize = (1.0, 1.0)
        u_vec = np.array([1.0, 0.0])
        v_vec = np.array([0.0, 1.0])
        
        result = create_coordinate_mesh(origin, grid_size, adjusted_meshsize, u_vec, v_vec)
        
        # Shape should be (2, 4, 5) for 2D coordinates
        assert result.shape == (2, 4, 5)

    def test_origin_at_zero(self):
        """First cell should be near origin."""
        from voxcity.geoprocessor.raster.core import create_coordinate_mesh
        
        origin = np.array([0.0, 0.0])
        grid_size = (3, 3)
        adjusted_meshsize = (1.0, 1.0)
        u_vec = np.array([1.0, 0.0])
        v_vec = np.array([0.0, 1.0])
        
        result = create_coordinate_mesh(origin, grid_size, adjusted_meshsize, u_vec, v_vec)
        
        # Origin point should be at (0, 0)
        assert result[0, 0, 0] == pytest.approx(0.0)
        assert result[1, 0, 0] == pytest.approx(0.0)

    def test_meshsize_scaling(self):
        """Coordinates should scale with meshsize."""
        from voxcity.geoprocessor.raster.core import create_coordinate_mesh
        
        origin = np.array([0.0, 0.0])
        grid_size = (3, 3)
        u_vec = np.array([1.0, 0.0])
        v_vec = np.array([0.0, 1.0])
        
        result1 = create_coordinate_mesh(origin, grid_size, (1.0, 1.0), u_vec, v_vec)
        result2 = create_coordinate_mesh(origin, grid_size, (2.0, 2.0), u_vec, v_vec)
        
        # With 2x meshsize, coordinates should be 2x larger
        # For cell (0,1,0) - after one step in y direction
        assert result2[1, 1, 0] == pytest.approx(2 * result1[1, 1, 0])


class TestCreateCellPolygon:
    """Tests for create_cell_polygon function."""

    def test_returns_polygon(self):
        """Should return a Shapely Polygon."""
        from voxcity.geoprocessor.raster.core import create_cell_polygon
        from shapely.geometry import Polygon
        
        origin = np.array([0.0, 0.0])
        u_vec = np.array([1.0, 0.0])
        v_vec = np.array([0.0, 1.0])
        adjusted_meshsize = (1.0, 1.0)
        
        result = create_cell_polygon(origin, 0, 0, adjusted_meshsize, u_vec, v_vec)
        
        assert isinstance(result, Polygon)

    def test_cell_at_origin(self):
        """Cell at (0,0) should have correct corners."""
        from voxcity.geoprocessor.raster.core import create_cell_polygon
        
        origin = np.array([0.0, 0.0])
        u_vec = np.array([1.0, 0.0])
        v_vec = np.array([0.0, 1.0])
        adjusted_meshsize = (1.0, 1.0)
        
        result = create_cell_polygon(origin, 0, 0, adjusted_meshsize, u_vec, v_vec)
        
        # Should be a unit square at origin
        coords = list(result.exterior.coords)
        assert len(coords) == 5  # Closed polygon has 5 points
        assert result.area == pytest.approx(1.0)

    def test_cell_at_different_position(self):
        """Cell at (2,1) should be offset from origin."""
        from voxcity.geoprocessor.raster.core import create_cell_polygon
        
        origin = np.array([0.0, 0.0])
        u_vec = np.array([1.0, 0.0])
        v_vec = np.array([0.0, 1.0])
        adjusted_meshsize = (1.0, 1.0)
        
        result = create_cell_polygon(origin, 2, 1, adjusted_meshsize, u_vec, v_vec)
        
        # Centroid should be at (2.5, 1.5)
        centroid = result.centroid
        assert centroid.x == pytest.approx(2.5)
        assert centroid.y == pytest.approx(1.5)

    def test_rectangular_meshsize(self):
        """Should support rectangular (non-square) cells."""
        from voxcity.geoprocessor.raster.core import create_cell_polygon
        
        origin = np.array([0.0, 0.0])
        u_vec = np.array([1.0, 0.0])
        v_vec = np.array([0.0, 1.0])
        adjusted_meshsize = (2.0, 3.0)
        
        result = create_cell_polygon(origin, 0, 0, adjusted_meshsize, u_vec, v_vec)
        
        # Area should be 2 * 3 = 6
        assert result.area == pytest.approx(6.0)


class TestComputeGridShape:
    """Tests for compute_grid_shape function."""

    def test_returns_tuple(self):
        """Should return tuple of two ints."""
        from voxcity.geoprocessor.raster.core import compute_grid_shape
        
        # Rectangle approximately 100m x 100m
        rectangle_vertices = [
            (139.7000, 35.7000),
            (139.7010, 35.7000),
            (139.7010, 35.7010),
            (139.7000, 35.7010),
        ]
        
        result = compute_grid_shape(rectangle_vertices, meshsize=10.0)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_larger_meshsize_fewer_cells(self):
        """Larger meshsize should give fewer grid cells."""
        from voxcity.geoprocessor.raster.core import compute_grid_shape
        
        rectangle_vertices = [
            (139.7000, 35.7000),
            (139.7010, 35.7000),
            (139.7010, 35.7010),
            (139.7000, 35.7010),
        ]
        
        small_mesh = compute_grid_shape(rectangle_vertices, meshsize=1.0)
        large_mesh = compute_grid_shape(rectangle_vertices, meshsize=10.0)
        
        # Larger meshsize should give fewer cells
        assert large_mesh[0] < small_mesh[0]
        assert large_mesh[1] < small_mesh[1]

    def test_positive_dimensions(self):
        """Grid should have at least 1 cell in each dimension."""
        from voxcity.geoprocessor.raster.core import compute_grid_shape
        
        rectangle_vertices = [
            (139.7000, 35.7000),
            (139.7001, 35.7000),
            (139.7001, 35.7001),
            (139.7000, 35.7001),
        ]
        
        result = compute_grid_shape(rectangle_vertices, meshsize=1.0)
        
        assert result[0] >= 1
        assert result[1] >= 1
