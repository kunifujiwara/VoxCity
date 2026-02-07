"""
Tests for voxcity.simulator.solar.radiation Numba kernels.
Covers: compute_solar_irradiance_for_all_faces, _trace_direct_masked,
compute_solar_irradiance_for_all_faces_masked
"""

import numpy as np
import pytest

from voxcity.simulator.solar.radiation import (
    compute_solar_irradiance_for_all_faces,
    _trace_direct_masked,
    compute_solar_irradiance_for_all_faces_masked,
)


def _make_simple_scene():
    """Create a simple 10x10x10 voxel scene with ground."""
    grid = np.zeros((10, 10, 10), dtype=np.int32)
    grid[:, :, 0] = 1  # ground
    return grid


class TestTraceDirectMasked:
    def test_clear_path(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        origin = np.array([5.0, 5.0, 5.0])
        direction = np.array([0.0, 0.0, 1.0])
        blocked, T = _trace_direct_masked(is_tree, is_opaque, origin, direction, 0.5)
        assert blocked is False
        assert T == pytest.approx(1.0)

    def test_blocked_by_opaque(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque[5, 5, 7] = True  # Opaque above observer
        origin = np.array([5.0, 5.0, 5.0])
        direction = np.array([0.0, 0.0, 1.0])
        blocked, T = _trace_direct_masked(is_tree, is_opaque, origin, direction, 0.5)
        assert blocked is True

    def test_tree_attenuation(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        is_tree[5, 5, 6] = True
        origin = np.array([5.0, 5.0, 5.0])
        direction = np.array([0.0, 0.0, 1.0])
        blocked, T = _trace_direct_masked(is_tree, is_opaque, origin, direction, 0.5)
        assert T < 1.0

    def test_zero_direction(self):
        is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)
        origin = np.array([2.0, 2.0, 2.0])
        direction = np.array([0.0, 0.0, 0.0])
        blocked, T = _trace_direct_masked(is_tree, is_opaque, origin, direction, 0.5)
        assert blocked is False
        assert T == 1.0

    def test_diagonal_direction(self):
        is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        origin = np.array([2.0, 2.0, 2.0])
        direction = np.array([1.0, 1.0, 1.0])
        blocked, T = _trace_direct_masked(is_tree, is_opaque, origin, direction, 0.5)
        assert blocked is False
        assert T == pytest.approx(1.0)


class TestComputeSolarIrradianceForAllFaces:
    def test_basic(self):
        grid = _make_simple_scene()
        n_faces = 4
        face_centers = np.array([
            [5.0, 5.0, 5.0],
            [3.0, 3.0, 5.0],
            [7.0, 7.0, 5.0],
            [5.0, 5.0, 3.0],
        ], dtype=np.float64)
        face_normals = np.array([
            [0.0, 0.0, 1.0],  # Upward
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],  # Side-facing
        ], dtype=np.float64)
        face_svf = np.array([0.8, 0.9, 0.7, 0.5], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        hit_values = np.array([0], dtype=np.int32)
        grid_bounds = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float64)

        direct, diffuse, glob = compute_solar_irradiance_for_all_faces(
            face_centers, face_normals, face_svf,
            sun_dir, 800.0, 200.0,
            grid, 1.0, 0.6, 1.0,
            hit_values, False,
            grid_bounds, 0.5
        )
        assert direct.shape == (n_faces,)
        assert diffuse.shape == (n_faces,)
        assert glob.shape == (n_faces,)

    def test_boundary_faces_excluded(self):
        grid = _make_simple_scene()
        # Face on the boundary edge
        face_centers = np.array([[0.0, 5.0, 5.0]], dtype=np.float64)
        face_normals = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)  # Vertical, on x_min
        face_svf = np.array([0.5], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        hit_values = np.array([0], dtype=np.int32)
        grid_bounds = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float64)

        direct, diffuse, glob = compute_solar_irradiance_for_all_faces(
            face_centers, face_normals, face_svf,
            sun_dir, 800.0, 200.0,
            grid, 1.0, 0.6, 1.0,
            hit_values, False,
            grid_bounds, 0.5
        )
        assert np.isnan(direct[0])


class TestComputeSolarIrradianceForAllFacesMasked:
    def test_basic(self):
        grid = _make_simple_scene()
        is_tree = (grid == -2)
        is_opaque = (grid == -3)
        n_faces = 2
        face_centers = np.array([
            [5.0, 5.0, 5.0],
            [3.0, 3.0, 5.0],
        ], dtype=np.float64)
        face_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        face_svf = np.array([0.8, 0.9], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        att = np.exp(-0.6 * 1.0)

        direct, diffuse, glob = compute_solar_irradiance_for_all_faces_masked(
            face_centers, face_normals, face_svf,
            sun_dir, 800.0, 200.0,
            is_tree, is_opaque,
            1.0, att,
            0.0, 0.0, 0.0,
            10.0, 10.0, 10.0,
            0.5
        )
        assert direct.shape == (n_faces,)
        assert diffuse.shape == (n_faces,)
        # With sun overhead and clear sky, direct should be positive for upward faces
        assert direct[0] >= 0.0

    def test_nan_svf(self):
        """NaN SVF should still produce a result (function handles it gracefully)."""
        grid = _make_simple_scene()
        is_tree = (grid == -2)
        is_opaque = (grid == -3)
        face_centers = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        face_svf = np.array([np.nan], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        direct, diffuse, glob = compute_solar_irradiance_for_all_faces_masked(
            face_centers, face_normals, face_svf,
            sun_dir, 800.0, 200.0,
            is_tree, is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.5
        )
        # Function should still return arrays of the right shape
        assert direct.shape == (1,)
        assert diffuse.shape == (1,)
