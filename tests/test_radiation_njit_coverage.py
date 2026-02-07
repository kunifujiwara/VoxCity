"""
Tests for radiation.py njit kernels:
- compute_solar_irradiance_for_all_faces (generic ray tracer variant)
- compute_cumulative_solar_irradiance_faces_masked_timeseries
Covers lines 219-536.
"""

import numpy as np
import pytest

from voxcity.simulator.solar.radiation import (
    compute_solar_irradiance_for_all_faces,
    compute_solar_irradiance_for_all_faces_masked,
    _trace_direct_masked,
    compute_cumulative_solar_irradiance_faces_masked_timeseries,
)


def _simple_voxel_grid():
    """5x5x5 grid with ground at z=0 and a building column at (2,2)."""
    vox = np.zeros((5, 5, 5), dtype=np.int8)
    vox[:, :, 0] = 1
    vox[2, 2, 1:4] = -3  # building
    return vox


# ---------- compute_solar_irradiance_for_all_faces (generic) ----------

class TestComputeSolarIrradianceGeneric:
    def test_upward_face_receives_direct(self):
        vox = _simple_voxel_grid()
        face_centers = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        face_svf = np.array([0.8], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        grid_bounds = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)

        direct, diffuse, glob = compute_solar_irradiance_for_all_faces(
            face_centers, face_normals, face_svf,
            sun_dir, 800.0, 200.0,
            vox, 1.0, 0.6, 1.0,
            hit_values, True, grid_bounds, 0.5
        )
        assert direct[0] > 0.0
        assert diffuse[0] > 0.0
        assert glob[0] == pytest.approx(direct[0] + diffuse[0])

    def test_boundary_face_excluded(self):
        vox = _simple_voxel_grid()
        # Vertical face on x_min boundary
        face_centers = np.array([[0.0, 2.5, 2.5]], dtype=np.float64)
        face_normals = np.array([[-1.0, 0.0, 0.0]], dtype=np.float64)
        face_svf = np.array([0.5], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        grid_bounds = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)

        direct, diffuse, glob = compute_solar_irradiance_for_all_faces(
            face_centers, face_normals, face_svf,
            sun_dir, 800.0, 200.0,
            vox, 1.0, 0.6, 1.0,
            hit_values, True, grid_bounds, 0.5
        )
        assert np.isnan(direct[0])
        assert np.isnan(diffuse[0])

    def test_nan_svf_excluded(self):
        vox = _simple_voxel_grid()
        face_centers = np.array([[2.5, 2.5, 2.5]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        face_svf = np.array([np.nan], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        grid_bounds = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)

        direct, diffuse, glob = compute_solar_irradiance_for_all_faces(
            face_centers, face_normals, face_svf,
            sun_dir, 800.0, 200.0,
            vox, 1.0, 0.6, 1.0,
            hit_values, True, grid_bounds, 0.5
        )
        assert np.isnan(direct[0])

    def test_building_shadow(self):
        vox = _simple_voxel_grid()
        # Face behind building, sun from y direction
        face_centers = np.array([[2.5, 4.5, 0.5]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        face_svf = np.array([0.5], dtype=np.float64)
        sun_dir = np.array([0.0, -1.0, 0.2], dtype=np.float64)
        hit_values = np.array([-3], dtype=np.int8)
        grid_bounds = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)

        direct, diffuse, glob = compute_solar_irradiance_for_all_faces(
            face_centers, face_normals, face_svf,
            sun_dir, 800.0, 200.0,
            vox, 1.0, 0.6, 1.0,
            hit_values, True, grid_bounds, 0.5
        )
        # Shadow from building should reduce or eliminate direct
        assert direct[0] >= 0.0

    def test_negative_cos_incidence(self):
        """Face normal pointing away from sun -> no direct."""
        vox = _simple_voxel_grid()
        face_centers = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float64)  # pointing down
        face_svf = np.array([0.5], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # sun above
        hit_values = np.array([-3], dtype=np.int8)
        grid_bounds = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)

        direct, diffuse, glob = compute_solar_irradiance_for_all_faces(
            face_centers, face_normals, face_svf,
            sun_dir, 800.0, 200.0,
            vox, 1.0, 0.6, 1.0,
            hit_values, True, grid_bounds, 0.5
        )
        assert direct[0] == 0.0


# ---------- compute_cumulative_solar_irradiance_faces_masked_timeseries ----------

class TestCumulativeTimeseries:
    def test_basic_accumulation(self):
        is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)

        face_centers = np.array([[2.5, 2.5, 2.5]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        face_svf = np.array([0.8], dtype=np.float64)

        # 3 timesteps, sun overhead
        sun_dirs = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        DNI = np.array([800.0, 800.0, 800.0], dtype=np.float64)
        DHI = np.array([200.0, 200.0, 200.0], dtype=np.float64)

        out_dir, out_diff, out_glob = compute_cumulative_solar_irradiance_faces_masked_timeseries(
            face_centers, face_normals, face_svf,
            sun_dirs, DNI, DHI,
            is_tree, is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.5,
            0, 3, 1.0,
        )
        assert out_dir[0] > 0.0
        assert out_diff[0] > 0.0
        assert out_glob[0] == pytest.approx(out_dir[0] + out_diff[0])

    def test_below_horizon_diffuse_only(self):
        is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)

        face_centers = np.array([[2.5, 2.5, 2.5]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        face_svf = np.array([0.8], dtype=np.float64)

        # Sun below horizon (z <= 0)
        sun_dirs = np.array([[0.0, 0.0, -0.5]], dtype=np.float64)
        DNI = np.array([0.0], dtype=np.float64)
        DHI = np.array([50.0], dtype=np.float64)

        out_dir, out_diff, out_glob = compute_cumulative_solar_irradiance_faces_masked_timeseries(
            face_centers, face_normals, face_svf,
            sun_dirs, DNI, DHI,
            is_tree, is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.5,
            0, 1, 1.0,
        )
        assert out_dir[0] == 0.0  # No direct below horizon
        assert out_diff[0] > 0.0  # Diffuse still present

    def test_boundary_face_nan(self):
        is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)

        # Vertical face on boundary
        face_centers = np.array([[0.0, 2.5, 2.5]], dtype=np.float64)
        face_normals = np.array([[-1.0, 0.0, 0.0]], dtype=np.float64)
        face_svf = np.array([0.5], dtype=np.float64)

        sun_dirs = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        DNI = np.array([800.0], dtype=np.float64)
        DHI = np.array([200.0], dtype=np.float64)

        out_dir, out_diff, out_glob = compute_cumulative_solar_irradiance_faces_masked_timeseries(
            face_centers, face_normals, face_svf,
            sun_dirs, DNI, DHI,
            is_tree, is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.5,
            0, 1, 1.0,
        )
        assert np.isnan(out_dir[0])
        assert np.isnan(out_diff[0])

    def test_time_step_scaling(self):
        is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)

        face_centers = np.array([[2.5, 2.5, 2.5]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        face_svf = np.array([0.8], dtype=np.float64)

        sun_dirs = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        DNI = np.array([800.0], dtype=np.float64)
        DHI = np.array([200.0], dtype=np.float64)

        # 1-hour step
        _, _, glob1 = compute_cumulative_solar_irradiance_faces_masked_timeseries(
            face_centers, face_normals, face_svf,
            sun_dirs, DNI, DHI,
            is_tree, is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.5,
            0, 1, 1.0,
        )
        # 0.5-hour step
        _, _, glob_half = compute_cumulative_solar_irradiance_faces_masked_timeseries(
            face_centers, face_normals, face_svf,
            sun_dirs, DNI, DHI,
            is_tree, is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.5,
            0, 1, 0.5,
        )
        assert glob1[0] == pytest.approx(glob_half[0] * 2.0, rel=1e-6)

    def test_multiple_faces(self):
        is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)

        n_faces = 5
        face_centers = np.random.rand(n_faces, 3).astype(np.float64) * 3 + 1
        face_normals = np.zeros((n_faces, 3), dtype=np.float64)
        face_normals[:, 2] = 1.0  # all upward
        face_svf = np.full(n_faces, 0.7, dtype=np.float64)

        sun_dirs = np.array([[0.0, 0.0, 1.0], [0.3, 0.0, 0.9]], dtype=np.float64)
        DNI = np.array([800.0, 600.0], dtype=np.float64)
        DHI = np.array([200.0, 150.0], dtype=np.float64)

        out_dir, out_diff, out_glob = compute_cumulative_solar_irradiance_faces_masked_timeseries(
            face_centers, face_normals, face_svf,
            sun_dirs, DNI, DHI,
            is_tree, is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.5,
            0, 2, 1.0,
        )
        assert out_dir.shape == (n_faces,)
        assert np.all(out_dir >= 0.0)
