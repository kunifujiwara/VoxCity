"""
Tests for radiation.py njit helpers:
  - _trace_direct_masked (lines 303-361)
  - compute_solar_irradiance_for_all_faces_masked boundary/svf-nan branches
  - compute_cumulative_solar_irradiance_faces_masked_timeseries
"""
import numpy as np
import pytest


class TestTraceDirectMasked:
    """Test _trace_direct_masked njit function."""

    def test_clear_path(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)
        origin = np.array([2.0, 2.0, 2.0])
        direction = np.array([0.0, 0.0, 1.0])
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, 0.8)
        assert not blocked
        assert T == 1.0

    def test_opaque_block(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_is_opaque[2, 2, 4] = True  # block above
        origin = np.array([2.0, 2.0, 2.0])
        direction = np.array([0.0, 0.0, 1.0])
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, 0.8)
        assert blocked

    def test_tree_attenuation(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        # Place several tree voxels
        for k in range(3, 8):
            vox_is_tree[5, 5, k] = True
        origin = np.array([5.0, 5.0, 1.0])
        direction = np.array([0.0, 0.0, 1.0])
        att = 0.9  # high transmittance per voxel
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, att)
        # Should pass through with attenuation
        assert T < 1.0

    def test_zero_direction(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((3, 3, 3), dtype=np.bool_)
        vox_is_opaque = np.zeros((3, 3, 3), dtype=np.bool_)
        origin = np.array([1.0, 1.0, 1.0])
        direction = np.array([0.0, 0.0, 0.0])
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, 0.8)
        assert not blocked
        assert T == 1.0

    def test_diagonal_ray(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        origin = np.array([1.0, 1.0, 1.0])
        direction = np.array([1.0, 1.0, 1.0])
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, 0.8)
        assert not blocked


class TestComputeSolarIrradianceFacesMasked:
    """Test compute_solar_irradiance_for_all_faces_masked with boundary/NaN cases."""

    def test_boundary_face_nan(self):
        """Vertical face on boundary should get NaN."""
        from voxcity.simulator.solar.radiation import compute_solar_irradiance_for_all_faces_masked
        face_centers = np.array([[0.0, 2.5, 2.0]], dtype=np.float64)  # on x_min boundary
        face_normals = np.array([[-1.0, 0.0, 0.0]], dtype=np.float64)  # vertical
        face_svf = np.array([0.5], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        vox_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_opaque = np.zeros((5, 5, 5), dtype=np.bool_)

        fd, ff, fg = compute_solar_irradiance_for_all_faces_masked(
            face_centers, face_normals, face_svf, sun_dir,
            800.0, 100.0, vox_tree, vox_opaque, 1.0, 0.5,
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.05
        )
        assert np.isnan(fd[0])
        assert np.isnan(fg[0])

    def test_nan_svf_face(self):
        """Face with NaN SVF: diffuse and global are NaN, direct may still be computed."""
        from voxcity.simulator.solar.radiation import compute_solar_irradiance_for_all_faces_masked
        face_centers = np.array([[2.5, 2.5, 2.0]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        face_svf = np.array([np.nan], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        vox_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_opaque = np.zeros((5, 5, 5), dtype=np.bool_)

        fd, ff, fg = compute_solar_irradiance_for_all_faces_masked(
            face_centers, face_normals, face_svf, sun_dir,
            800.0, 100.0, vox_tree, vox_opaque, 1.0, 0.5,
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.05
        )
        # NaN SVF -> diffuse is NaN, global is NaN
        assert np.isnan(ff[0])
        assert np.isnan(fg[0])

    def test_upward_face_clear_sky(self):
        """Upward face with clear sky -> direct + diffuse."""
        from voxcity.simulator.solar.radiation import compute_solar_irradiance_for_all_faces_masked
        face_centers = np.array([[2.5, 2.5, 2.0]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        face_svf = np.array([0.8], dtype=np.float64)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        vox_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_opaque = np.zeros((5, 5, 5), dtype=np.bool_)

        fd, ff, fg = compute_solar_irradiance_for_all_faces_masked(
            face_centers, face_normals, face_svf, sun_dir,
            800.0, 100.0, vox_tree, vox_opaque, 1.0, 0.5,
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.05
        )
        assert fd[0] > 0  # direct
        assert ff[0] == pytest.approx(80.0)  # 0.8 * 100
        assert fg[0] > 80.0  # global = direct + diffuse


class TestCumulativeTimeseries:
    """Test compute_cumulative_solar_irradiance_faces_masked_timeseries."""

    def test_basic_cumulative(self):
        from voxcity.simulator.solar.radiation import compute_cumulative_solar_irradiance_faces_masked_timeseries
        n_faces = 2
        T = 3
        face_centers = np.array([[2.5, 2.5, 2.0], [2.5, 2.5, 3.0]], dtype=np.float64)
        face_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        face_svf = np.array([0.8, 0.5], dtype=np.float64)
        sun_dirs = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64)
        DNI = np.array([800.0, 600.0, 0.0], dtype=np.float64)
        DHI = np.array([100.0, 80.0, 50.0], dtype=np.float64)
        vox_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_opaque = np.zeros((5, 5, 5), dtype=np.bool_)

        out_dir, out_diff, out_glob = compute_cumulative_solar_irradiance_faces_masked_timeseries(
            face_centers, face_normals, face_svf,
            sun_dirs, DNI, DHI,
            vox_tree, vox_opaque, 1.0, 0.5,
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.05,
            0, T, 1.0
        )
        assert out_dir.shape == (n_faces,)
        assert out_diff.shape == (n_faces,)
        assert out_glob.shape == (n_faces,)
        # Third timestep has sun below horizon -> diffuse only
        assert out_diff[0] > 0
