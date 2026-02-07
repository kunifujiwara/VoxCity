"""Tests for radiation.py njit functions: _trace_direct_masked, compute_solar_irradiance_for_all_faces_masked, compute_cumulative_solar_irradiance_faces_masked_timeseries."""
import numpy as np
import pytest


class TestTraceDirectMasked:
    """Direct tests of the _trace_direct_masked njit kernel."""

    def test_open_sky_no_hit(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)
        origin = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, 0.5)
        assert not blocked
        assert T == pytest.approx(1.0)

    def test_opaque_hit(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_is_opaque[2, 2, 4] = True
        origin = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, 0.5)
        assert blocked

    def test_tree_attenuates(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_tree[5, 5, 5] = True
        origin = np.array([5.0, 5.0, 3.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, 0.5)
        # One tree voxel with att=0.5 → T = 0.5
        assert T < 1.0

    def test_many_trees_cutoff(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        for k in range(3, 9):
            vox_is_tree[5, 5, k] = True
        origin = np.array([5.0, 5.0, 2.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # att=0.1 per voxel, cutoff=0.01 → after 2 voxels T=0.01, hits cutoff
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, 0.1, 0.01)
        assert blocked

    def test_zero_direction_returns_no_hit(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((5, 5, 5), dtype=np.bool_)
        vox_is_opaque = np.zeros((5, 5, 5), dtype=np.bool_)
        origin = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        direction = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, 0.5)
        assert not blocked
        assert T == pytest.approx(1.0)

    def test_diagonal_direction(self):
        from voxcity.simulator.solar.radiation import _trace_direct_masked
        vox_is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        origin = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        direction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        blocked, T = _trace_direct_masked(vox_is_tree, vox_is_opaque, origin, direction, 0.5)
        assert not blocked
        assert T == pytest.approx(1.0)


class TestComputeSolarIrradianceAllFacesMasked:
    """Tests for the parallel face irradiance kernel."""

    def _make_face_data(self, n=4):
        centers = np.array([
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 2.0],
            [4.0, 4.0, 2.0],
        ], dtype=np.float64)[:n]
        normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (n, 1))
        svf = np.full(n, 0.7, dtype=np.float64)
        return centers, normals, svf

    def test_basic_computation(self):
        from voxcity.simulator.solar.radiation import compute_solar_irradiance_for_all_faces_masked
        centers, normals, svf = self._make_face_data(4)
        vox_is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        fd, ff, fg = compute_solar_irradiance_for_all_faces_masked(
            centers, normals, svf, sun_dir,
            500.0, 200.0,
            vox_is_tree, vox_is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.05
        )
        assert fd.shape == (4,)
        assert ff.shape == (4,)
        assert fg.shape == (4,)
        # Direct should be ~500 (cos_incidence=1), diffuse = 0.7*200 = 140
        assert np.all(fd > 0)
        assert np.all(ff > 0)
        np.testing.assert_allclose(fg, fd + ff)

    def test_boundary_face_excluded(self):
        from voxcity.simulator.solar.radiation import compute_solar_irradiance_for_all_faces_masked
        # Face at boundary with vertical normal should be NaN
        centers = np.array([[0.0, 5.0, 5.0]], dtype=np.float64)  # x=0 = x_min
        normals = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)  # vertical (nz~0)
        svf = np.array([0.8], dtype=np.float64)
        vox_is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        sun_dir = np.array([1.0, 0.0, 0.5], dtype=np.float64)
        fd, ff, fg = compute_solar_irradiance_for_all_faces_masked(
            centers, normals, svf, sun_dir,
            500.0, 200.0,
            vox_is_tree, vox_is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.05
        )
        assert np.isnan(fd[0])
        assert np.isnan(ff[0])
        assert np.isnan(fg[0])

    def test_with_opaque_blocks_direct(self):
        from voxcity.simulator.solar.radiation import compute_solar_irradiance_for_all_faces_masked
        centers = np.array([[5.0, 5.0, 3.0]], dtype=np.float64)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        svf = np.array([0.8], dtype=np.float64)
        vox_is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        # Place opaque block above the face
        vox_is_opaque[5, 5, 5] = True
        sun_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        fd, ff, fg = compute_solar_irradiance_for_all_faces_masked(
            centers, normals, svf, sun_dir,
            500.0, 200.0,
            vox_is_tree, vox_is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.05
        )
        # Direct should be blocked → 0
        assert fd[0] == pytest.approx(0.0)
        # Diffuse should still be present
        assert ff[0] > 0


class TestCumulativeSolarIrradianceFacesMaskedTimeseries:
    """Tests for the per-timestep cumulative building face kernel."""

    def test_basic_timeseries(self):
        from voxcity.simulator.solar.radiation import compute_cumulative_solar_irradiance_faces_masked_timeseries
        n = 2
        centers = np.array([[5.0, 5.0, 3.0], [5.0, 5.0, 4.0]], dtype=np.float64)
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        svf = np.array([0.8, 0.6], dtype=np.float64)
        T = 3
        sun_dirs = np.array([
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.866],
            [0.0, 0.5, 0.866],
        ], dtype=np.float64)
        DNI = np.array([500.0, 400.0, 300.0], dtype=np.float64)
        DHI = np.array([100.0, 80.0, 60.0], dtype=np.float64)
        vox_is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        out_dir, out_diff, out_glob = compute_cumulative_solar_irradiance_faces_masked_timeseries(
            centers, normals, svf, sun_dirs, DNI, DHI,
            vox_is_tree, vox_is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.05,
            0, T, 1.0
        )
        assert out_dir.shape == (n,)
        assert out_diff.shape == (n,)
        assert out_glob.shape == (n,)
        # Should be cumulative positive values
        assert np.all(out_dir >= 0)
        assert np.all(out_diff > 0)
        np.testing.assert_allclose(out_glob, out_dir + out_diff)

    def test_below_horizon_diffuse_only(self):
        from voxcity.simulator.solar.radiation import compute_cumulative_solar_irradiance_faces_masked_timeseries
        centers = np.array([[5.0, 5.0, 3.0]], dtype=np.float64)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        svf = np.array([0.5], dtype=np.float64)
        # Sun below horizon (dz <= 0)
        sun_dirs = np.array([[0.0, 0.0, -0.5]], dtype=np.float64)
        DNI = np.array([500.0], dtype=np.float64)
        DHI = np.array([100.0], dtype=np.float64)
        vox_is_tree = np.zeros((10, 10, 10), dtype=np.bool_)
        vox_is_opaque = np.zeros((10, 10, 10), dtype=np.bool_)
        out_dir, out_diff, out_glob = compute_cumulative_solar_irradiance_faces_masked_timeseries(
            centers, normals, svf, sun_dirs, DNI, DHI,
            vox_is_tree, vox_is_opaque,
            1.0, 0.5,
            0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 0.05,
            0, 1, 1.0
        )
        # No direct, only diffuse
        assert out_dir[0] == pytest.approx(0.0)
        assert out_diff[0] == pytest.approx(50.0)  # 0.5 * 100 * 1h
        assert out_glob[0] == pytest.approx(50.0)
