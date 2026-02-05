"""Tests for voxcity.simulator.solar.sky module."""
import pytest
import numpy as np


class TestTregenzaPatches:
    """Tests for Tregenza sky discretization."""

    def test_generates_145_patches(self):
        """Should generate exactly 145 Tregenza patches."""
        from voxcity.simulator.solar.sky import generate_tregenza_patches
        
        patches, directions, solid_angles = generate_tregenza_patches()
        
        assert len(patches) == 145
        assert len(directions) == 145
        assert len(solid_angles) == 145

    def test_patch_shape(self):
        """Patches should be (n, 2) with azimuth and elevation."""
        from voxcity.simulator.solar.sky import generate_tregenza_patches
        
        patches, _, _ = generate_tregenza_patches()
        
        assert patches.shape == (145, 2)

    def test_direction_shape(self):
        """Directions should be (n, 3) unit vectors."""
        from voxcity.simulator.solar.sky import generate_tregenza_patches
        
        _, directions, _ = generate_tregenza_patches()
        
        assert directions.shape == (145, 3)

    def test_directions_are_unit_vectors(self):
        """All direction vectors should have unit length."""
        from voxcity.simulator.solar.sky import generate_tregenza_patches
        
        _, directions, _ = generate_tregenza_patches()
        norms = np.linalg.norm(directions, axis=1)
        
        np.testing.assert_array_almost_equal(norms, np.ones(145), decimal=10)

    def test_solid_angles_positive(self):
        """All solid angles should be positive."""
        from voxcity.simulator.solar.sky import generate_tregenza_patches
        
        _, _, solid_angles = generate_tregenza_patches()
        
        assert np.all(solid_angles > 0)

    def test_solid_angles_sum_to_hemisphere(self):
        """Sum of solid angles should equal hemisphere area (2π)."""
        from voxcity.simulator.solar.sky import generate_tregenza_patches
        
        _, _, solid_angles = generate_tregenza_patches()
        
        # 2π ≈ 6.283
        assert solid_angles.sum() == pytest.approx(2 * np.pi, rel=0.01)

    def test_azimuths_in_valid_range(self):
        """Azimuths should be in [0, 360)."""
        from voxcity.simulator.solar.sky import generate_tregenza_patches
        
        patches, _, _ = generate_tregenza_patches()
        azimuths = patches[:, 0]
        
        assert np.all(azimuths >= 0)
        assert np.all(azimuths < 360)

    def test_elevations_in_valid_range(self):
        """Elevations should be in (0, 90]."""
        from voxcity.simulator.solar.sky import generate_tregenza_patches
        
        patches, _, _ = generate_tregenza_patches()
        elevations = patches[:, 1]
        
        assert np.all(elevations > 0)
        assert np.all(elevations <= 90)


class TestGetTregenzaPatchIndex:
    """Tests for get_tregenza_patch_index function."""

    def test_below_horizon_returns_negative(self):
        """Negative elevation should return -1."""
        from voxcity.simulator.solar.sky import get_tregenza_patch_index
        
        result = get_tregenza_patch_index(180.0, -5.0)
        assert result == -1

    def test_zenith_returns_valid_index(self):
        """90 degree elevation (zenith) should return valid index."""
        from voxcity.simulator.solar.sky import get_tregenza_patch_index
        
        result = get_tregenza_patch_index(0.0, 90.0)
        # Last patch (index 144) is zenith
        assert 0 <= result <= 144

    def test_returns_index_in_valid_range(self):
        """Should return index in [0, 144] for valid positions."""
        from voxcity.simulator.solar.sky import get_tregenza_patch_index
        
        for az in [0, 90, 180, 270]:
            for elev in [10, 30, 50, 70, 85]:
                result = get_tregenza_patch_index(float(az), float(elev))
                assert 0 <= result <= 144, f"Failed for az={az}, elev={elev}"


class TestReinhartPatches:
    """Tests for Reinhart sky discretization."""

    def test_mf1_equals_tregenza(self):
        """MF=1 Reinhart should equal Tregenza (145 patches)."""
        from voxcity.simulator.solar.sky import generate_reinhart_patches
        
        patches, _, _ = generate_reinhart_patches(mf=1)
        
        assert len(patches) == 145

    def test_mf2_has_more_patches(self):
        """MF=2 Reinhart should have more patches than Tregenza."""
        from voxcity.simulator.solar.sky import generate_reinhart_patches
        
        patches_mf1, _, _ = generate_reinhart_patches(mf=1)
        patches_mf2, _, _ = generate_reinhart_patches(mf=2)
        
        assert len(patches_mf2) > len(patches_mf1)

    def test_directions_are_unit_vectors(self):
        """All direction vectors should have unit length."""
        from voxcity.simulator.solar.sky import generate_reinhart_patches
        
        _, directions, _ = generate_reinhart_patches(mf=2)
        norms = np.linalg.norm(directions, axis=1)
        
        np.testing.assert_array_almost_equal(norms, np.ones(len(directions)), decimal=10)


class TestUniformGridPatches:
    """Tests for uniform grid sky discretization."""

    def test_generates_expected_count(self):
        """Should generate n_azimuth × n_elevation patches."""
        from voxcity.simulator.solar.sky import generate_uniform_grid_patches
        
        n_az, n_el = 36, 9
        patches, _, _ = generate_uniform_grid_patches(n_azimuth=n_az, n_elevation=n_el)
        
        assert len(patches) == n_az * n_el

    def test_directions_are_unit_vectors(self):
        """All direction vectors should have unit length."""
        from voxcity.simulator.solar.sky import generate_uniform_grid_patches
        
        _, directions, _ = generate_uniform_grid_patches(n_azimuth=36, n_elevation=9)
        norms = np.linalg.norm(directions, axis=1)
        
        np.testing.assert_array_almost_equal(norms, np.ones(len(directions)), decimal=10)


class TestFibonacciPatches:
    """Tests for Fibonacci sky discretization."""

    def test_generates_requested_count(self):
        """Should generate approximately requested number of patches."""
        from voxcity.simulator.solar.sky import generate_fibonacci_patches
        
        n_requested = 100
        patches, _, _ = generate_fibonacci_patches(n_patches=n_requested)
        
        # Fibonacci might not give exactly n patches but should be close
        assert abs(len(patches) - n_requested) <= 10

    def test_directions_are_unit_vectors(self):
        """All direction vectors should have unit length."""
        from voxcity.simulator.solar.sky import generate_fibonacci_patches
        
        _, directions, _ = generate_fibonacci_patches(n_patches=100)
        norms = np.linalg.norm(directions, axis=1)
        
        np.testing.assert_array_almost_equal(norms, np.ones(len(directions)), decimal=10)

    def test_elevations_above_horizon(self):
        """All patches should be above horizon (positive z)."""
        from voxcity.simulator.solar.sky import generate_fibonacci_patches
        
        _, directions, _ = generate_fibonacci_patches(n_patches=100)
        
        # z component should be positive (above horizon)
        assert np.all(directions[:, 2] >= 0)
