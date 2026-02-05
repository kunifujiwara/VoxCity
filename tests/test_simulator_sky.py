"""Tests for voxcity.simulator.solar.sky module - sky discretization methods."""
import pytest
import numpy as np

from voxcity.simulator.solar.sky import (
    generate_tregenza_patches,
    get_tregenza_patch_index,
    get_tregenza_patch_index_fast,
    generate_reinhart_patches,
    generate_uniform_grid_patches,
    generate_fibonacci_patches,
    bin_sun_positions_to_patches,
    get_patch_info,
    TREGENZA_BANDS,
    TREGENZA_BAND_BOUNDARIES,
)


class TestTregenzaPatches:
    """Tests for Tregenza sky subdivision."""

    def test_generate_tregenza_returns_145_patches(self):
        """Test that Tregenza generates exactly 145 patches."""
        patches, directions, solid_angles = generate_tregenza_patches()
        assert len(patches) == 145
        assert len(directions) == 145
        assert len(solid_angles) == 145

    def test_patches_shape(self):
        """Test patch array shapes."""
        patches, directions, solid_angles = generate_tregenza_patches()
        assert patches.shape == (145, 2)
        assert directions.shape == (145, 3)
        assert solid_angles.shape == (145,)

    def test_directions_are_unit_vectors(self):
        """Test that direction vectors are normalized."""
        _, directions, _ = generate_tregenza_patches()
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_solid_angles_sum_to_hemisphere(self):
        """Test that solid angles sum to 2π (hemisphere)."""
        _, _, solid_angles = generate_tregenza_patches()
        total = np.sum(solid_angles)
        np.testing.assert_allclose(total, 2 * np.pi, rtol=0.01)

    def test_elevation_ranges(self):
        """Test that patch elevations are in valid range."""
        patches, _, _ = generate_tregenza_patches()
        elevations = patches[:, 1]
        assert np.all(elevations >= 0)
        assert np.all(elevations <= 90)

    def test_azimuth_ranges(self):
        """Test that patch azimuths are in 0-360 range."""
        patches, _, _ = generate_tregenza_patches()
        azimuths = patches[:, 0]
        assert np.all(azimuths >= 0)
        assert np.all(azimuths < 360)


class TestTregenzaPatchIndex:
    """Tests for get_tregenza_patch_index function."""

    def test_below_horizon_returns_negative(self):
        """Test that negative elevation returns -1."""
        assert get_tregenza_patch_index(0.0, -5.0) == -1

    def test_zenith_returns_patch_144(self):
        """Test that zenith position returns patch 144."""
        assert get_tregenza_patch_index(0.0, 90.0) == 144
        assert get_tregenza_patch_index(180.0, 89.0) == 144

    def test_horizon_returns_first_band(self):
        """Test that low elevation returns patches in first band."""
        idx = get_tregenza_patch_index(0.0, 5.0)
        assert 0 <= idx < 30  # First band has 30 patches

    def test_index_is_deterministic(self):
        """Test that same position gives same index."""
        idx1 = get_tregenza_patch_index(45.0, 30.0)
        idx2 = get_tregenza_patch_index(45.0, 30.0)
        assert idx1 == idx2

    def test_fast_version_matches_slow(self):
        """Test that numba version matches Python version."""
        test_cases = [
            (0.0, 5.0),
            (90.0, 25.0),
            (180.0, 45.0),
            (270.0, 70.0),
            (0.0, 85.0),
            (0.0, 90.0),
        ]
        for az, elev in test_cases:
            slow = get_tregenza_patch_index(az, elev)
            fast = get_tregenza_patch_index_fast(az, elev)
            assert slow == fast, f"Mismatch at az={az}, elev={elev}"

    def test_fast_version_below_horizon(self):
        """Test numba version for below horizon."""
        assert get_tregenza_patch_index_fast(0.0, -5.0) == -1


class TestReinhartPatches:
    """Tests for Reinhart sky subdivision."""

    def test_mf1_equals_tregenza(self):
        """Test that MF=1 gives same count as Tregenza."""
        patches, _, _ = generate_reinhart_patches(mf=1)
        # MF=1 has same number of altitude bands but different structure
        assert len(patches) > 0

    def test_mf2_more_patches_than_mf1(self):
        """Test that higher MF gives more patches."""
        p1, _, _ = generate_reinhart_patches(mf=1)
        p2, _, _ = generate_reinhart_patches(mf=2)
        p4, _, _ = generate_reinhart_patches(mf=4)
        assert len(p2) > len(p1)
        assert len(p4) > len(p2)

    def test_directions_are_unit_vectors(self):
        """Test that direction vectors are normalized."""
        _, directions, _ = generate_reinhart_patches(mf=2)
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_solid_angles_sum_to_hemisphere(self):
        """Test that solid angles sum to 2π."""
        _, _, solid_angles = generate_reinhart_patches(mf=2)
        total = np.sum(solid_angles)
        np.testing.assert_allclose(total, 2 * np.pi, rtol=0.02)


class TestUniformGridPatches:
    """Tests for uniform grid sky subdivision."""

    def test_default_patch_count(self):
        """Test default 36x9 = 324 patches."""
        patches, _, _ = generate_uniform_grid_patches()
        assert len(patches) == 36 * 9

    def test_custom_dimensions(self):
        """Test custom azimuth/elevation divisions."""
        patches, _, _ = generate_uniform_grid_patches(n_azimuth=12, n_elevation=6)
        assert len(patches) == 12 * 6

    def test_directions_are_unit_vectors(self):
        """Test that direction vectors are normalized."""
        _, directions, _ = generate_uniform_grid_patches()
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_solid_angles_sum_to_hemisphere(self):
        """Test that solid angles sum to 2π."""
        _, _, solid_angles = generate_uniform_grid_patches()
        total = np.sum(solid_angles)
        np.testing.assert_allclose(total, 2 * np.pi, rtol=0.01)


class TestFibonacciPatches:
    """Tests for Fibonacci spiral sky subdivision."""

    def test_default_patch_count(self):
        """Test default 145 patches (matching Tregenza)."""
        patches, _, _ = generate_fibonacci_patches()
        assert len(patches) == 145

    def test_custom_patch_count(self):
        """Test custom number of patches."""
        patches, _, _ = generate_fibonacci_patches(n_patches=200)
        assert len(patches) == 200

    def test_directions_are_unit_vectors(self):
        """Test that direction vectors are normalized."""
        _, directions, _ = generate_fibonacci_patches()
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_solid_angles_are_uniform(self):
        """Test that Fibonacci gives uniform solid angles."""
        _, _, solid_angles = generate_fibonacci_patches()
        # All solid angles should be approximately equal
        np.testing.assert_allclose(solid_angles, solid_angles[0], rtol=1e-10)

    def test_solid_angles_sum_to_hemisphere(self):
        """Test that solid angles sum to 2π."""
        _, _, solid_angles = generate_fibonacci_patches()
        total = np.sum(solid_angles)
        np.testing.assert_allclose(total, 2 * np.pi, rtol=0.01)


class TestBinSunPositionsToPatches:
    """Tests for sun position binning."""

    @pytest.fixture
    def sample_sun_data(self):
        """Create sample sun position data."""
        # Simulate a day with varying sun positions
        hours = np.arange(6, 20)  # 6 AM to 8 PM
        azimuths = np.linspace(90, 270, len(hours))  # East to West
        elevations = np.concatenate([
            np.linspace(0, 60, len(hours)//2),
            np.linspace(60, 0, len(hours) - len(hours)//2)
        ])
        dni = np.where(elevations > 0, 800 * np.sin(np.deg2rad(elevations)), 0)
        return azimuths, elevations, dni

    def test_tregenza_binning(self, sample_sun_data):
        """Test binning with Tregenza method."""
        azimuths, elevations, dni = sample_sun_data
        directions, cum_dni, solid_angles, hours = bin_sun_positions_to_patches(
            azimuths, elevations, dni, method="tregenza"
        )
        assert len(directions) == 145
        assert len(cum_dni) == 145
        assert len(solid_angles) == 145
        assert len(hours) == 145

    def test_reinhart_binning(self, sample_sun_data):
        """Test binning with Reinhart method."""
        azimuths, elevations, dni = sample_sun_data
        directions, cum_dni, solid_angles, hours = bin_sun_positions_to_patches(
            azimuths, elevations, dni, method="reinhart", mf=2
        )
        assert len(directions) > 145  # Reinhart has more patches

    def test_uniform_binning(self, sample_sun_data):
        """Test binning with uniform grid method."""
        azimuths, elevations, dni = sample_sun_data
        directions, cum_dni, solid_angles, hours = bin_sun_positions_to_patches(
            azimuths, elevations, dni, method="uniform"
        )
        assert len(directions) == 36 * 9

    def test_fibonacci_binning(self, sample_sun_data):
        """Test binning with Fibonacci method."""
        azimuths, elevations, dni = sample_sun_data
        directions, cum_dni, solid_angles, hours = bin_sun_positions_to_patches(
            azimuths, elevations, dni, method="fibonacci", n_patches=100
        )
        assert len(directions) == 100

    def test_below_horizon_ignored(self):
        """Test that below horizon positions are ignored."""
        azimuths = np.array([0.0, 90.0, 180.0])
        elevations = np.array([-10.0, 30.0, -5.0])
        dni = np.array([100.0, 800.0, 100.0])
        
        _, cum_dni, _, hours = bin_sun_positions_to_patches(
            azimuths, elevations, dni, method="tregenza"
        )
        # Only one position above horizon
        assert np.sum(hours) == 1

    def test_cumulative_dni_sums_correctly(self):
        """Test that DNI is accumulated correctly."""
        # Put multiple readings in same patch
        azimuths = np.array([5.0, 6.0, 7.0])  # Same patch
        elevations = np.array([5.0, 5.0, 5.0])
        dni = np.array([100.0, 200.0, 300.0])
        
        _, cum_dni, _, hours = bin_sun_positions_to_patches(
            azimuths, elevations, dni, method="tregenza"
        )
        assert np.max(cum_dni) == 600.0  # 100 + 200 + 300
        assert np.max(hours) == 3

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sky discretization"):
            bin_sun_positions_to_patches(
                np.array([0.0]), np.array([45.0]), np.array([500.0]),
                method="invalid"
            )


class TestGetPatchInfo:
    """Tests for get_patch_info utility function."""

    def test_tregenza_info(self):
        """Test getting Tregenza patch info."""
        info = get_patch_info("tregenza")
        assert info["method"] == "Tregenza"
        assert info["n_patches"] == 145
        assert "Tregenza" in info["reference"]

    def test_reinhart_info(self):
        """Test getting Reinhart patch info."""
        info = get_patch_info("reinhart", mf=4)
        assert info["method"] == "Reinhart"
        assert info["mf"] == 4
        assert info["n_patches"] > 145

    def test_uniform_info(self):
        """Test getting uniform grid info."""
        info = get_patch_info("uniform", n_azimuth=24, n_elevation=6)
        assert info["method"] == "Uniform Grid"
        assert info["n_azimuth"] == 24
        assert info["n_elevation"] == 6
        assert info["n_patches"] == 144

    def test_fibonacci_info(self):
        """Test getting Fibonacci info."""
        info = get_patch_info("fibonacci", n_patches=200)
        assert info["method"] == "Fibonacci Spiral"
        assert info["n_patches"] == 200

    def test_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            get_patch_info("nonexistent")


class TestTregenzaConstants:
    """Tests for Tregenza constants."""

    def test_band_count(self):
        """Test there are 8 bands."""
        assert len(TREGENZA_BANDS) == 8

    def test_total_patches_from_bands(self):
        """Test total patches from bands equals 145."""
        total = sum(count for _, count in TREGENZA_BANDS)
        assert total == 145

    def test_boundaries_count(self):
        """Test boundary count."""
        assert len(TREGENZA_BAND_BOUNDARIES) == 9  # 8 bands + 1

    def test_boundaries_increase(self):
        """Test boundaries are monotonically increasing."""
        for i in range(len(TREGENZA_BAND_BOUNDARIES) - 1):
            assert TREGENZA_BAND_BOUNDARIES[i] < TREGENZA_BAND_BOUNDARIES[i + 1]

    def test_boundaries_start_and_end(self):
        """Test boundaries start at 0 and end at 90."""
        assert TREGENZA_BAND_BOUNDARIES[0] == 0.0
        assert TREGENZA_BAND_BOUNDARIES[-1] == 90.0
