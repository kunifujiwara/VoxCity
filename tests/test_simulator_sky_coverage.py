"""
Comprehensive tests for voxcity.simulator.solar.sky to improve coverage.
"""

import numpy as np
import pytest

from voxcity.simulator.solar.sky import (
    TREGENZA_BANDS,
    TREGENZA_BAND_BOUNDARIES,
    generate_tregenza_patches,
    get_tregenza_patch_index,
    get_tregenza_patch_index_fast,
    generate_reinhart_patches,
    generate_uniform_grid_patches,
    generate_fibonacci_patches,
)


class TestTregenzaConstants:
    def test_band_count(self):
        assert len(TREGENZA_BANDS) == 8

    def test_boundary_count(self):
        assert len(TREGENZA_BAND_BOUNDARIES) == 9

    def test_total_patches(self):
        total = sum(n for _, n in TREGENZA_BANDS)
        assert total == 145

    def test_boundaries_ascending(self):
        for i in range(len(TREGENZA_BAND_BOUNDARIES) - 1):
            assert TREGENZA_BAND_BOUNDARIES[i] < TREGENZA_BAND_BOUNDARIES[i + 1]

    def test_boundary_range(self):
        assert TREGENZA_BAND_BOUNDARIES[0] == 0.0
        assert TREGENZA_BAND_BOUNDARIES[-1] == 90.0


class TestGenerateTregenzaPatches:
    def test_patch_count(self):
        patches, directions, solid_angles = generate_tregenza_patches()
        assert patches.shape == (145, 2)
        assert directions.shape == (145, 3)
        assert solid_angles.shape == (145,)

    def test_unit_directions(self):
        _, directions, _ = generate_tregenza_patches()
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_positive_solid_angles(self):
        _, _, solid_angles = generate_tregenza_patches()
        assert np.all(solid_angles > 0)

    def test_solid_angles_sum_to_hemisphere(self):
        _, _, solid_angles = generate_tregenza_patches()
        total = np.sum(solid_angles)
        # Hemisphere solid angle = 2π
        assert total == pytest.approx(2 * np.pi, rel=0.01)

    def test_elevation_range(self):
        patches, _, _ = generate_tregenza_patches()
        elevations = patches[:, 1]
        assert np.min(elevations) >= 0.0
        assert np.max(elevations) <= 90.0

    def test_azimuth_range(self):
        patches, _, _ = generate_tregenza_patches()
        azimuths = patches[:, 0]
        assert np.min(azimuths) >= 0.0
        assert np.max(azimuths) <= 360.0

    def test_directions_upper_hemisphere(self):
        _, directions, _ = generate_tregenza_patches()
        assert np.all(directions[:, 2] >= 0)  # z >= 0


class TestGetTregenzaPatchIndex:
    def test_below_horizon(self):
        idx = get_tregenza_patch_index(0.0, -1.0)
        assert idx == -1

    def test_zenith(self):
        idx = get_tregenza_patch_index(0.0, 90.0)
        assert idx == 144

    def test_near_zenith(self):
        idx = get_tregenza_patch_index(0.0, 85.0)
        assert idx == 144

    def test_first_band(self):
        idx = get_tregenza_patch_index(0.0, 6.0)
        assert 0 <= idx < 30

    def test_valid_range(self):
        idx = get_tregenza_patch_index(180.0, 45.0)
        assert 0 <= idx <= 144

    def test_all_azimuths_mapped(self):
        indices = set()
        for az in range(0, 360, 10):
            idx = get_tregenza_patch_index(float(az), 6.0)
            indices.add(idx)
        # First band has 30 patches, should get many unique indices
        assert len(indices) > 10

    def test_wrap_azimuth(self):
        idx1 = get_tregenza_patch_index(0.0, 30.0)
        idx2 = get_tregenza_patch_index(360.0, 30.0)
        assert idx1 == idx2


class TestGetTregenzaPatchIndexFast:
    def test_below_horizon(self):
        idx = get_tregenza_patch_index_fast(0.0, -5.0)
        assert idx == -1

    def test_zenith(self):
        idx = get_tregenza_patch_index_fast(0.0, 90.0)
        assert idx == 144

    def test_matches_slow_version(self):
        test_cases = [
            (0.0, 6.0), (90.0, 30.0), (180.0, 50.0),
            (270.0, 70.0), (45.0, 85.0), (135.0, 15.0),
        ]
        for az, el in test_cases:
            idx_slow = get_tregenza_patch_index(az, el)
            idx_fast = get_tregenza_patch_index_fast(az, el)
            assert idx_slow == idx_fast, f"Mismatch at az={az}, el={el}"


class TestGenerateReinhartPatches:
    def test_mf1_equals_tregenza(self):
        patches, directions, solid_angles = generate_reinhart_patches(mf=1)
        t_patches, _, _ = generate_tregenza_patches()
        assert patches.shape[0] == t_patches.shape[0]

    def test_mf2_more_patches(self):
        patches, directions, solid_angles = generate_reinhart_patches(mf=2)
        assert patches.shape[0] > 145

    def test_unit_directions(self):
        _, directions, _ = generate_reinhart_patches(mf=2)
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_positive_solid_angles(self):
        _, _, solid_angles = generate_reinhart_patches(mf=2)
        assert np.all(solid_angles > 0)

    def test_mf4(self):
        patches, _, _ = generate_reinhart_patches(mf=4)
        assert patches.shape[0] > 500

    def test_upper_hemisphere(self):
        _, directions, _ = generate_reinhart_patches(mf=2)
        assert np.all(directions[:, 2] >= 0)


class TestGenerateUniformGridPatches:
    def test_basic(self):
        patches, directions, solid_angles = generate_uniform_grid_patches(
            n_azimuth=12, n_elevation=6
        )
        assert patches.shape[0] == 72  # 12 * 6
        assert directions.shape[1] == 3

    def test_unit_directions(self):
        _, directions, _ = generate_uniform_grid_patches(
            n_azimuth=8, n_elevation=4
        )
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_single_patch(self):
        patches, directions, solid_angles = generate_uniform_grid_patches(
            n_azimuth=1, n_elevation=1
        )
        assert patches.shape[0] == 1


class TestGenerateFibonacciPatches:
    def test_basic(self):
        patches, directions, solid_angles = generate_fibonacci_patches(n_patches=100)
        assert patches.shape[0] == 100

    def test_unit_directions(self):
        _, directions, _ = generate_fibonacci_patches(n_patches=50)
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_upper_hemisphere(self):
        _, directions, _ = generate_fibonacci_patches(n_patches=200)
        assert np.all(directions[:, 2] >= 0)

    def test_positive_solid_angles(self):
        _, _, solid_angles = generate_fibonacci_patches(n_patches=100)
        assert np.all(solid_angles > 0)
