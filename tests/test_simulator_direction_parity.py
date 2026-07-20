"""Parity: direction_to_axis_vector reproduces the historical inline azimuth math.

The old formulas are reproduced VERBATIM here; these tests pin the refactor.
"""

import numpy as np

from voxcity.utils.orientation import direction_to_axis_vector

AZIMUTHS = [0.0, 30.0, 90.0, 137.5, 180.0, 270.0, 359.0]
ELEVATIONS = [0.0, 10.0, 45.0, 89.9]
ROTATIONS = [0.0, -25.0, 30.0, 90.0]


class TestRadiationSiteParity:
    def test_scalar_site_bitwise(self):
        for az in AZIMUTHS:
            for el in ELEVATIONS:
                for rot in ROTATIONS:
                    # Old inline formula (radiation.py, both sites), verbatim:
                    azimuth_radians = np.deg2rad(az - rot)
                    elevation_radians = np.deg2rad(el)
                    dx = np.cos(elevation_radians) * np.cos(azimuth_radians)
                    dy = np.cos(elevation_radians) * np.sin(azimuth_radians)
                    dz = np.sin(elevation_radians)
                    old = np.array([dx, dy, dz], dtype=np.float64)

                    new = direction_to_axis_vector(az, el, rot)
                    np.testing.assert_array_equal(new, old)


class TestRayFanParity:
    def test_grid_fan_matches_old_loop(self):
        from voxcity.simulator.common.geometry import _generate_ray_directions_grid

        N_az, N_el, el_min, el_max = 36, 9, 10.0, 80.0

        # Old implementation, verbatim (radians-native linspace):
        azimuth_angles = np.linspace(0.0, 2.0 * np.pi, N_az, endpoint=False)
        elevation_angles = np.deg2rad(np.linspace(el_min, el_max, N_el))
        old = np.empty((N_az * N_el, 3), dtype=np.float64)
        out_idx = 0
        for elevation in elevation_angles:
            cos_elev = np.cos(elevation)
            sin_elev = np.sin(elevation)
            for azimuth in azimuth_angles:
                old[out_idx, 0] = cos_elev * np.cos(azimuth)
                old[out_idx, 1] = cos_elev * np.sin(azimuth)
                old[out_idx, 2] = sin_elev
                out_idx += 1

        new = _generate_ray_directions_grid(N_az, N_el, el_min, el_max)
        assert new.shape == old.shape
        assert new.dtype == np.float64
        assert new.flags["C_CONTIGUOUS"]
        # Azimuth generation moved from linspace(0, 2*pi) to
        # deg2rad(linspace(0, 360)) — identical to within 1 ulp of the
        # angle, hence the tight tolerance instead of array_equal.
        np.testing.assert_allclose(new, old, rtol=0.0, atol=1e-12)


class TestSkyPatchParity:
    """Each generator returns patches as an (N, 2) array of
    (azimuth_degrees, elevation_degrees) and directions as an (N, 3) array
    of (dx, dy, dz); every direction must equal the old inline formula
    applied to its own patch's (az_deg, elev_deg) — exactly, since all sky
    sites are degree-based."""

    @staticmethod
    def _old_formula(az_deg, elev_deg):
        az_rad = np.deg2rad(az_deg)
        elev_rad = np.deg2rad(elev_deg)
        dx = np.cos(elev_rad) * np.cos(az_rad)
        dy = np.cos(elev_rad) * np.sin(az_rad)
        dz = np.sin(elev_rad)
        return dx, dy, dz

    def _assert_directions_match(self, patches, directions):
        assert len(patches) == len(directions) > 0
        for (az_deg, elev_deg), d in zip(patches, directions):
            np.testing.assert_array_equal(
                np.asarray(d, dtype=np.float64),
                np.asarray(self._old_formula(az_deg, elev_deg), dtype=np.float64),
            )

    def test_tregenza(self):
        from voxcity.simulator.solar.sky import generate_tregenza_patches
        # Returns (patches, directions, solid_angles); patches is (145, 2)
        # of (az_deg, elev_deg), directions is (145, 3) of (dx, dy, dz).
        patches, directions, solid_angles = generate_tregenza_patches()
        self._assert_directions_match(patches, directions)

    def test_reinhart(self):
        from voxcity.simulator.solar.sky import generate_reinhart_patches
        # Returns (patches, directions, solid_angles); same (N, 2)/(N, 3)
        # shapes as generate_tregenza_patches.
        patches, directions, solid_angles = generate_reinhart_patches(mf=2)
        self._assert_directions_match(patches, directions)

    def test_uniform_grid(self):
        from voxcity.simulator.solar.sky import generate_uniform_grid_patches
        # Returns (patches, directions, solid_angles); same (N, 2)/(N, 3)
        # shapes as generate_tregenza_patches.
        patches, directions, solid_angles = generate_uniform_grid_patches(
            n_azimuth=12, n_elevation=4
        )
        self._assert_directions_match(patches, directions)
