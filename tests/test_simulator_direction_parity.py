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
