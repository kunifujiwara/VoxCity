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
