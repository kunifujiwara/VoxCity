"""The direct-beam cosine must come from surfaces.normal, not `direction`."""
import numpy as np
import pytest

from voxcity.simulator_gpu.solar.radiation import surface_incidence_cosine


def test_cosine_follows_the_stored_normal():
    r2 = float(np.sqrt(0.5))
    assert surface_incidence_cosine((r2, r2, 0.0), (r2, r2, 0.0)) == pytest.approx(1.0, abs=1e-6)


def test_axis_normal_is_unchanged():
    r2 = float(np.sqrt(0.5))
    assert surface_incidence_cosine((1.0, 0.0, 0.0), (r2, r2, 0.0)) == pytest.approx(r2, abs=1e-6)


def test_back_face_clamps_to_zero():
    assert surface_incidence_cosine((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)) == 0.0
