"""Tests for the axis contract: AXES constants, direction_to_axis_vector, check_axes."""

import numpy as np
import pytest

from voxcity.utils.orientation import (
    AXES,
    AXES_ATTR,
    direction_to_axis_vector,
    check_axes,
)


class TestConstants:
    def test_axes_tokens(self):
        assert AXES == ("north", "east", "up")

    def test_axes_attr_derived(self):
        assert AXES_ATTR == ",".join(AXES) == "north,east,up"


class TestDirectionToAxisVector:
    def test_az0_points_plus_axis0(self):
        v = direction_to_axis_vector(0.0)
        np.testing.assert_allclose(v, [1.0, 0.0, 0.0], atol=1e-15)

    def test_az90_points_plus_axis1(self):
        v = direction_to_axis_vector(90.0)
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-15)

    def test_elevation90_points_plus_axis2(self):
        v = direction_to_axis_vector(0.0, 90.0)
        np.testing.assert_allclose(v, [0.0, 0.0, 1.0], atol=1e-15)

    def test_rotation_cancels_azimuth(self):
        a = direction_to_axis_vector(30.0, 20.0, 30.0)
        b = direction_to_axis_vector(0.0, 20.0, 0.0)
        np.testing.assert_array_equal(a, b)

    def test_unit_norm(self):
        for az in (0.0, 37.0, 123.4, 359.0):
            for el in (0.0, 15.0, 89.0):
                v = direction_to_axis_vector(az, el)
                assert abs(np.linalg.norm(v) - 1.0) < 1e-12

    def test_scalar_returns_shape_3(self):
        assert direction_to_axis_vector(45.0, 10.0).shape == (3,)

    def test_array_rotation_angle_broadcasts(self):
        az = np.array([0.0, 90.0])
        rot = np.array([0.0, 90.0])
        vec = direction_to_axis_vector(az, 0.0, rot)
        assert vec.shape == (2, 3)
        # az - rot = [0, 0] for both → both point along +axis0
        np.testing.assert_allclose(vec[0], [1.0, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(vec[1], [1.0, 0.0, 0.0], atol=1e-15)

    def test_broadcasting_matches_scalar_loop(self):
        az = np.array([0.0, 45.0, 90.0, 200.0])
        el = np.array([0.0, 10.0, 45.0, 80.0])
        vec = direction_to_axis_vector(az, el)
        assert vec.shape == (4, 3)
        for k in range(4):
            np.testing.assert_array_equal(
                vec[k], direction_to_axis_vector(az[k], el[k])
            )

    def test_broadcasting_2d_outer(self):
        az = np.array([0.0, 90.0, 180.0])[None, :]     # (1, 3)
        el = np.array([0.0, 45.0])[:, None]            # (2, 1)
        vec = direction_to_axis_vector(az, el)
        assert vec.shape == (2, 3, 3)
        np.testing.assert_array_equal(
            vec[1, 2], direction_to_axis_vector(180.0, 45.0)
        )

    def test_met_from_direction_example(self):
        # A wind FROM the north (met 0 deg) blows TOWARD the south: -axis0.
        v = direction_to_axis_vector(0.0 + 180.0)
        np.testing.assert_allclose(v, [-1.0, 0.0, 0.0], atol=1e-15)


class TestCheckAxes:
    def test_passes_on_v3_attrs(self):
        check_axes({"axes": AXES_ATTR})  # no raise

    def test_passes_on_bytes_value(self):
        check_axes({"axes": AXES_ATTR.encode("utf-8")})

    def test_missing_attr_names_migrate(self):
        with pytest.raises(ValueError, match="migrate_h5"):
            check_axes({})

    def test_foreign_value_states_both_sides(self):
        with pytest.raises(ValueError) as exc_info:
            check_axes({"axes": "east,north,up"})
        msg = str(exc_info.value)
        assert "east,north,up" in msg
        assert AXES_ATTR in msg

    def test_accepts_h5py_file_object(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        p = tmp_path / "t.h5"
        with h5py.File(p, "w") as f:
            f.attrs["axes"] = AXES_ATTR
        with h5py.File(p, "r") as f:
            check_axes(f)  # no raise

    def test_accepts_path(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        p = tmp_path / "t.h5"
        with h5py.File(p, "w") as f:
            f.attrs["axes"] = AXES_ATTR
        check_axes(str(p))  # no raise
