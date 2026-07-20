"""Tests for GridProjector.from_city / from_h5 (v3 geometry, strict on pre-v3)."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from voxcity.utils.projector import GridProjector

from tests.conftest import make_city, write_v2_file, RECT  # shared fixtures


class TestFromCity:
    def test_round_trip_within_half_cell(self):
        city = make_city(meshsize=5.0)
        proj = GridProjector.from_city(city)
        for i, j in [(0, 0), (1, 3), (3, 4)]:
            lon, lat = proj.cell_to_lon_lat(i, j)
            assert proj.lon_lat_to_cell(lon, lat) == (i, j)

    def test_origin_cell_centre_is_inside_rect(self):
        city = make_city()
        proj = GridProjector.from_city(city)
        lon, lat = proj.cell_to_lon_lat(0, 0)
        assert 0.0 < lon < 0.01 and 0.0 < lat < 0.01

    def test_bounds_fallback_when_no_vertices(self):
        city = make_city(meshsize=5.0, extras={})
        proj = GridProjector.from_city(city)
        # bounds (0,0,0.01,0.01) → origin cell centre lands inside the AOI
        lon, lat = proj.cell_to_lon_lat(0, 0)
        assert 0.0 < lon < 0.01 and 0.0 < lat < 0.01

    def test_noncanonical_rect_matches_from_h5(self, tmp_path):
        # A non-canonical vertex order in extras must project identically to
        # from_h5 (which reads the file's normalized geometry): from_city
        # normalizes just like save does, so the two agree.
        from voxcity.io import save_results_h5

        noncanonical = RECT[2:] + RECT[:2]  # [NE, SE, SW, NW]
        city = make_city(meshsize=5.0, extras={"rectangle_vertices": noncanonical})
        p = str(tmp_path / "c.h5")
        save_results_h5(p, city)

        a = GridProjector.from_city(city)
        b = GridProjector.from_h5(p)
        np.testing.assert_allclose(a.cell_to_lon_lat(2, 3), b.cell_to_lon_lat(2, 3))


class TestFromH5:
    def test_equals_from_city(self, tmp_path):
        from voxcity.io import save_results_h5

        city = make_city(meshsize=5.0)
        p = str(tmp_path / "c.h5")
        save_results_h5(p, city)
        a = GridProjector.from_city(city)
        b = GridProjector.from_h5(p)
        np.testing.assert_allclose(a.cell_to_lon_lat(2, 3), b.cell_to_lon_lat(2, 3))

    def test_pre_v3_raises_migrate_error(self, tmp_path):
        p = write_v2_file(tmp_path / "old.h5")
        with pytest.raises(ValueError, match="migrate_h5"):
            GridProjector.from_h5(p)

    def test_v3_tag_but_missing_geometry_raises_clear_error(self, tmp_path):
        from voxcity.utils.orientation import AXES_ATTR

        p = str(tmp_path / "truncated.h5")
        with h5py.File(p, "w") as f:
            f.attrs["axes"] = AXES_ATTR
            f.attrs["meshsize"] = 2.0
            # axes present so check_axes passes, but no rectangle_vertices dataset
        with pytest.raises(ValueError, match="corrupted"):
            GridProjector.from_h5(p)
