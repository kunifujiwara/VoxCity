"""Tests for VoxCity.to_xarray(): named dims, cell-centre coords, zero-copy, attrs."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from voxcity.utils.orientation import AXES, AXES_ATTR

from tests.conftest import make_city, RECT


class TestToXarray:
    def test_dims_are_axes_tokens(self):
        ds = make_city().to_xarray()
        assert ds["voxel"].dims == AXES
        for name in ("building_height", "building_id", "land_cover", "dem", "canopy_top"):
            assert ds[name].dims == AXES[:2]

    def test_cell_centre_coordinates(self):
        ds = make_city(shape=(4, 5, 6), meshsize=2.0).to_xarray()
        np.testing.assert_allclose(ds["north"].values, (np.arange(4) + 0.5) * 2.0)
        np.testing.assert_allclose(ds["east"].values, (np.arange(5) + 0.5) * 2.0)
        np.testing.assert_allclose(ds["up"].values, (np.arange(6) + 0.5) * 2.0)

    def test_zero_copy(self):
        city = make_city()
        ds = city.to_xarray()
        assert np.shares_memory(ds["voxel"].values, city.voxels.classes)
        assert np.shares_memory(ds["dem"].values, city.dem.elevation)

    def test_attrs(self):
        ds = make_city().to_xarray()
        assert ds.attrs["axes"] == AXES_ATTR
        assert ds.attrs["rotation_angle"] == 0.0
        assert ds.attrs["meshsize"] == 2.0
        assert ds.attrs["crs"] == "EPSG:4326"
        np.testing.assert_allclose(
            ds.attrs["rectangle_vertices"], np.asarray(RECT, dtype=float)
        )

    def test_south_edge_is_isel_north_0(self):
        city = make_city()
        city.dem.elevation[0, :] = 7.0  # row 0 = south edge by contract
        ds = city.to_xarray()
        np.testing.assert_allclose(ds["dem"].isel(north=0).values, 7.0)

    def test_no_extras_defaults_rotation_zero(self):
        ds = make_city(extras={}).to_xarray()
        assert ds.attrs["rotation_angle"] == 0.0
        assert "rectangle_vertices" not in ds.attrs

    def test_rotated_city_rotation_angle(self):
        from tests.conftest import rotated_rect

        city = make_city(extras={"rectangle_vertices": rotated_rect(25.0)})
        ds = city.to_xarray()
        assert ds.attrs["rotation_angle"] == pytest.approx(25.0, abs=1e-3)
        assert ds["voxel"].dims == AXES

    def test_canopy_bottom_present_when_set(self):
        city = make_city()
        city.tree_canopy.bottom = np.zeros_like(city.dem.elevation)
        ds = city.to_xarray()
        assert "canopy_bottom" in ds
        assert ds["canopy_bottom"].dims == AXES[:2]

    def test_excludes_min_heights_and_extras(self):
        city = make_city(extras={"rectangle_vertices": RECT, "custom_key": "x"})
        ds = city.to_xarray()
        assert "min_heights" not in ds.variables
        assert "building_min_heights" not in ds.variables
        assert "custom_key" not in ds.attrs

    def test_noncanonical_rect_agrees_with_save(self, tmp_path):
        # A non-canonical vertex order in extras must not make the view disagree
        # with the file the same city would save to: to_xarray normalizes just
        # like save_results_h5, so rotation_angle and stored rectangle match.
        import h5py
        from voxcity.io import save_results_h5

        # RECT is [SW, NW, NE, SE]; rotate the list so v0 is no longer SW.
        noncanonical = RECT[2:] + RECT[:2]  # [NE, SE, SW, NW]
        city = make_city(extras={"rectangle_vertices": noncanonical})

        ds = city.to_xarray()
        p = str(tmp_path / "c.h5")
        save_results_h5(p, city)
        with h5py.File(p, "r") as f:
            file_rot = float(f.attrs["rotation_angle"])
            file_rect = f["rectangle_vertices"][:]

        assert ds.attrs["rotation_angle"] == pytest.approx(file_rot, abs=1e-9)
        np.testing.assert_allclose(ds.attrs["rectangle_vertices"], file_rect)
