"""Tests for exporter/netcdf.py to improve coverage."""

import numpy as np
import json
import pytest


class TestVoxelToXarrayDataset:
    def test_basic_3d(self):
        from voxcity.exporter.netcdf import voxel_to_xarray_dataset
        grid = np.zeros((4, 5, 6), dtype=np.int8)
        grid[1, 2, 3] = -3  # building
        ds = voxel_to_xarray_dataset(grid, voxel_size_m=5.0)
        assert "voxels" in ds
        assert ds["voxels"].shape == (4, 5, 6)
        assert ds.attrs["meshsize_m"] == 5.0
        assert ds["voxels"].dims == ("y", "x", "z")

    def test_coordinates(self):
        from voxcity.exporter.netcdf import voxel_to_xarray_dataset
        grid = np.ones((3, 4, 2), dtype=np.int8)
        ds = voxel_to_xarray_dataset(grid, voxel_size_m=10.0)
        np.testing.assert_array_equal(ds.coords["y"].values, [0, 10, 20])
        np.testing.assert_array_equal(ds.coords["x"].values, [0, 10, 20, 30])
        np.testing.assert_array_equal(ds.coords["z"].values, [0, 10])

    def test_rectangle_vertices(self):
        from voxcity.exporter.netcdf import voxel_to_xarray_dataset
        verts = [(139.0, 35.0), (139.0, 36.0), (140.0, 36.0), (140.0, 35.0)]
        grid = np.ones((2, 2, 2), dtype=np.int8)
        ds = voxel_to_xarray_dataset(grid, 5.0, rectangle_vertices=verts)
        stored = json.loads(ds.attrs["rectangle_vertices_lonlat_json"])
        assert len(stored) == 4
        assert stored[0] == [139.0, 35.0]

    def test_no_vertices(self):
        from voxcity.exporter.netcdf import voxel_to_xarray_dataset
        grid = np.ones((2, 2, 2), dtype=np.int8)
        ds = voxel_to_xarray_dataset(grid, 5.0, rectangle_vertices=None)
        assert ds.attrs["rectangle_vertices_lonlat_json"] == ""

    def test_extra_attrs(self):
        from voxcity.exporter.netcdf import voxel_to_xarray_dataset
        grid = np.ones((2, 2, 2), dtype=np.int8)
        ds = voxel_to_xarray_dataset(grid, 5.0, extra_attrs={"foo": "bar"})
        assert ds.attrs["foo"] == "bar"

    def test_non_3d_raises(self):
        from voxcity.exporter.netcdf import voxel_to_xarray_dataset
        with pytest.raises(ValueError, match="3D"):
            voxel_to_xarray_dataset(np.zeros((4, 5)), 5.0)


class TestSaveVoxelNetcdf:
    def test_roundtrip(self, tmp_path):
        from voxcity.exporter.netcdf import save_voxel_netcdf
        import xarray as xr
        grid = np.zeros((3, 4, 5), dtype=np.int8)
        grid[1, 2, 3] = -3
        out = save_voxel_netcdf(grid, tmp_path / "test.nc", voxel_size_m=5.0)
        assert out.endswith("test.nc")
        ds = xr.open_dataset(out)
        np.testing.assert_array_equal(ds["voxels"].values, grid)
        ds.close()

    def test_creates_parent_dirs(self, tmp_path):
        from voxcity.exporter.netcdf import save_voxel_netcdf
        nested = tmp_path / "a" / "b" / "c" / "out.nc"
        grid = np.ones((2, 2, 2), dtype=np.int8)
        out = save_voxel_netcdf(grid, nested, 5.0)
        assert nested.exists()

    def test_with_vertices_and_extras(self, tmp_path):
        from voxcity.exporter.netcdf import save_voxel_netcdf
        import xarray as xr
        verts = [(0.0, 0.0), (0.0, 1.0)]
        grid = np.ones((2, 2, 2), dtype=np.int8)
        out = save_voxel_netcdf(
            grid, tmp_path / "v.nc", 5.0,
            rectangle_vertices=verts,
            extra_attrs={"note": "test"},
        )
        ds = xr.open_dataset(out)
        assert ds.attrs["note"] == "test"
        ds.close()


class TestNetCDFExporter:
    def test_export(self, tmp_path):
        from voxcity.exporter.netcdf import NetCDFExporter
        from voxcity.models import VoxCity, VoxelGrid, GridMetadata, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid
        meta = GridMetadata(crs="EPSG:4326", bounds=(139.0, 35.0, 140.0, 36.0), meshsize=5.0)
        vg = VoxelGrid(classes=np.ones((3, 4, 2), dtype=np.int8), meta=meta)
        bg = BuildingGrid(heights=np.zeros((3, 4)), min_heights=np.empty((3, 4), dtype=object), ids=np.zeros((3, 4)), meta=meta)
        lc = LandCoverGrid(classes=np.zeros((3, 4), dtype=np.int8), meta=meta)
        dg = DemGrid(elevation=np.zeros((3, 4)), meta=meta)
        cg = CanopyGrid(top=np.zeros((3, 4)), meta=meta)
        vc = VoxCity(voxels=vg, buildings=bg, land_cover=lc, dem=dg, tree_canopy=cg)
        vc.extras["rectangle_vertices"] = [(139.0, 35.0), (139.0, 36.0)]
        exporter = NetCDFExporter()
        result = exporter.export(vc, str(tmp_path), "voxcity")
        assert (tmp_path / "voxcity.nc").exists()

    def test_non_voxcity_raises(self, tmp_path):
        from voxcity.exporter.netcdf import NetCDFExporter
        exporter = NetCDFExporter()
        with pytest.raises(TypeError, match="VoxCity"):
            exporter.export("not_voxcity", str(tmp_path), "bad")
