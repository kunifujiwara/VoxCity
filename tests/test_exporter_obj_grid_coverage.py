"""Tests for exporter/obj.py – covers grid_to_obj and export_obj with VoxCity input."""

import numpy as np
import os
import tempfile
import pytest


class TestGridToObj:
    """Tests for grid_to_obj function (lines 525-672)."""

    def test_basic_export(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        dem = np.array([[10.0, 10.0], [10.0, 10.0]])
        grid_to_obj(values, dem, str(tmp_path), "grid_test", cell_size=5.0, offset=0.0)
        assert (tmp_path / "grid_test.obj").exists()
        assert (tmp_path / "grid_test.mtl").exists()

    def test_obj_has_vertices_and_faces(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dem = np.zeros_like(values)
        grid_to_obj(values, dem, str(tmp_path), "check", cell_size=1.0, offset=0.0)
        content = (tmp_path / "check.obj").read_text()
        assert "v " in content
        assert "f " in content
        assert "usemtl " in content

    def test_mtl_has_materials(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        dem = np.zeros_like(values)
        grid_to_obj(values, dem, str(tmp_path), "mtl_check", cell_size=1.0, offset=0.0)
        content = (tmp_path / "mtl_check.mtl").read_text()
        assert "newmtl " in content
        assert "Kd " in content

    def test_with_nan_values(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.array([[1.0, np.nan], [np.nan, 4.0]])
        dem = np.array([[5.0, 5.0], [5.0, 5.0]])
        grid_to_obj(values, dem, str(tmp_path), "nan_test", cell_size=1.0, offset=0.0)
        assert (tmp_path / "nan_test.obj").exists()

    def test_custom_colormap(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.array([[0.5, 1.5], [2.5, 3.5]])
        dem = np.zeros_like(values)
        grid_to_obj(values, dem, str(tmp_path), "cmap", cell_size=1.0, offset=0.0,
                     colormap_name="plasma")
        assert (tmp_path / "cmap.obj").exists()

    def test_custom_vmin_vmax(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.array([[1.0, 5.0], [3.0, 7.0]])
        dem = np.zeros_like(values)
        grid_to_obj(values, dem, str(tmp_path), "vrange", cell_size=1.0, offset=0.0,
                     vmin=0.0, vmax=10.0)
        assert (tmp_path / "vrange.obj").exists()

    def test_shape_mismatch_raises(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.ones((3, 3))
        dem = np.ones((2, 2))
        with pytest.raises(ValueError, match="same shape"):
            grid_to_obj(values, dem, str(tmp_path), "bad", cell_size=1.0, offset=0.0)

    def test_vmin_equals_vmax_raises(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.ones((2, 2)) * 5.0
        dem = np.zeros_like(values)
        with pytest.raises(ValueError, match="vmin and vmax"):
            grid_to_obj(values, dem, str(tmp_path), "eq", cell_size=1.0, offset=0.0)

    def test_invalid_colormap_raises(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        dem = np.zeros_like(values)
        with pytest.raises(ValueError, match="not recognized"):
            grid_to_obj(values, dem, str(tmp_path), "bad_cm", cell_size=1.0, offset=0.0,
                         colormap_name="not_a_real_colormap")

    def test_nonzero_offset(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        dem = np.array([[10.0, 20.0], [30.0, 40.0]])
        grid_to_obj(values, dem, str(tmp_path), "offset", cell_size=5.0, offset=100.0)
        assert (tmp_path / "offset.obj").exists()

    def test_alpha_transparency(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        dem = np.zeros_like(values)
        grid_to_obj(values, dem, str(tmp_path), "alpha", cell_size=1.0, offset=0.0,
                     alpha=0.5)
        content = (tmp_path / "alpha.mtl").read_text()
        assert "d 0.5" in content

    def test_num_colors(self, tmp_path):
        from voxcity.exporter.obj import grid_to_obj
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        dem = np.zeros_like(values)
        grid_to_obj(values, dem, str(tmp_path), "nc", cell_size=1.0, offset=0.0,
                     num_colors=8)
        assert (tmp_path / "nc.obj").exists()


class TestExportObjVoxCity:
    """Test export_obj accepting a VoxCity instance."""

    def test_voxcity_input(self, tmp_path):
        from voxcity.exporter.obj import export_obj
        from voxcity.models import VoxCity, VoxelGrid, GridMetadata, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid
        grid = np.zeros((4, 4, 4), dtype=np.int8)
        grid[1:3, 1:3, 0:2] = -3
        meta = GridMetadata(crs="EPSG:4326", bounds=(0, 0, 1, 1), meshsize=5.0)
        vg = VoxelGrid(classes=grid, meta=meta)
        bg = BuildingGrid(heights=np.zeros((4, 4)), min_heights=np.empty((4, 4), dtype=object), ids=np.zeros((4, 4)), meta=meta)
        lc = LandCoverGrid(classes=np.zeros((4, 4), dtype=np.int8), meta=meta)
        dg = DemGrid(elevation=np.zeros((4, 4)), meta=meta)
        cg = CanopyGrid(top=np.zeros((4, 4)), meta=meta)
        vc = VoxCity(voxels=vg, buildings=bg, land_cover=lc, dem=dg, tree_canopy=cg)
        export_obj(vc, str(tmp_path), "vc_test")
        assert (tmp_path / "vc_test.obj").exists()
        assert (tmp_path / "vc_test.mtl").exists()

    def test_multiple_materials(self, tmp_path):
        from voxcity.exporter.obj import export_obj
        grid = np.zeros((5, 5, 5), dtype=np.int8)
        grid[1, 1, 1] = -3  # building
        grid[3, 3, 3] = -2  # vegetation
        grid[0, 0, 0] = 1   # land cover
        export_obj(grid, str(tmp_path), "multi", voxel_size=1.0)
        content = (tmp_path / "multi.mtl").read_text()
        assert content.count("newmtl ") >= 3

    def test_all_axes_coverage(self, tmp_path):
        """A solid block ensures all 6 face directions get exposed boundary faces."""
        from voxcity.exporter.obj import export_obj
        grid = np.zeros((5, 5, 5), dtype=np.int8)
        grid[1:4, 1:4, 1:4] = -3
        export_obj(grid, str(tmp_path), "block", voxel_size=2.0)
        content = (tmp_path / "block.obj").read_text()
        assert content.count("f ") > 0
