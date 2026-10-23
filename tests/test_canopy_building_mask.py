"""Canopy is masked under building footprints, and CHMv2 is the default canopy."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voxcity.generator.pipeline import mask_canopy_under_buildings

CHMV2 = "High Resolution 1m Global Canopy Height Maps v2"


class TestMaskHelper:
    def test_zeroes_top_and_bottom_on_building_cells_only(self):
        bh = np.array([[0.0, 12.0], [3.0, 0.0]])
        top = np.full((2, 2), 9.0)
        bottom = np.full((2, 2), 4.0)
        new_top, new_bottom = mask_canopy_under_buildings(top, bottom, bh)
        np.testing.assert_array_equal(new_top, [[9.0, 0.0], [0.0, 9.0]])
        np.testing.assert_array_equal(new_bottom, [[4.0, 0.0], [0.0, 4.0]])

    def test_inputs_are_not_modified(self):
        bh = np.array([[5.0]])
        top, bottom = np.array([[9.0]]), np.array([[4.0]])
        mask_canopy_under_buildings(top, bottom, bh)
        assert top[0, 0] == 9.0 and bottom[0, 0] == 4.0

    def test_nan_building_height_is_not_a_building(self):
        bh = np.array([[np.nan]])
        new_top, _ = mask_canopy_under_buildings(np.array([[9.0]]), np.array([[4.0]]), bh)
        assert new_top[0, 0] == 9.0

    def test_missing_bottom_is_passed_through(self):
        new_top, new_bottom = mask_canopy_under_buildings(np.array([[9.0]]), None, np.array([[5.0]]))
        assert new_top[0, 0] == 0.0 and new_bottom is None

    def test_shape_mismatch_leaves_canopy_unchanged(self):
        top = np.ones((2, 3))
        new_top, _ = mask_canopy_under_buildings(top, None, np.ones((3, 2)))
        np.testing.assert_array_equal(new_top, top)


def _cfg(**overrides):
    d = dict(
        meshsize=1.0,
        rectangle_vertices=[(139.75, 35.68), (139.75, 35.69), (139.76, 35.69), (139.76, 35.68)],
        crs="EPSG:4326", land_cover_source="OpenStreetMap", building_source="OpenStreetMap",
        canopy_height_source=CHMV2, dem_source="Flat", output_dir="output", gridvis=False,
        parallel_download=False, remove_perimeter_object=None, trunk_height_ratio=0.3,
        land_cover_options={}, building_options={}, canopy_options={}, dem_options={},
    )
    d.update(overrides)
    return SimpleNamespace(**d)


def _run_pipeline(tmp_path, **run_kwargs):
    """Run the sequential pipeline with stubbed sources; return the canopy the voxelizer got."""
    n = 6
    bh = np.zeros((n, n))
    bh[:, :3] = 10.0                       # west half is building
    bmin = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            bmin[i, j] = []
    with patch("voxcity.generator.pipeline.Voxelizer") as vox_cls, \
         patch("voxcity.generator.pipeline.DemSourceFactory") as dem_f, \
         patch("voxcity.generator.pipeline.CanopySourceFactory") as can_f, \
         patch("voxcity.generator.pipeline.BuildingSourceFactory") as bld_f, \
         patch("voxcity.generator.pipeline.LandCoverSourceFactory") as lc_f, \
         patch("voxcity.generator.grids.get_last_effective_land_cover_source", return_value="OpenStreetMap"):
        lc_f.create.return_value = MagicMock(build_grid=MagicMock(return_value=np.ones((n, n), dtype=np.int8)))
        bld_f.create.return_value = MagicMock(build_grids=MagicMock(return_value=(bh, bmin, np.zeros((n, n), int), None)))
        can_f.create.return_value = MagicMock(build_grids=MagicMock(return_value=(np.full((n, n), 8.0), np.full((n, n), 3.0))))
        dem_f.create.return_value = MagicMock(build_grid=MagicMock(return_value=np.zeros((n, n))))
        vox = MagicMock()
        vox.generate_combined.return_value = np.zeros((n, n, 4), dtype=np.int8)
        vox_cls.return_value = vox

        from voxcity.generator.pipeline import VoxCityPipeline
        cfg = _cfg(output_dir=str(tmp_path))
        VoxCityPipeline(meshsize=1.0, rectangle_vertices=cfg.rectangle_vertices).run(cfg, **run_kwargs)
        kw = vox.generate_combined.call_args.kwargs
    return kw["tree_grid_ori"], kw["canopy_bottom_height_grid_ori"]


def test_pipeline_masks_canopy_under_buildings_by_default(tmp_path):
    top, bottom = _run_pipeline(tmp_path)
    assert np.all(top[:, :3] == 0) and np.all(bottom[:, :3] == 0)
    assert np.all(top[:, 3:] == 8.0) and np.all(bottom[:, 3:] == 3.0)


def test_pipeline_mask_can_be_switched_off(tmp_path):
    top, _ = _run_pipeline(tmp_path, mask_canopy_under_buildings=False)
    assert np.all(top == 8.0)


@pytest.mark.parametrize(
    "verts",
    [
        [(-122.42, 37.77), (-122.42, 37.78), (-122.41, 37.78), (-122.41, 37.77)],   # USA
        [(139.75, 35.68), (139.75, 35.69), (139.76, 35.69), (139.76, 35.68)],       # Japan
        [(13.366, 52.513), (13.366, 52.518), (13.374, 52.518), (13.374, 52.513)],   # Germany
        [(36.80, -1.29), (36.80, -1.28), (36.81, -1.28), (36.81, -1.29)],           # Kenya
    ],
)
def test_auto_selection_picks_chmv2_everywhere(verts):
    from voxcity.generator.api import auto_select_data_sources
    assert auto_select_data_sources(verts)["canopy_height_source"] == CHMV2
